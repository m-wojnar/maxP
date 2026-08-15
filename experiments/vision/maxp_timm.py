"""timm model registry and ParametrizedModule wrappers for vision experiments."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Attention, PatchEmbed, maybe_add_mask, resolve_self_attn_mask
from timm.layers.format import Format, nchw_to
from timm.models.vision_transformer import VisionTransformer

from maxp import ParametrizedModule


# --- LLaMA-3-matched block internals (RMSNorm + SwiGLU + split-qkv) -----------
# This ViT mirrors the LM (LLaMA-3) experiment's transformer body so the two
# width-ladder runs differ only in {dataset, modality, task}, not architecture.

def compute_ffn_hidden_dim(dim: int, multiple_of: int = 256,
                           ffn_dim_multiplier: float = 1.3) -> int:
    """SwiGLU hidden dim, identical to LM's torchtitan compute_ffn_hidden_dim."""
    hidden_dim = int(2 * 4 * dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


class _SwiGLU(nn.Module):
    """SwiGLU FFN with SEPARATE w1/w2/w3 (mirrors LM FeedForward).

    forward = w2(silu(w1(x)) * w3(x)). Separate projections (not packed) so each
    gets its own ParametrizedModule and a per-type alignment key (w1/w2/w3).
    Accepts the kwargs timm's Block passes to ``mlp_layer`` but computes its own
    hidden dim so it stays matched to LM regardless of timm's mlp_ratio.
    """

    def __init__(self, in_features: int, hidden_features: int | None = None,
                 act_layer=None, bias: bool = True, drop: float = 0.0, **kwargs) -> None:
        super().__init__()
        hidden = compute_ffn_hidden_dim(in_features)
        self.w1 = nn.Linear(in_features, hidden, bias=bias)
        self.w3 = nn.Linear(in_features, hidden, bias=bias)
        self.w2 = nn.Linear(hidden, in_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# Attention logit scale knob. head_dim is held constant (64) across the width
# ladder, so this is a width-independent constant — BOTH options give a flat
# coord-check (verified) and neither affects muP width-transfer. We use
# "inv_sqrt" = 1/sqrt(head_dim) to match the LM (LLaMA-3) run and because it
# leaves the attn-output projection better-conditioned than the tiny-temperature
# "inv_head_dim" = 1/head_dim (the previous vision value).
_ATTN_SCALE_MODE = "inv_sqrt"


def set_attn_scale_mode(mode: str) -> None:
    global _ATTN_SCALE_MODE
    assert mode in ("inv_head_dim", "inv_sqrt"), mode
    _ATTN_SCALE_MODE = mode


def _attn_scale(head_dim: int) -> float:
    if _ATTN_SCALE_MODE == "inv_sqrt":
        return 1.0 / math.sqrt(head_dim)
    return 1.0 / head_dim


@dataclass(frozen=True, slots=True)
class ScaleConfig:
    model_name: str
    family: str
    image_size: int
    drop_path_rate: float = 0.0
    hidden: int = 0
    depth: int = 0
    dropout: float = 0.0
    embed_dim: int = 0
    num_heads: int = 0


# ViT width-only ladder (mirrors the Llama3 s1..s5 ladder):
#   width axis = embed_dim, with head_dim = 64 and depth = 12 held CONSTANT.
#   drop_path / dropout = 0 at every scale so the only thing varying is width
#   (regularization that scales with width would confound the muP transfer test).
SCALE_CONFIGS: dict[str, ScaleConfig] = {
    "debug": ScaleConfig(
        model_name="vit_base_patch16_224", family="vit", image_size=224,
        embed_dim=128, num_heads=2, depth=2,
    ),
    "s1": ScaleConfig(
        model_name="vit_base_patch16_224", family="vit", image_size=224,
        embed_dim=256, num_heads=4, depth=12,
    ),
    "s2": ScaleConfig(
        model_name="vit_base_patch16_224", family="vit", image_size=224,
        embed_dim=512, num_heads=8, depth=12,
    ),
    "s3": ScaleConfig(
        model_name="vit_base_patch16_224", family="vit", image_size=224,
        embed_dim=1024, num_heads=16, depth=12,
    ),
    "s4": ScaleConfig(
        model_name="vit_base_patch16_224", family="vit", image_size=224,
        embed_dim=2048, num_heads=32, depth=12,
    ),
    "s5": ScaleConfig(
        model_name="vit_base_patch16_224", family="vit", image_size=224,
        embed_dim=4096, num_heads=64, depth=12,
    ),
}


class Patchify(nn.Module):
    def __init__(self, patch_size: tuple[int, int]):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p_h, p_w = self.patch_size
        assert H % p_h == 0 and W % p_w == 0, "Image dimensions must be divisible by patch size."
        x = x.reshape(B, C, H // p_h, p_h, W // p_w, p_w)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, (H // p_h) * (W // p_w), C * p_h * p_w)
        return x


class _ScaledAttention(Attention):
    """MHA with SEPARATE wq/wk/wv/wo (mirrors LM), no QK-norm.

    timm's fused ``qkv``/``proj`` are replaced by individually-wrapped
    projections so each gets a per-type alignment key. QK-norm is omitted (LM
    has none; timm's qk_norm already defaults off). Logit scale via the
    coord-check-selected knob (see ``set_attn_scale_mode``).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim = self.qkv.in_features
        qkv_bias = self.qkv.bias is not None
        proj_bias = self.proj.bias is not None
        del self.qkv
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.wo = nn.Linear(dim, dim, bias=proj_bias)
        del self.proj
        self.scale = _attn_scale(self.head_dim)

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,
    ) -> torch.Tensor:
        B, N, C = x.shape
        q = self.wq(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
                is_causal=is_causal,
                scale=self.scale,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn_bias = resolve_self_attn_mask(N, attn, attn_mask, is_causal)
            attn = maybe_add_mask(attn, attn_bias)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.wo(x)
        x = self.proj_drop(x)
        return x


class _LinearPatchEmbed(PatchEmbed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        patch_dim = self.proj.in_channels * self.patch_size[0] * self.patch_size[1]
        self.proj = nn.Linear(patch_dim, self.proj.out_channels)
        self.patchify = Patchify(self.patch_size)

    def forward(self, x):
        _, _, H, W = x.shape
        
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
            
        x = self.patchify(x)
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(x.size(0), -1, H // self.patch_size[0], W // self.patch_size[1])
        
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        
        x = self.norm(x)
        return x


def create_model(
    *,
    scale: str,
    num_classes: int,
    image_size: int | None = None,
) -> nn.Module:
    cfg = SCALE_CONFIGS.get(scale)
    img_size = image_size if image_size is not None else cfg.image_size

    kwargs: dict = {
        "pretrained": False,
        "num_classes": num_classes,
        "drop_path_rate": cfg.drop_path_rate,
        "img_size": img_size,
        "embed_dim": cfg.embed_dim,
        "depth": cfg.depth,
        "num_heads": cfg.num_heads,
        "attn_layer": _ScaledAttention,
        "embed_layer": _LinearPatchEmbed,
        "norm_layer": nn.RMSNorm,   # LLaMA-3: RMSNorm (not LayerNorm)
        "mlp_layer": _SwiGLU,       # LLaMA-3: SwiGLU w1/w2/w3 (not GELU MLP)
    }
    return timm.create_model(cfg.model_name, **kwargs)


def _get_child(module: nn.Module, key: str) -> nn.Module:
    if key.isdigit():
        return module[int(key)]  # type: ignore[index]
    return getattr(module, key)


def _set_child(module: nn.Module, key: str, child: nn.Module) -> None:
    if key.isdigit():
        module[int(key)] = child  # type: ignore[index]
        return
    setattr(module, key, child)


def _resolve_parent_and_key(root: nn.Module, module_path: str) -> tuple[nn.Module, str]:
    parts = module_path.split(".")
    if not parts:
        raise ValueError("Empty module path.")
    parent = root
    for key in parts[:-1]:
        parent = _get_child(parent, key)
    return parent, parts[-1]


def _get_module(root: nn.Module, module_path: str) -> nn.Module:
    parent, key = _resolve_parent_and_key(root, module_path)
    return _get_child(parent, key)


def _wrap_module(model: nn.Module, module_path: str, width_dim: int, layer_type: str, **pm_kw) -> bool:
    parent, key = _resolve_parent_and_key(model, module_path)
    module = _get_child(parent, key)
    if isinstance(module, ParametrizedModule):
        return False
    wrapped = ParametrizedModule(module, width_dim=width_dim, layer_type=layer_type, **pm_kw)
    _set_child(parent, key, wrapped)
    return True


def _install_vit_wrappers(model: nn.Module):
    embed_dim = int(getattr(model, "embed_dim"))

    for block in model.blocks:
        attn = block.attn
        ffn = block.mlp

        _wrap_module(attn, "wq", embed_dim, "hidden")
        _wrap_module(attn, "wk", embed_dim, "hidden")
        _wrap_module(attn, "wv", embed_dim, "hidden")
        _wrap_module(attn, "wo", embed_dim, "hidden")

        _wrap_module(ffn, "w1", ffn.w1.in_features, "hidden")
        _wrap_module(ffn, "w3", ffn.w3.in_features, "hidden")
        _wrap_module(ffn, "w2", ffn.w2.in_features, "hidden")

    _wrap_module(model, "patch_embed.proj", embed_dim, "embedding")
    # _wrap_module(model, "pos_embed", embed_dim, "embedding")
    # _wrap_module(model, "cls_token", embed_dim, "embedding")
    _wrap_module(model, "head", embed_dim, "readout", a=0.0)


def install_pm_wrappers(model: nn.Module):
    if isinstance(model, VisionTransformer):
        _install_vit_wrappers(model)
    else:
        raise TypeError(
            f"Unsupported model type: {model.__class__.__name__}. Expected a timm VisionTransformer."
        )


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
