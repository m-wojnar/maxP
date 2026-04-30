"""timm model registry and ParametrizedModule wrappers for vision experiments."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Attention, PatchEmbed, maybe_add_mask, resolve_self_attn_mask
from timm.layers.format import Format, nchw_to

from maxp import ParametrizedModule


@dataclass(frozen=True, slots=True)
class ScaleConfig:
    model_name: str
    family: str
    image_size: int
    drop_path_rate: float = 0.0
    hidden: int = 0
    depth: int = 0
    dropout: float = 0.0


SCALE_CONFIGS: dict[str, ScaleConfig] = {
    "debug": ScaleConfig(
        model_name="vit_tiny_patch16_224", family="vit", image_size=224, drop_path_rate=0.0
    ),
    "vit-s": ScaleConfig(
        model_name="vit_small_patch16_224", family="vit", image_size=224, drop_path_rate=0.1
    ),
    "vit-b": ScaleConfig(
        model_name="vit_base_patch16_224", family="vit", image_size=224, drop_path_rate=0.2
    ),
    "vit-l": ScaleConfig(
        model_name="vit_large_patch16_224", family="vit", image_size=224, drop_path_rate=0.4
    ),
    "mlp-s": ScaleConfig(
        model_name="mlp", family="mlp", image_size=224, hidden=256, depth=4, dropout=0.0
    ),
    "mlp-m": ScaleConfig(
        model_name="mlp", family="mlp", image_size=224, hidden=512, depth=6, dropout=0.1
    ),
    "mlp-b": ScaleConfig(
        model_name="mlp", family="mlp", image_size=224, hidden=1024, depth=8, dropout=0.2
    ),
    "mlp-l": ScaleConfig(
        model_name="mlp", family="mlp", image_size=224, hidden=2048, depth=8, dropout=0.3
    ),
}


class MLPBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.linear = nn.Linear(hidden, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.linear(self.norm(x))))
    

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


class ScalableMLP(nn.Module):
    def __init__(
        self,
        hidden: int,
        depth: int,
        num_classes: int,
        dropout: float = 0.0,
        image_size: int = 224,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        in_features = 3 * patch_size * patch_size

        self.patchify = Patchify(self.patch_size)
        self.embed = nn.Linear(in_features, hidden)
        self.blocks = nn.ModuleList([MLPBlock(hidden, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, num_classes)

    def set_grad_checkpointing(self, enable: bool = False) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patchify(x)
        x = self.embed(x)
        x = x.mean(dim=1)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return self.head(x)


class _ScaledAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        d_model = kwargs.get("dim", args[0] if len(args) > 0 else None)
        num_heads = kwargs.get("num_heads", args[1] if len(args) > 1 else None)
        head_dim = kwargs.get("attn_head_dim", args[2] if len(args) > 2 else d_model // num_heads)
        
        self.scale = 1.0 / head_dim
        self.qk_score = ParametrizedModule(
            lambda q, k: (q, k), width_dim=head_dim,
            layer_type="readout", scale_output=False
        )
    
    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.qk_score(q, k)
        q, k = self.q_norm(q), self.k_norm(k)

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

        x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        x = self.norm(x)
        x = self.proj(x)
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

    if cfg.family == "mlp":
        return ScalableMLP(
            hidden=cfg.hidden,
            depth=cfg.depth,
            num_classes=num_classes,
            dropout=cfg.dropout,
            image_size=img_size,
        )

    kwargs: dict = {
        "pretrained": False,
        "num_classes": num_classes,
        "drop_path_rate": cfg.drop_path_rate,
        "img_size": img_size,
        "attn_layer": _ScaledAttention,
        "embed_layer": _LinearPatchEmbed,
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
        
        _wrap_module(attn, "qkv", embed_dim, "hidden")
        _wrap_module(attn, "proj", embed_dim, "hidden")
        
        _wrap_module(ffn, "fc1", block.mlp.fc1.in_features, "hidden")
        _wrap_module(ffn, "fc2", block.mlp.fc2.in_features, "hidden")
    
    _wrap_module(model, "patch_embed.proj", embed_dim, "embedding")
    # _wrap_module(model, "pos_embed", embed_dim, "embedding")
    # _wrap_module(model, "cls_token", embed_dim, "embedding")
    _wrap_module(model, "head", embed_dim, "readout", a=0.0)


def _install_mlp_wrappers(model: ScalableMLP) -> None:
    hidden = model.embed.out_features
    
    for block in model.blocks:
        _wrap_module(block, "linear", hidden, "hidden")

    _wrap_module(model, "embed", hidden, "embedding")
    _wrap_module(model, "head", hidden, "readout", a=0.0)


def install_pm_wrappers(model: nn.Module):
    if isinstance(model, ScalableMLP):
        _install_mlp_wrappers(model)
    elif "visiontransformer" in model.__class__.__name__.lower():
        _install_vit_wrappers(model)
    else:
        raise TypeError(
            f"Unsupported model type: {model.__class__.__name__}. Expected ViT or ScalableMLP."
        )


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
