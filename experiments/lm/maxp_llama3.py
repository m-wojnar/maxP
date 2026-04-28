"""MaxP-compatible LLaMA-3 model definition and scale configs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, Linear, RMSNorm, RoPE, compute_ffn_hidden_dim
from torchtitan.models.common.config_utils import get_attention_config, make_ffn_config, make_gqa_config
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.llama3 import parallelize_llama as _parallelize_llama
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from maxp_converter import post_optimizer_build_fn


# Weight inits — mirrors llama3 upstream defaults
_LINEAR_INIT = {"weight": partial(nn.init.trunc_normal_, std=0.02), "bias": nn.init.zeros_}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int) -> dict:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _build_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    hidden_dim: int,
    attn_backend: str = "sdpa",
) -> list:
    inner_attention, mask_type = get_attention_config(attn_backend)
    return [
        Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
            attention=make_gqa_config(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                wqkv_param_init=_LINEAR_INIT,
                wo_param_init=_depth_init(layer_id),
                inner_attention=inner_attention,
                mask_type=mask_type,
                rope_backend="complex",
            ),
            feed_forward=make_ffn_config(
                dim=dim,
                hidden_dim=hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            ),
        )
        for layer_id in range(n_layers)
    ]


def _make_model_config(
    *,
    dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    vocab_size: int = 128256,
    attn_backend: str = "sdpa",
) -> Llama3Model.Config:
    hidden_dim = compute_ffn_hidden_dim(dim, multiple_of=256, ffn_dim_multiplier=1.3)
    return Llama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=False,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=hidden_dim,
            attn_backend=attn_backend,
        ),
    )


def parallelize_llama(model, **kwargs):
    # Force reshard_after_forward="always" so no_grad forwards used for
    # alignment measurement don't leave norm+output layers unsharded.
    parallelism = kwargs.get("parallelism")
    if parallelism is not None:
        parallelism.fsdp_reshard_after_forward = "always"
    return _parallelize_llama(model, **kwargs)


# Scale definitions: (dim, n_layers, n_heads, n_kv_heads[, vocab_size])
# Approximate parameter counts (no weight tying, vocab=128256):
#   debug: tiny 4-layer (CPU smoke tests, vocab=2048)
#   s1:  ~21M   s2: ~81M   s3: ~218M   s4: ~1.09B   s5: ~2.71B
SCALE_CONFIGS: dict[str, dict] = {
    "debug": dict(dim=256, n_layers=4, n_heads=4, n_kv_heads=2),
    "s1": dict(dim=512, n_layers=6, n_heads=8, n_kv_heads=4),
    "s2": dict(dim=768, n_layers=10, n_heads=12, n_kv_heads=4),
    "s3": dict(dim=1024, n_layers=16, n_heads=16, n_kv_heads=4),
    "s4": dict(dim=2048, n_layers=20, n_heads=16, n_kv_heads=4),
    "s5": dict(dim=2560, n_layers=32, n_heads=20, n_kv_heads=4),
}


def maxp_model_registry(scale: str, method: str, attn_backend: str = "sdpa") -> ModelSpec:
    """Build a ModelSpec for maxP LLaMA-3 training.

    Parametrization config (alignment_warmup, solve_interval, etc.) lives in
    MaxPConverter.Config and is passed via Trainer.Config.model_converters.

    Args:
        scale: One of "debug", "s1" … "s5".
        method: "maxP" (dynamic), "mup-full" (static, full align), or
            "mup-no" (static, no align).
        attn_backend: Attention backend ("sdpa", "flex", "varlen").
    """
    if scale not in SCALE_CONFIGS:
        raise ValueError(f"Unknown scale '{scale}'. Choose from {list(SCALE_CONFIGS)}")

    kwargs = SCALE_CONFIGS[scale].copy()
    model_config = _make_model_config(attn_backend=attn_backend, **kwargs)

    return ModelSpec(
        name="maxp_llama3",
        flavor=f"{scale}_{method}",
        model=model_config,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=post_optimizer_build_fn,
        state_dict_adapter=Llama3StateDictAdapter,
    )


def compute_steps(
    scale: str,
    seq_len: int,
    local_batch_size: int,
    world_size: int,
    token_multiplier: int = 20,
) -> int:
    """Return training steps for token_multiplier × non-embed-params tokens.

    Instantiates the model on meta device (zero memory) to count parameters.
    Non-embed params = total params minus tok_embeddings and LM head.
    """
    import torch

    cfg = SCALE_CONFIGS[scale]
    vocab_size = cfg.get("vocab_size", 128256)
    dim = cfg["dim"]
    arch_kwargs = {k: v for k, v in cfg.items() if k != "vocab_size"}
    model_config = _make_model_config(vocab_size=vocab_size, **arch_kwargs)
    with torch.device("meta"):
        model = Llama3Model(model_config)
    total = sum(p.numel() for p in model.parameters())
    non_embed = total - 2 * vocab_size * dim  # tok_embeddings + LM head
    tokens = token_multiplier * non_embed
    return math.ceil(tokens / (seq_len * local_batch_size * world_size))
