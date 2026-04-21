"""MaxP model converter and post-optimizer-build hook for torchtitan integration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConverter

from maxp import ParametrizedModule, Parametrization


def _wrap(parent: nn.Module, attr: str, width_dim: int, layer_type: str, **pm_kw) -> None:
    """Replace parent.<attr> with a ParametrizedModule in-place."""
    layer = getattr(parent, attr)
    setattr(
        parent,
        attr,
        ParametrizedModule(layer, width_dim=width_dim, layer_type=layer_type, **pm_kw),
    )


def install_pm_wrappers(model: nn.Module) -> None:
    """Install ParametrizedModule wrappers on all LLaMA-3 layers in-place.

    Purely structural — safe to call on a meta-device model.
    Wraps attention projections, FFN weights, token embedding, and LM head.
    The SDPA inner_attention is annotated as a=0.0 readout to avoid
    double-scaling (SDPA already divides by sqrt(head_dim)).
    """
    from torchtitan.models.llama3.model import Llama3Model
    from torchtitan.models.common.attention import FusedQKVLinear

    assert isinstance(model, Llama3Model), f"Expected Llama3Model, got {type(model)}"
    cfg = model.config
    d_model: int = cfg.dim
    head_dim: int = next(iter(model.layers.values())).attention.head_dim

    for block in model.layers.values():
        attn = block.attention
        ffn = block.feed_forward

        qkv = attn.qkv_linear
        if isinstance(qkv, FusedQKVLinear):
            _wrap(qkv, "wqkv", d_model, "hidden")
        else:
            _wrap(qkv, "wq", d_model, "hidden")
            _wrap(qkv, "wk", d_model, "hidden")
            _wrap(qkv, "wv", d_model, "hidden")

        _wrap(attn, "wo", d_model, "hidden")

        # SDPA already scales by 1/sqrt(head_dim), so set a=0 to avoid double-scaling
        attn.inner_attention = ParametrizedModule(
            attn.inner_attention,
            width_dim=head_dim,
            layer_type="readout",
            a=0.0,
        )

        d_ff: int = ffn.w1.out_features
        _wrap(ffn, "w1", d_model, "hidden")
        _wrap(ffn, "w3", d_model, "hidden")
        _wrap(ffn, "w2", d_ff, "hidden")

    _wrap(model, "tok_embeddings", d_model, "embedding")
    _wrap(model, "output", d_model, "readout")


class MaxPConverter(Configurable, ModelConverter):
    """ModelConverter that installs ParametrizedModule wrappers on a LLaMA-3 model.

    install_pm_wrappers() is called during convert() while the model is still
    on meta device — purely structural, no tensor operations.
    Parametrization() (LP solve + weight reinit) runs later in post_optimizer_build_fn,
    after init_weights() has materialized real tensors.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        model_compile_enabled: bool,
    ) -> None:
        pass  # stateless

    def convert(self, model: nn.Module) -> None:
        install_pm_wrappers(model)

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]) -> None:
        pass  # static maxP: no dynamic re-solve after optimizer steps


def _make_sample_input(model: nn.Module) -> torch.Tensor | None:
    """Build a deterministic sample input for DAG tracing.

    Returns None if vocab_size cannot be determined (falls back to chain graph).
    Uses a fixed seed so all ranks produce identical inputs (no collective needed).
    """
    cfg = getattr(model, "config", None)
    vocab_size = getattr(cfg, "vocab_size", None)
    if vocab_size is None:
        return None
    try:
        device = next(iter(model.parameters())).device
    except StopIteration:
        device = torch.device("cpu")
    gen = torch.Generator().manual_seed(0)
    return torch.randint(0, vocab_size, (1, 16), generator=gen).to(device)


def make_post_optimizer_build_fn(
    method: str,
    lr_prefactor: float,
    alignment_warmup: int = 10,
    solve_interval: int = 100,
    sample_size: int = 32,
    c_ema: float = 0.0,
) -> Callable:
    """Return a post_optimizer_build_fn for maxP initialization.

    The returned function is called by Trainer after init_weights() and
    optimizer.build(), but before lr_scheduler.build().  It runs
    Parametrization (LP solve + weight reinit) and replaces each inner
    AdamW's param_groups with maxP per-layer groups, so LRSchedulersContainer
    records our per-layer initial_lr values correctly.

    For method="maxP" (dynamic), the Parametrization object is stored on the
    model as ``_maxp_param`` so MaxPTrainer.train_step can call param.step()
    each iteration.  For "mup-full"/"mup-no" (static), LP is solved once
    and never updated.

    Args:
        method: "maxP" (dynamic), "mup-full" (static, full align), or
            "mup-no" (static, no align).
        lr_prefactor: Base LR multiplier (e.g. 1e-3).
        alignment_warmup: Steps before first dynamic re-solve (maxP only).
        solve_interval: Re-solve LP every N steps (maxP only).
        sample_size: Number of sequences for alignment measurement (maxP only).
        c_ema: EMA smoothing for c values toward LP targets (maxP only).
    """
    is_dynamic = method == "maxP"
    alignment = "no" if method == "mup-no" else "full"

    def fn(
        optimizers,
        model_parts: list[nn.Module],
        parallel_dims: ParallelDims,
    ) -> None:
        for i, model in enumerate(model_parts):
            sample_input = _make_sample_input(model)
            param = Parametrization(
                model,
                sample_input=sample_input,
                alignment=alignment,
                lr_prefactor=lr_prefactor,
                warmup_steps=alignment_warmup if is_dynamic else 0,
                solve_interval=solve_interval,
                sample_size=sample_size,
                c_ema=c_ema,
            )
            optimizer = optimizers.optimizers[i]
            non_lr_defaults = {k: v for k, v in optimizer.defaults.items() if k != "lr"}
            merged = [
                {"params": g["params"], "lr": g["lr"],
                 "layer_name": g.get("layer_name", "_other"), **non_lr_defaults}
                for g in param.param_groups
            ]
            # Replace the inner AdamW's param_groups in-place.
            # LRSchedulersContainer (built next) records these as initial_lr.
            optimizer.param_groups[:] = merged

            if is_dynamic:
                model._maxp_param = param

            model._maxp_align = {
                name: (pm.align_z0_dW, pm.align_dZ_w0, pm.align_dZ_dW)
                for name, pm in param._pms
                if pm.weight is not None and pm.align_z0_dW is not None
            }

    return fn
