"""MaxP model converter and post-optimizer-build hook for torchtitan integration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConverter
from torchtitan.protocols.module import Module

from maxp import ParametrizedModule, Parametrization


class LlamaParametrizedModule(Module, ParametrizedModule):
    """Marker class for ParametrizedModules in LLaMA-3 models."""
    pass


def _wrap(parent: nn.Module, attr: str, width_dim: int, layer_type: str, **pm_kw) -> None:
    """Replace parent.<attr> with a LlamaParametrizedModule in-place."""
    layer = getattr(parent, attr)
    setattr(parent, attr, LlamaParametrizedModule(layer, width_dim=width_dim, layer_type=layer_type, **pm_kw))


class _SDPAWrapper(Module):
    """Route q, k through a readout PM so the graph matches
    attn_score topology: r = min(min(r_q, r_k) + a, r_v).
    scale_output is set to False as PM returns a tuple that shouldn't be
    scalar-multiplied; the physical pm.scale = head_dim^-a is instead
    injected into SDPA's `scale` kwarg (applied to logits before softmax).
    """
    def __init__(self, inner, head_dim):
        super().__init__()
        self.inner = inner
        self.score = LlamaParametrizedModule(
            lambda q, k: (q, k), width_dim=head_dim,
            layer_type="readout", scale_output=False,
        )

    def forward(self, q, k, v, **kw):
        q, k = self.score(q, k)
        kw.pop("scale", None)
        return self.inner(q, k, v, scale=self.score.scale, **kw)


def install_pm_wrappers(model: nn.Module) -> None:
    """Install LlamaParametrizedModule wrappers on all LLaMA-3 layers in-place.

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

        attn.inner_attention = _SDPAWrapper(attn.inner_attention, head_dim)

        d_ff: int = ffn.w1.out_features
        _wrap(ffn, "w1", d_model, "hidden")
        _wrap(ffn, "w3", d_model, "hidden")
        _wrap(ffn, "w2", d_ff, "hidden")

    _wrap(model, "tok_embeddings", d_model, "embedding")
    _wrap(model, "output", d_model, "readout")


class MaxPConverter(Configurable, ModelConverter):
    """Installs PM wrappers and runs maxP parametrization before model compilation.

    convert() wraps layers and calls Parametrization (LP solve + weight reinit)
    so forward hooks are correctly traced by torch.compile.
    post_optimizer_build_fn() (returned by make_post_optimizer_build_fn) patches
    optimizer param_groups with per-layer LRs after optimizer.build().
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        method: str = "maxP"          # "maxP" | "mup-no" | "maxP-meas"
        lr_prefactor: float = 1e-3
        alignment_warmup: int = 10    # maxP / measure_only
        solve_interval: int = 100     # maxP / measure_only
        sample_size: int = 32         # maxP / measure_only
        c_ema: float = 0.0            # maxP only
        use_training_activations: bool = False  # maxP only
        measure_only: bool = False    # measure + log alignment, never touch LRs
        alignment_table: str | None = None  # JSON path; required for maxP-meas

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        model_compile_enabled: bool,
    ) -> None:
        self._cfg = config

    def convert(self, model: nn.Module) -> None:
        install_pm_wrappers(model)
        cfg = self._cfg
        if cfg.method not in ("maxP", "mup-no", "maxP-meas"):
            raise ValueError(f"Unknown method '{cfg.method}'")

        if cfg.measure_only and cfg.method == "maxP":
            raise ValueError("measure_only contradicts dynamic method 'maxP'; use 'mup-no'")

        alignment_overrides = None
        if cfg.method == "maxP-meas":
            if cfg.alignment_table is None:
                raise ValueError("method 'maxP-meas' requires alignment_table")
            alignment_overrides = _load_alignment_table(cfg.alignment_table)

        param = Parametrization(
            model,
            sample_input=_make_sample_input(model),
            alignment="no",
            lr_prefactor=cfg.lr_prefactor,
            alignment_overrides=alignment_overrides,
            warmup_steps=cfg.alignment_warmup,
            solve_interval=cfg.solve_interval,
            sample_size=cfg.sample_size,
            c_ema=cfg.c_ema,
            measure_only=cfg.measure_only,
            use_training_activations=cfg.use_training_activations,
        )
        model._maxp_param = param
        if cfg.method == "maxP" or cfg.measure_only:
            model._is_dynamic = True
            model._maxp_align = {
                name: (pm.align_z0_dW, pm.align_dZ_w0, pm.align_dZ_dW)
                for name, pm in param._pms
                if pm.weight is not None and pm.align_z0_dW is not None
            }

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]) -> None:
        pass


def _load_alignment_table(path: str) -> dict[str, tuple[float, float, float]]:
    """Load measured alignment from JSON written by export_alignment.py.

    Keys are exact PM names, leaf suffixes (e.g. "wo") or layer types;
    values are [align_z0_dW, align_dZ_w0, align_dZ_dW].
    """
    import json

    with open(path) as f:
        table = json.load(f)
    return {k: tuple(v) for k, v in table.items()}


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


def post_optimizer_build_fn(optimizers, model_parts: list[nn.Module], parallel_dims: ParallelDims) -> None:
    """Called by Trainer after init_weights() and optimizer.build().

    Re-runs the tensor-bound bits of Parametrization (weight init + per-layer
    param_groups) now that the model has been materialized off meta device,
    and splices the resulting groups into the optimizer.
    """
    for optimizer, model in zip(optimizers, model_parts):
        model._maxp_param.refresh(optimizer=optimizer)
