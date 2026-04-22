"""ParametrizedModule — marks an op as needing ABC parametrization."""

import torch
import torch.nn as nn


class ParametrizedModule(nn.Module):
    """Marks an op as needing ABC parametrization.

    Wraps either an ``nn.Module`` (whose parameters are visible via the
    standard ``parameters()`` walk) or a bare callable (e.g. ``lambda q, k:
    q @ k.T``, which has no learnable parameters).

    Args:
        module_or_fn: An ``nn.Module`` or bare callable to wrap.
        width_dim: Fan-in for this op (the dimension that scales with width).
        layer_type: ``"embedding"``, ``"hidden"``, or ``"readout"``.
        a: Optional override for the output multiplier exponent.
            If set, :class:`Parametrization` will use this value instead of
            looking up the default for ``layer_type``.  Useful for ops that
            need non-standard scaling (e.g. ``a=1.0`` for muP attention
            logits QK^T → 1/d scaling instead of the readout default 1/√d).
        b: Optional override for the init variance exponent.
            If set, :class:`Parametrization` will use this value instead of
            the default.
        c: Optional override for the learning rate exponent.
            If set, :class:`Parametrization` will use this value instead of
            solving for it via LP.
        scale_output: If ``False``, the solved multiplier is not physically
            applied to the output tensor in ``forward()``.  Useful for ops
            that handle scaling internally (like SDPA via the ``scale`` arg).

    Attributes:
        inner: The wrapped ``nn.Module``, or ``None`` for bare callables.
        scale: Output multiplier, set to ``width_dim ** (-a)`` by
            :class:`Parametrization`.
        align_z0_dW: Alignment of the z0 @ dW^T term, or ``None`` before parametrization.
        align_dZ_w0: Alignment of the dZ @ W0^T term, or ``None`` before parametrization.
        align_dZ_dW: Alignment of the dZ @ dW^T cross term, or ``None`` before parametrization.
    """

    def __init__(
        self,
        module_or_fn,
        width_dim: int,
        layer_type: str = "hidden",
        *,
        a: float | None = None,
        b: float | None = None,
        c: float | None = None,
        scale_output: bool = True,
    ):
        super().__init__()
        if isinstance(module_or_fn, nn.Module):
            self.inner = module_or_fn
        else:
            self._fn = module_or_fn
            self.inner = None
        self.width_dim = width_dim
        self.layer_type = layer_type
        self.scale = 1.0
        self.scale_output = scale_output

        # Per-PM (a, b, c) overrides — Parametrization respects these if set
        self.a: float | None = a
        self.b: float | None = b
        self.c: float | None = c

        # Alignment (set by Parametrization from preset or measurement)
        self.align_z0_dW: float | None = None   # alignment of z0 @ dW^T term
        self.align_dZ_w0: float | None = None   # alignment of dZ @ W0^T term
        self.align_dZ_dW: float | None = None   # alignment of dZ @ dW^T term
        # Initial snapshot for alignment measurement
        self._z0: torch.Tensor | None = None
        self._w0: torch.Tensor | None = None
        # Seed-based W0 regeneration (alternative to storing _w0)
        self._w0_seed: int | None = None
        self._w0_std: float | None = None

    @property
    def weight(self) -> torch.nn.Parameter | None:
        """The primary weight parameter for this op, or None for activation-only ops."""
        if self.inner is None:
            return None
        w = getattr(self.inner, "weight", None)
        if isinstance(w, torch.nn.Parameter):
            return w
        return None

    def forward(self, *args, **kwargs):
        if self.inner is not None:
            out = self.inner(*args, **kwargs)
        else:
            out = self._fn(*args, **kwargs)
        
        if self.scale_output:
            return self.scale * out
        else:
            return out
