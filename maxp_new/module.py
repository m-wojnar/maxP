"""ParametrizedModule — marks an op as needing ABC parametrization."""

import torch
import torch.nn as nn


class ParametrizedModule(nn.Module):
    """Marks an op as needing ABC parametrization.

    Wraps either an ``nn.Module`` (whose parameters are visible via the
    standard ``parameters()`` walk) or a bare callable (e.g. ``lambda q, k:
    q @ k.T``, which has no learnable parameters).

    Attributes:
        inner: The wrapped ``nn.Module``, or ``None`` for bare callables.
        width_dim: Fan-in for this op (the dimension that scales with width).
        layer_type: ``"embedding"``, ``"hidden"``, or ``"readout"``.
        scale: Output multiplier, set to ``width_dim ** (-a)`` by
            :class:`Parametrization`.
        alpha: Alignment of the z_0 @ dw^T term, or ``None`` before parametrization.
        omega: Alignment of the dz @ w_0^T term, or ``None`` before parametrization.
        u: Alignment of the dz @ dw^T cross term, or ``None`` before parametrization.
    """

    def __init__(self, module_or_fn, width_dim: int, layer_type: str = "hidden"):
        super().__init__()
        if isinstance(module_or_fn, nn.Module):
            self.inner = module_or_fn
        else:
            self._fn = module_or_fn
            self.inner = None
        self.width_dim = width_dim
        self.layer_type = layer_type
        self.scale = 1.0

        # Alignment (set by Parametrization from preset or measurement)
        self.alpha: float | None = None
        self.omega: float | None = None
        self.u: float | None = None
        # Initial snapshot for alignment measurement
        self._z0: torch.Tensor | None = None
        self._w0: torch.Tensor | None = None

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
        return self.scale * out
