"""
Alignment computation for dynamic LP re-solve.

Computes alignment metrics between initial and current weights/activations
for a single layer.  These metrics measure how efficiently the output
decomposition terms scale with width:

    y = z @ W^T = z0 @ W0^T + z0 @ dW^T + dz @ W0^T + dz @ dW^T

- align_z0_dW: alignment of z0 @ dW^T  (weight-change term)
- align_dZ_w0: alignment of dZ @ W0^T  (activation-change term)
- align_dZ_dW: alignment of dZ @ dW^T  (cross term)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _rms(x: Tensor, dim: int | None = None) -> Tensor:
    """Root-mean-square norm (float64, with epsilon for stability)."""
    x = x.double()
    if dim is None:
        return torch.sqrt(torch.mean(x ** 2) + 1e-32)
    return torch.sqrt(torch.mean(x ** 2, dim=dim) + 1e-32)


def _sanitize(v: float) -> float:
    """Replace inf/nan with safe defaults."""
    if math.isnan(v):
        return 0.0
    if math.isinf(v):
        return 1.0 if v > 0 else 0.0
    return v


def compute_alignment(
    z0: Tensor,
    w0: Tensor,
    z: Tensor,
    w: Tensor,
    fan_in: int,
) -> tuple[float, float, float]:
    """Compute (align_z0_dW, align_dZ_w0, align_dZ_dW) alignment metrics for one layer.

    Each metric measures how efficiently one term of the output decomposition
    scales with width, using log-scale RMS norms.

    Args:
        z0: Initial input activations, shape ``(batch, fan_in)``.
        w0: Initial weight matrix, shape ``(fan_out, fan_in)``.
        z: Current input activations, same shape as *z0*.
        w: Current weight matrix, same shape as *w0*.
        fan_in: Width dimension (used as log base).

    Returns:
        Tuple ``(align_z0_dW, align_dZ_w0, align_dZ_dW)``.
        Values are sanitised (no inf/nan).
    """
    # Work in float64 for precision
    z0 = z0.detach().double()
    w0 = w0.detach().double()
    z = z.detach().double()
    w = w.detach().double()

    dz = z - z0
    dw = w - w0

    log_base = math.log(max(fan_in, 2))  # avoid log(1)=0

    # Norms of components
    z0_n = _rms(z0, dim=-1)    # (batch,)
    dz_n = _rms(dz, dim=-1)    # (batch,)
    dw_n = _rms(dw)            # scalar
    w0_n = _rms(w0)            # scalar

    # Check magnitudes of actual changes (not epsilon-padded norms)
    dw_mag = torch.abs(dw).max().item()
    dz_mag = torch.abs(dz).max().item()

    align_z0_dW = 0.0
    align_dZ_w0 = 0.0
    align_dZ_dW = 0.0

    if dw_mag > 1e-12 or dz_mag > 1e-12:
        if dw_mag > 1e-12:
            o = z0 @ dw.T
            o_n = _rms(o, dim=-1)
            align_z0_dW = torch.mean(
                (torch.log(o_n) - torch.log(z0_n * dw_n)) / log_base
            ).item()

        if dz_mag > 1e-12 and w0_n.item() > 1e-16:
            o = dz @ w0.T
            o_n = _rms(o, dim=-1)
            align_dZ_w0 = torch.mean(
                (torch.log(o_n) - torch.log(dz_n * w0_n)) / log_base
            ).item()

        if dz_mag > 1e-12 and dw_mag > 1e-12:
            o = dz @ dw.T
            o_n = _rms(o, dim=-1)
            align_dZ_dW = torch.mean(
                (torch.log(o_n) - torch.log(dz_n * dw_n)) / log_base
            ).item()

    return _sanitize(align_z0_dW), _sanitize(align_dZ_w0), _sanitize(align_dZ_dW)
