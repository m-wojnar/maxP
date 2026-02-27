"""
Alignment computation for dynamic LP re-solve.

Computes alignment metrics (alpha, omega, u) between initial and current
weights/activations for a single layer.  These metrics measure how
efficiently the output decomposition terms scale with width:

    y = z @ w^T = z_0 @ w_0^T + z_0 @ dw^T + dz @ w_0^T + dz @ dw^T

- alpha: alignment of z_0 @ dw^T  (weight-change term)
- omega: alignment of dz @ w_0^T  (activation-change term)
- u:     alignment of dz @ dw^T   (cross term)
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


def _rms(x: Tensor, dim: int | None = None) -> Tensor:
    """Root-mean-square norm (float64, with epsilon for stability)."""
    x = x.double()
    if dim is None:
        return torch.sqrt(torch.mean(x ** 2) + 1e-32)
    return torch.sqrt(torch.mean(x ** 2, dim=dim) + 1e-32)


def _spectral(x: Tensor) -> Tensor:
    """Spectral norm (largest singular value)."""
    return torch.linalg.norm(x.double(), 2)


def _l2(x: Tensor, dim: int | None = None) -> Tensor:
    """L2 norm (float64, with epsilon)."""
    x = x.double()
    if dim is None:
        return torch.sqrt(torch.sum(x ** 2) + 1e-32)
    return torch.sqrt(torch.sum(x ** 2, dim=dim) + 1e-32)


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
    norm_mode: str = "rms",
) -> tuple[float, float, float]:
    """Compute (alpha, omega, u) alignment metrics for one layer.

    Args:
        z0: Initial input activations, shape ``(batch, fan_in)``.
        w0: Initial weight matrix, shape ``(fan_out, fan_in)``.
        z: Current input activations, same shape as *z0*.
        w: Current weight matrix, same shape as *w0*.
        fan_in: Width dimension (used as log base in rms mode).
        norm_mode: ``"rms"`` (default) or ``"spectral"``.

    Returns:
        Tuple ``(alpha, omega, u)``.  Values are sanitised (no inf/nan).
    """
    norm_mode = norm_mode.lower().strip()
    if norm_mode not in {"rms", "spectral"}:
        raise ValueError(f"norm_mode must be 'rms' or 'spectral', got '{norm_mode}'")

    # Work in float64 for precision
    z0 = z0.detach().double()
    w0 = w0.detach().double()
    z = z.detach().double()
    w = w.detach().double()

    dz = z - z0
    dw = w - w0

    log_base = math.log(max(fan_in, 2))  # avoid log(1)=0

    if norm_mode == "spectral":
        vec_norm = _l2
        mat_norm = _spectral
        use_log = False
    else:  # rms
        vec_norm = _rms
        mat_norm = _rms
        use_log = True

    # Norms of components
    z0_n = vec_norm(z0, dim=-1)   # (batch,)
    dz_n = vec_norm(dz, dim=-1)   # (batch,)
    dw_n = mat_norm(dw)           # scalar
    w0_n = mat_norm(w0)           # scalar

    # Check magnitudes of actual changes (not epsilon-padded norms)
    dw_mag = torch.abs(dw).max().item()
    dz_mag = torch.abs(dz).max().item()

    alpha = 0.0
    omega = 0.0
    u = 0.0

    if dw_mag > 1e-12 or dz_mag > 1e-12:
        # Alpha: alignment of z_0 @ dw^T
        if dw_mag > 1e-12:
            o_alpha = z0 @ dw.T
            o_alpha_n = vec_norm(o_alpha, dim=-1)
            if use_log:
                alpha = torch.mean(
                    (torch.log(o_alpha_n) - torch.log(z0_n * dw_n)) / log_base
                ).item()
            else:
                alpha = torch.mean(o_alpha_n / (z0_n * dw_n)).item()

        # Omega: alignment of dz @ w0^T
        if dz_mag > 1e-12 and w0_n.item() > 1e-16:
            o_omega = dz @ w0.T
            o_omega_n = vec_norm(o_omega, dim=-1)
            if use_log:
                omega = torch.mean(
                    (torch.log(o_omega_n) - torch.log(dz_n * w0_n)) / log_base
                ).item()
            else:
                omega = torch.mean(o_omega_n / (dz_n * w0_n)).item()

        # U: alignment of dz @ dw^T
        if dz_mag > 1e-12 and dw_mag > 1e-12:
            o_u = dz @ dw.T
            o_u_n = vec_norm(o_u, dim=-1)
            if use_log:
                u = torch.mean(
                    (torch.log(o_u_n) - torch.log(dz_n * dw_n)) / log_base
                ).item()
            else:
                u = torch.mean(o_u_n / (dz_n * dw_n)).item()

    return _sanitize(alpha), _sanitize(omega), _sanitize(u)


def compute_alignments_for_pms(
    snapshots: list[tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]] | None],
    fan_ins: list[int],
    norm_mode: str = "rms",
) -> tuple[list[float], list[float], list[float]]:
    """Compute alignment for a list of PM snapshots.

    Args:
        snapshots: List of ``((z0, w0), (z, w))`` per PM, or ``None``
            for activation-only PMs.
        fan_ins: Fan-in per PM.
        norm_mode: ``"rms"`` or ``"spectral"``.

    Returns:
        Tuple of ``(alpha_list, omega_list, u_list)``.
    """
    alpha_list: list[float] = []
    omega_list: list[float] = []
    u_list: list[float] = []

    for i, snap in enumerate(snapshots):
        if snap is None:
            # Default: full alignment assumption for skipped PMs
            alpha_list.append(1.0)
            omega_list.append(0.5)
            u_list.append(1.0)
            continue
        (z0, w0), (z, w) = snap
        a, o, uu = compute_alignment(z0, w0, z, w, fan_ins[i], norm_mode)
        alpha_list.append(a)
        omega_list.append(o)
        u_list.append(uu)

    return alpha_list, omega_list, u_list
