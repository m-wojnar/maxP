"""
Alignment computation for maxP scheduler.

Computes alignment metrics (alpha, omega, u) between initial and current
weights/activations for each layer. These metrics measure how the network's
output decomposition changes during training.
"""

import numpy as np
import torch

from maxp.tracer import TraceWindow


def _rms_norm(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    x = x.double()
    if dim is None:
        return torch.sqrt(torch.mean(x ** 2) + 1e-32)
    return torch.sqrt(torch.mean(x ** 2, dim=dim) + 1e-32)


def _l2_norm(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    x = x.double()
    if dim is None:
        return torch.sqrt(torch.sum(x ** 2) + 1e-32)
    return torch.sqrt(torch.sum(x ** 2, dim=dim) + 1e-32)


def _spectral_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x.double(), 2)


def _resample_w0(shape: tuple[int, ...], bl: float, std_prefactor: float = 1.0) -> torch.Tensor:
    """
    Resample initial weights using ABC parametrization initialization.
    
    Args:
        shape: Weight tensor shape (out_features, in_features).
        bl: b_l exponent from ABC parametrization.
        std_prefactor: Initialization std multiplier (default 1.0).
    
    Returns:
        Resampled weight tensor.
    """

    n = shape[1]  # fan-in (width)
    var_l = n ** (-2 * bl)
    std = std_prefactor * (var_l ** 0.5)
    return torch.randn(shape, dtype=torch.float64) * std


def compute_alignment(
    window: TraceWindow,
    resample_w0: bool | None = None,
    std_prefactor: float = 1.0,
    norm_mode: str = "rms",
) -> tuple[list[float], list[float], list[float]]:
    """
    Compute alignment metrics for all layers.
    
    The output y of each layer can be decomposed as:
        y = z @ w^T = z_0 @ w_0^T + z_0 @ Δw^T + Δz @ w_0^T + Δz @ Δw^T
    
    The alignment metrics measure the "efficiency" of each term:
        - alpha: alignment of z_0 @ Δw^T term
        - omega: alignment of Δz @ w_0^T term  
        - u: alignment of Δz @ Δw^T term
    
    Args:
        window: TraceWindow containing initial and current layer states.
        resample_w0: Override window's resample_w0 setting. If True, resample
            w_0 from N(0, width^{-b_l}) instead of using stored initial weights.
        std_prefactor: Std multiplier for resampling (default 1.0).
    
    Returns:
        Tuple of (alpha, omega, u) where each is a list of floats,
        one per layer. These are ratio-style alignment measurements
        (matching the original sketch implementation).
    """

    if resample_w0 is None:
        resample_w0 = window.resample_w0

    norm_mode = norm_mode.lower().strip()
    if norm_mode not in {"spectral", "rms"}:
        raise ValueError("norm_mode must be one of {'spectral','rms'}")
    
    alpha_list: list[float] = []
    omega_list: list[float] = []
    u_list: list[float] = []
    
    for idx, layer_name in enumerate(window.layer_names):
        cur = window.current.layers[layer_name]
        init = window.init.layers[layer_name]

        base = float(window.fan_in[idx])
        log_base = torch.log(torch.tensor(base, dtype=torch.float64))

        # Current state
        z = cur.input
        w = cur.weight

        # Initial state
        z0 = init.input

        # For nn.Parameter layers (e.g., LayerNorm scale, pos_embed), there are no
        # input activations. These EMBEDDING layers don't participate in alignment
        # computation - just return 0.0 for their metrics.
        if z is None or z0 is None:
            alpha_list.append(0.0)
            omega_list.append(0.0)
            u_list.append(0.0)
            continue

        # Ensure weight tensor is available
        if w is None:
            raise RuntimeError(
                f"Missing weight tensor for layer {layer_name}. "
                "Ensure capture_initial() and capture() were called."
            )
        
        # Get w0: either stored or resampled
        if resample_w0 and window.bl is not None:
            bl = window.bl[idx]
            w0 = _resample_w0(w.shape, bl=bl, std_prefactor=std_prefactor)
        elif init.weight is not None:
            w0 = init.weight
        else:
            raise RuntimeError(
                f"No initial weights available for layer {layer_name}. "
                "Either provide bl for resampling or capture initial weights."
            )
        
        # Convert to double for precision
        z = z.double()
        w = w.double()
        z0 = z0.double()
        w0 = w0.double()
        
        # Compute differences
        dz = z - z0
        dw = w - w0
        
        # Choose norms and formula
        if norm_mode == "spectral":
            vec_norm = _l2_norm
            mat_norm = _spectral_norm
            use_log = False
        else:  # rms
            vec_norm = _rms_norm
            mat_norm = _rms_norm
            use_log = True

        # Norms used by alpha/omega/u
        z0_n = vec_norm(z0, dim=-1)
        dz_n = vec_norm(dz, dim=-1)
        dw_n = mat_norm(dw)
        w0_n = mat_norm(w0)
        
        # Initialize alignment values
        A_alpha = 0.0
        A_omega = 0.0
        A_u = 0.0
        
        # Check if there's any change
        has_change = torch.abs(dw_n + dz_n.sum()).item() > 1e-12
        
        if has_change:
            # Alpha alignment: y = z_0 @ Δw^T
            o_alpha = z0 @ dw.T
            o_alpha_n = vec_norm(o_alpha, dim=-1)
            if use_log:
                A_alpha = torch.mean((torch.log(o_alpha_n) - torch.log(z0_n * dw_n)) / log_base).item()
            else:
                A_alpha = torch.mean(o_alpha_n / (z0_n * dw_n)).item()
            
            # Omega and U are only meaningful for layers after the first
            # (first layer has dz = 0 by definition since input is fixed)
            if idx > 0:
                # Omega alignment: y = Δz @ w_0^T
                o_omega = dz @ w0.T
                o_omega_n = vec_norm(o_omega, dim=-1)
                if use_log:
                    A_omega = torch.mean((torch.log(o_omega_n) - torch.log(dz_n * w0_n)) / log_base).item()
                else:
                    A_omega = torch.mean(o_omega_n / (dz_n * w0_n)).item()
                
                # U alignment: y = Δz @ Δw^T
                o_u = dz @ dw.T
                o_u_n = vec_norm(o_u, dim=-1)
                if use_log:
                    A_u = torch.mean((torch.log(o_u_n) - torch.log(dz_n * dw_n)) / log_base).item()
                else:
                    A_u = torch.mean(o_u_n / (dz_n * dw_n)).item()
        
        alpha_list.append(A_alpha)
        omega_list.append(A_omega)
        u_list.append(A_u)
    
    # Sanitize: replace inf/nan with safe values
    def sanitize(vals: list[float]) -> list[float]:
        result = []
        for v in vals:
            if np.isnan(v):
                result.append(0.0)
            elif np.isneginf(v):
                result.append(0.0)
            elif np.isinf(v):
                result.append(1.0)
            else:
                result.append(v)
        return result
    
    return sanitize(alpha_list), sanitize(omega_list), sanitize(u_list)
