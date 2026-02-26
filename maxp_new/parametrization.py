"""Parametrization — main entry point for maxp_new."""

from __future__ import annotations

import torch
import torch.nn as nn

from maxp_new.solver import find_c
from maxp_new.utils import ParametrizedModule


# Alignment assumptions: (alpha, omega, u) per layer
_ALIGNMENT_PRESETS = {
    "full": (1.0, 0.5, 1.0),
    "no": (0.0, 0.0, 0.0),
}

# Default (a, b) per layer type for muP
# Constraints: embedding a+b=0, hidden a+b=0.5, readout a+b>=0.5
_DEFAULT_AB = {
    "embedding": (-0.5, 0.5),  # a+b=0, scale=sqrt(n), std=1/sqrt(n)
    "hidden":    (0.0, 0.5),   # a+b=0.5, no multiplier, std=1/sqrt(n)
    "readout":   (0.5, 0.5),   # a+b=1.0, scale=1/sqrt(n), std=1/sqrt(n)
}

# Canonical ordering for the LP solver chain
_CANONICAL_ORDER = ["embedding", "hidden", "readout"]


class Parametrization:
    """Apply ABC parametrization to a model containing ParametrizedModule markers.

    Walks the model via ``named_modules()`` to find all
    :class:`ParametrizedModule` instances.  For each one it:

    1. Reads ``width_dim`` and ``layer_type``.
    2. Looks up ``(a, b)`` from *ab_overrides* or the built-in defaults.
    3. Re-initialises weights: ``std = std_prefactor * width_dim ** (-b)``.
    4. Sets the output scale: ``pm.scale = width_dim ** (-a)``.
    5. Solves for ``c`` per layer type via LP, then computes
       ``lr = lr_prefactor * width_dim ** (-c)`` for the param group.

    Parameter-less modules (bare callables) only get their scale set.

    All other learnable parameters (LayerNorm, biases not inside a
    ParametrizedModule, etc.) are collected into a single ``"_other"`` group
    at ``lr_prefactor``.

    Args:
        model: PyTorch model (modified in-place).
        optimizer_type: ``"adam"`` or ``"sgd"`` — needed for LP solver.
        alignment: ``"full"`` or ``"no"`` — alignment assumption for solving c.
        lr_prefactor: Base learning rate multiplier.
        std_prefactor: Multiplier for init std.
        ab_overrides: Optional dict mapping layer_type → ``(a, b)`` to
            override the built-in defaults.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        optimizer_type: str = "adam",
        alignment: str = "full",
        lr_prefactor: float = 1e-3,
        std_prefactor: float = 1.0,
        ab_overrides: dict[str, tuple[float, float]] | None = None,
    ):
        self.model = model
        self.lr_prefactor = lr_prefactor

        ab = dict(_DEFAULT_AB)
        if ab_overrides:
            ab.update(ab_overrides)

        # Discover all ParametrizedModule instances
        pms: list[tuple[str, ParametrizedModule]] = [
            (name, mod) for name, mod in model.named_modules()
            if isinstance(mod, ParametrizedModule)
        ]

        if not pms:
            raise ValueError(
                "No ParametrizedModule instances found in the model. "
                "Wrap relevant layers with ParametrizedModule before calling Parametrization."
            )

        # Validate layer types
        for name, pm in pms:
            if pm.layer_type not in ab:
                raise ValueError(
                    f"Unknown layer_type '{pm.layer_type}' for '{name}'. "
                    f"Supported: {list(ab.keys())} (or pass ab_overrides)."
                )

        # Determine which types have learnable parameters
        types_with_params: set[str] = set()
        for _, pm in pms:
            if pm.inner is not None and any(True for _ in pm.inner.parameters()):
                types_with_params.add(pm.layer_type)

        # Build virtual chain in canonical order and solve for c per type.
        # The LP solver expects: embedding(a+b=0) → hidden(a+b=0.5) → readout(a+b≥0.5).
        chain_types = [lt for lt in _CANONICAL_ORDER if lt in types_with_params]

        if len(chain_types) >= 2:
            chain_al = [ab[lt][0] for lt in chain_types]
            chain_bl = [ab[lt][1] for lt in chain_types]
            chain_cl = self._solve_c(chain_al, chain_bl, optimizer_type, alignment)
            c_by_type: dict[str, float] = dict(zip(chain_types, chain_cl))
        elif len(chain_types) == 1:
            # Single type — no solver needed, use c=0 (lr = lr_prefactor)
            c_by_type = {chain_types[0]: 0.0}
        else:
            c_by_type = {}

        # Apply: init weights, set scale, build param groups
        names: list[str] = []
        al: list[float] = []
        bl: list[float] = []
        cl: list[float | None] = []
        fan_ins: list[int] = []

        parametrized_ids: set[int] = set()
        groups: list[dict] = []

        for name, pm in pms:
            lt = pm.layer_type
            a, b = ab[lt]
            fan_in = pm.width_dim
            has_params = pm.inner is not None and any(True for _ in pm.inner.parameters())

            names.append(name)
            al.append(a)
            bl.append(b)
            fan_ins.append(fan_in)

            # Set output scale
            pm.scale = fan_in ** (-a) if a != 0.0 else 1.0

            if has_params:
                c = c_by_type[lt]
                cl.append(c)

                # Re-initialise weights
                inner = pm.inner
                assert inner is not None
                for pname, param in inner.named_parameters():
                    if param.ndim >= 2:
                        std = std_prefactor * (fan_in ** (-b))
                        with torch.no_grad():
                            param.normal_(mean=0.0, std=std)
                    elif "bias" in pname:
                        with torch.no_grad():
                            param.zero_()

                # Build param group
                params = list(inner.parameters())
                for p in params:
                    parametrized_ids.add(id(p))
                groups.append({
                    "params": params,
                    "lr": lr_prefactor * (fan_in ** (-c)),
                    "layer_name": name,
                    "fan_in": fan_in,
                    "c": float(c),
                    "maxp_managed": True,
                })
            else:
                cl.append(None)

        # Collect all other parameters (LayerNorm, etc.)
        other = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in parametrized_ids
        ]
        if other:
            groups.append({
                "params": other,
                "lr": lr_prefactor,
                "layer_name": "_other",
                "maxp_managed": False,
            })

        self._param_groups = groups
        self._al = al
        self._bl = bl
        self._cl = cl
        self._fan_ins = fan_ins

    @property
    def param_groups(self) -> list[dict]:
        return self._param_groups

    def step(self):
        """Phase 2 stub — dynamic alignment measurement + re-solve."""

    @staticmethod
    def _solve_c(al, bl, optimizer_type, alignment):
        preset = _ALIGNMENT_PRESETS.get(alignment)
        if preset is None:
            raise ValueError(f"Unknown alignment '{alignment}'. Supported: {list(_ALIGNMENT_PRESETS)}")
        alpha_val, omega_val, u_val = preset
        n = len(al)
        alpha = [alpha_val] * n
        omega = [omega_val] * n
        u = [u_val] * n
        cl, _rl = find_c(al, bl, alpha, omega, u, optimizer_type=optimizer_type)
        return cl
