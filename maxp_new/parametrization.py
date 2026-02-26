"""Parametrization — main entry point for maxp_new."""

from __future__ import annotations

import torch
import torch.nn as nn

from maxp_new.solver import find_c
from maxp_new.utils import ScaledModule, _set_nested_attr


# Alignment assumptions: (alpha, omega, u) per layer
_ALIGNMENT_PRESETS = {
    "full": (1.0, 0.5, 1.0),
    "no": (0.0, 0.0, 0.0),
}


class Parametrization:
    """Apply ABC parametrization to a model.

    Args:
        model: PyTorch model (modified in-place).
        layers: Dict mapping module name → {"a": float, "b": float} or
            {"a": float, "b": float, "c": float}. If c is omitted, it's
            solved for via LP.
        optimizer_type: "adam" or "sgd" — needed for solver.
        alignment: "full" or "no" — alignment assumption for solving c.
        lr_prefactor: Base learning rate multiplier.
        std_prefactor: Multiplier for init std.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        layers: dict[str, dict[str, float]],
        optimizer_type: str = "adam",
        alignment: str = "full",
        lr_prefactor: float = 1e-3,
        std_prefactor: float = 1.0,
    ):
        self.model = model
        self.lr_prefactor = lr_prefactor

        # Resolve layers from model
        names = list(layers.keys())
        modules = []
        for name in names:
            parts = name.split(".")
            mod = model
            for p in parts:
                mod = getattr(mod, p)
            modules.append(mod)

        al = [layers[n]["a"] for n in names]
        bl = [layers[n]["b"] for n in names]

        # Resolve c_l: user-provided or solved
        if all("c" in layers[n] for n in names):
            cl = [layers[n]["c"] for n in names]
        elif any("c" in layers[n] for n in names):
            raise ValueError("Either all layers must specify 'c' or none.")
        else:
            cl = self._solve_c(al, bl, optimizer_type, alignment)

        # Init weights + apply multipliers
        fan_ins: list[int] = []
        for name, mod, a, b in zip(names, modules, al, bl):
            fan_in = mod.weight.shape[1] if mod.weight.ndim >= 2 else mod.weight.shape[0]
            fan_ins.append(fan_in)

            std = std_prefactor * (fan_in ** (-b))
            with torch.no_grad():
                mod.weight.normal_(mean=0.0, std=std)
                if hasattr(mod, "bias") and mod.bias is not None:
                    mod.bias.zero_()

            if a != 0.0:
                _set_nested_attr(model, name, ScaledModule(mod, fan_in ** (-a)))

        # Build param groups
        parametrized_ids: set[int] = set()
        groups: list[dict] = []
        for name, mod, c, fan_in in zip(names, modules, cl, fan_ins):
            params = list(mod.parameters())
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

        other = [p for p in model.parameters() if p.requires_grad and id(p) not in parametrized_ids]
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
