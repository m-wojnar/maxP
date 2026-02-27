"""Parametrization — main entry point for maxp_new."""

from __future__ import annotations

import torch
import torch.nn as nn

from maxp_new.solver import find_c, find_c_dag
from maxp_new.module import ParametrizedModule


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

    When ``sample_input`` is provided, a DAG of PM-to-PM data flow is traced
    and the LP solver assigns a per-PM ``c`` value instead of collapsing by
    layer type.

    Parameter-less modules (bare callables) only get their scale set.

    All other learnable parameters (LayerNorm, biases not inside a
    ParametrizedModule, etc.) are collected into a single ``"_other"`` group
    at ``lr_prefactor``.

    Phase 2 — dynamic alignment:

    Call :meth:`capture_initial` once before training, then call
    :meth:`step` after each ``optimizer.step()`` to measure actual alignment,
    re-solve the LP, and update learning rates.

    Args:
        model: PyTorch model (modified in-place).
        optimizer_type: ``"adam"`` or ``"sgd"`` — needed for LP solver.
        alignment: ``"full"`` or ``"no"`` — alignment assumption for solving c.
        lr_prefactor: Base learning rate multiplier.
        std_prefactor: Multiplier for init std.
        ab_overrides: Optional dict mapping layer_type → ``(a, b)`` to
            override the built-in defaults.
        sample_input: Optional example input for DAG tracing.  When provided,
            the solver assigns per-PM c values based on the actual data flow
            graph instead of collapsing by layer type.
        warmup_steps: Steps before first dynamic re-solve (default 0).
        solve_interval: Re-solve every N steps (default 1).
        sample_size: Max batch samples kept for alignment measurement (default 32).
        norm_mode: ``"rms"`` or ``"spectral"`` for alignment computation.
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
        sample_input: torch.Tensor | None = None,
        # Phase 2 params
        warmup_steps: int = 0,
        solve_interval: int = 1,
        sample_size: int = 32,
        norm_mode: str = "rms",
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

        # Solve for c: DAG path (per-op) or chain path (per-type)
        graph = None
        if sample_input is not None:
            c_by_name, graph = self._solve_c_dag_initial(
                model, sample_input, ab, optimizer_type, alignment
            )
        else:
            c_by_name = self._solve_c_chain_static(pms, ab, optimizer_type, alignment)

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
            has_params = pm.weight is not None

            names.append(name)
            al.append(a)
            bl.append(b)
            fan_ins.append(fan_in)

            # Set output scale
            pm.scale = fan_in ** (-a) if a != 0.0 else 1.0

            if has_params:
                c = c_by_name[name]
                cl.append(c)

                # Re-initialise weights
                inner = pm.inner
                assert inner is not None
                with torch.no_grad():
                    pm.weight.normal_(mean=0.0, std=std_prefactor * (fan_in ** (-b)))
                    for pname, param in inner.named_parameters():
                        if "bias" in pname:
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

        # Phase 2 state
        self._pms = pms
        self._ab = ab
        self._optimizer_type = optimizer_type
        self._alignment = alignment
        self._std_prefactor = std_prefactor
        self._warmup_steps = warmup_steps
        self._solve_interval = solve_interval
        self._sample_size = sample_size
        self._norm_mode = norm_mode
        self._step_count = 0
        self._use_dag = sample_input is not None
        self._graph = graph  # saved OpGraph for DAG re-solve

        # Set initial alignment on each PM from the preset
        alpha_val, omega_val, u_val = _ALIGNMENT_PRESETS[alignment]
        for _name, pm in pms:
            pm.alpha = alpha_val
            pm.omega = omega_val
            pm.u = u_val

    @property
    def param_groups(self) -> list[dict]:
        return self._param_groups

    # ------------------------------------------------------------------
    # Phase 2: dynamic alignment
    # ------------------------------------------------------------------

    def capture_initial(self, sample_input: torch.Tensor) -> None:
        """Capture initial (z_0, w_0) for alignment measurement.

        Must be called before training starts (after ``__init__``).
        Stores ``_z0`` and ``_w0`` on each :class:`ParametrizedModule`.

        Args:
            sample_input: A batch of inputs to run through the model.
                Only the first ``sample_size`` samples are kept.
        """
        captured: dict[str, torch.Tensor] = {}
        hooks: list[torch.utils.hooks.RemovableHook] = []
        sample_size = self._sample_size

        for name, pm in self._pms:
            if pm.inner is None:
                continue

            def _hook(mod, inp, out, _name=name):
                captured[_name] = inp[0].detach().clone()[:sample_size]

            hooks.append(pm.inner.register_forward_hook(_hook))

        with torch.no_grad():
            self.model(sample_input)

        for h in hooks:
            h.remove()

        for name, pm in self._pms:
            if pm.inner is not None and name in captured:
                pm._z0 = captured[name]
                pm._w0 = pm.weight.detach().clone()

    def step(
        self,
        sample_input: torch.Tensor,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Measure alignment, re-solve LP, update optimizer LRs.

        Call this after ``optimizer.step()`` each training step.

        Args:
            sample_input: Batch of inputs for alignment measurement.
            optimizer: The optimizer whose ``param_groups`` to update.
                If ``None``, only updates ``self.param_groups`` (user
                must sync manually).
        """
        self._step_count += 1

        # Check that capture_initial() was called (at least one PM has _z0)
        if not any(pm._z0 is not None for _, pm in self._pms):
            raise RuntimeError(
                "Call capture_initial() before step()."
            )

        if self._step_count <= self._warmup_steps:
            return
        if self._step_count % self._solve_interval != 0:
            return

        # 1. Capture current (z, w) via hooks
        current: dict[str, torch.Tensor] = {}
        hooks: list[torch.utils.hooks.RemovableHook] = []
        sample_size = self._sample_size

        for name, pm in self._pms:
            if pm.inner is None:
                continue

            def _hook(mod, inp, out, _name=name):
                current[_name] = inp[0].detach().clone()[:sample_size]

            hooks.append(pm.inner.register_forward_hook(_hook))

        with torch.no_grad():
            self.model(sample_input)

        for h in hooks:
            h.remove()

        # 2. Compute alignment per PM, write back to PM
        from maxp_new.alignment import compute_alignment

        for name, pm in self._pms:
            if pm._z0 is None or name not in current:
                continue
            z0, w0 = pm._z0, pm._w0
            z = current[name]
            w = pm.weight.detach().clone()
            pm.alpha, pm.omega, pm.u = compute_alignment(
                z0, w0, z, w, fan_in=pm.width_dim, norm_mode=self._norm_mode
            )

        # 3. Re-solve LP
        if self._use_dag:
            c_by_name = self._resolve_dag()
        else:
            c_by_name = self._resolve_chain()

        # 4. Update param groups
        for group in self._param_groups:
            if not group.get("maxp_managed", False):
                continue
            name = group["layer_name"]
            if name in c_by_name:
                c = c_by_name[name]
                fan_in = group["fan_in"]
                group["lr"] = self.lr_prefactor * (fan_in ** (-c))
                group["c"] = float(c)

        # 5. Sync to optimizer if provided
        if optimizer is not None:
            for our_group, opt_group in zip(self._param_groups, optimizer.param_groups):
                opt_group["lr"] = our_group["lr"]

    def _resolve_chain(self) -> dict[str, float]:
        """Re-solve chain LP with per-PM alignment from PM attributes."""
        # Filter to weight-bearing PMs in discovery order
        weighted = [(name, pm) for name, pm in self._pms if pm.weight is not None]

        if len(weighted) >= 2:
            al = [self._ab[pm.layer_type][0] for _, pm in weighted]
            bl = [self._ab[pm.layer_type][1] for _, pm in weighted]
            alpha = [pm.alpha for _, pm in weighted]
            omega = [pm.omega for _, pm in weighted]
            u = [pm.u for _, pm in weighted]

            cl, _ = find_c(al, bl, alpha, omega, u,
                           optimizer_type=self._optimizer_type)
            return {name: c for (name, _), c in zip(weighted, cl)}
        elif len(weighted) == 1:
            return {weighted[0][0]: 0.0}
        else:
            return {}

    def _resolve_dag(self) -> dict[str, float]:
        """Re-solve DAG LP with alignment read from PM attributes."""
        assert self._graph is not None

        # Update alignment on graph nodes from PMs
        for name, pm in self._pms:
            if name in self._graph.nodes:
                node = self._graph.nodes[name]
                node.alpha = pm.alpha
                node.omega = pm.omega
                node.u = pm.u

        result = find_c_dag(self._graph, optimizer_type=self._optimizer_type)

        c_by_name: dict[str, float] = {}
        for name, (c_val, _r_val) in result.items():
            if c_val is not None:
                c_by_name[name] = c_val
        return c_by_name

    # ------------------------------------------------------------------
    # Static solvers (used at init time)
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_c_chain_static(pms, ab, optimizer_type, alignment) -> dict[str, float]:
        """Chain solver: pass full PM chain to LP, get per-PM c values."""
        preset = _ALIGNMENT_PRESETS.get(alignment)
        if preset is None:
            raise ValueError(f"Unknown alignment '{alignment}'. Supported: {list(_ALIGNMENT_PRESETS)}")

        alpha_val, omega_val, u_val = preset

        # Filter to weight-bearing PMs in discovery order
        weighted = [(name, pm) for name, pm in pms if pm.weight is not None]

        if len(weighted) >= 2:
            chain_al = [ab[pm.layer_type][0] for _, pm in weighted]
            chain_bl = [ab[pm.layer_type][1] for _, pm in weighted]
            n = len(chain_al)
            chain_cl, _ = find_c(
                chain_al, chain_bl,
                [alpha_val] * n, [omega_val] * n, [u_val] * n,
                optimizer_type=optimizer_type,
            )
            return {name: c for (name, _), c in zip(weighted, chain_cl)}
        elif len(weighted) == 1:
            return {weighted[0][0]: 0.0}
        else:
            return {}

    @staticmethod
    def _solve_c_dag_initial(model, sample_input, ab, optimizer_type, alignment):
        """DAG solver at init time.  Returns (c_by_name, graph)."""
        from maxp_new.dag import trace_pm_dag

        preset = _ALIGNMENT_PRESETS.get(alignment)
        if preset is None:
            raise ValueError(f"Unknown alignment '{alignment}'. Supported: {list(_ALIGNMENT_PRESETS)}")

        alpha_val, omega_val, u_val = preset

        graph = trace_pm_dag(model, sample_input, ab=ab)

        # Set alignment values on all nodes
        for node in graph.nodes.values():
            node.alpha = alpha_val
            node.omega = omega_val
            node.u = u_val

        result = find_c_dag(graph, optimizer_type=optimizer_type)

        c_by_name: dict[str, float] = {}
        for name, (c_val, _r_val) in result.items():
            if c_val is not None:
                c_by_name[name] = c_val
        return c_by_name, graph
