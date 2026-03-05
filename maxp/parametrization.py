"""Parametrization — main entry point for maxp."""

from __future__ import annotations

import torch
import torch.nn as nn

from maxp.solver import find_c
from maxp.module import ParametrizedModule
from maxp.dag import DagNode, OpGraph, _DEFAULT_AB


# Alignment assumptions: (alpha, omega, u) per layer
_ALIGNMENT_PRESETS = {
    "full": (1.0, 0.5, 1.0),
    "no": (0.5, 0.5, 0.5),
}


def _extract_c(result: dict) -> dict[str, float]:
    """Extract c values from solver result, dropping None entries."""
    return {name: c for name, (c, _) in result.items() if c is not None}


def _solve_graph(
    graph: OpGraph,
    optimizer_type: str,
    c_fixed: dict[str, float] | None = None,
) -> dict[str, float]:
    """Solve LP on graph, with trivial fallback for <2 weighted nodes."""
    weighted = [n for n in graph.nodes.values() if n.has_weight]
    if len(weighted) < 2:
        return {
            n.name: c_fixed.get(n.name, 0.0) if c_fixed else 0.0
            for n in weighted
        }
    # All fixed → skip solver entirely
    if c_fixed and all(n.name in c_fixed for n in weighted):
        return {n.name: c_fixed[n.name] for n in weighted}
    result = find_c(graph, optimizer_type=optimizer_type, c_fixed=c_fixed)
    return _extract_c(result)


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
        c_overrides: Optional dict mapping layer_type → ``c`` to override
            the LP-solved value.  Per-PM ``c`` takes priority over this.
        alignment_overrides: Optional dict mapping layer name or layer_type →
            ``(alpha, omega, u)`` to pin alignment for specific layers.
            Pinned layers skip dynamic measurement in :meth:`step`.
            Per-name entries take priority over per-type entries.
        sample_input: Optional example input for DAG tracing.  When provided,
            the solver assigns per-PM c values based on the actual data flow
            graph instead of collapsing by layer type.
        warmup_steps: Steps before first dynamic re-solve (default 0).
        solve_interval: Re-solve every N steps (default 1).
        sample_size: Max batch samples kept for alignment measurement (default 32).
        norm_mode: ``"rms"`` or ``"spectral"`` for alignment computation.
        c_ema: EMA factor for smoothing ``c`` values toward solver targets.
            Each step: ``c = c_ema * c + (1 - c_ema) * c_target``.
            0.0 means instant updates (default), 0.99 means very slow blending.
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
        c_overrides: dict[str, float] | None = None,
        alignment_overrides: dict[str, tuple[float, float, float]] | None = None,
        sample_input: torch.Tensor | None = None,
        # Phase 2 params
        warmup_steps: int = 0,
        solve_interval: int = 1,
        sample_size: int = 32,
        norm_mode: str = "rms",
        c_ema: float = 0.0,
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

        # Build graph and solve for c
        if sample_input is not None:
            graph = self._trace_graph(model, sample_input, ab, alignment)
        else:
            graph = self._build_chain_graph(pms, ab, alignment)

        # Apply alignment overrides to graph nodes
        if alignment_overrides:
            for name, pm in pms:
                if name not in graph.nodes:
                    continue
                override = alignment_overrides.get(name) or alignment_overrides.get(pm.layer_type)
                if override is not None:
                    node = graph.nodes[name]
                    node.alpha, node.omega, node.u = override

        # Collect user-provided c values (per-PM overrides take priority)
        c_fixed: dict[str, float] = {}
        for name, pm in pms:
            if pm.weight is None:
                continue
            if pm.c is not None:
                c_fixed[name] = pm.c
            elif c_overrides and pm.layer_type in c_overrides:
                c_fixed[name] = c_overrides[pm.layer_type]

        c_by_name = _solve_graph(graph, optimizer_type, c_fixed=c_fixed or None)

        # Apply: init weights, set scale, build param groups
        parametrized_ids: set[int] = set()
        groups: list[dict] = []

        for name, pm in pms:
            lt = pm.layer_type
            a_default, b_default = ab[lt]
            a = pm.a if pm.a is not None else a_default
            b = pm.b if pm.b is not None else b_default
            fan_in = pm.width_dim
            has_params = pm.weight is not None

            # Set output scale
            pm.scale = fan_in ** (-a) if a != 0.0 else 1.0

            if has_params:
                c = c_by_name[name]

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

        # Phase 2 state
        self._pms = pms
        self._c_fixed = c_fixed or None
        self._optimizer_type = optimizer_type
        self._warmup_steps = warmup_steps
        self._solve_interval = solve_interval
        self._sample_size = sample_size
        self._norm_mode = norm_mode
        self._c_ema = c_ema
        self._step_count = 0
        self._graph = graph

        # c_target tracks the latest solver output; c blends toward it each step
        self._c_target: dict[str, float] = {
            group["layer_name"]: group["c"]
            for group in groups if group.get("maxp_managed", False)
        }

        # Set initial alignment on each PM from the preset, with overrides
        alpha_val, omega_val, u_val = _ALIGNMENT_PRESETS[alignment]
        self._alignment_pinned: set[str] = set()
        for name, pm in pms:
            # Check overrides: per-name first, then per-layer_type
            override = None
            if alignment_overrides:
                if name in alignment_overrides:
                    override = alignment_overrides[name]
                elif pm.layer_type in alignment_overrides:
                    override = alignment_overrides[pm.layer_type]
            if override is not None:
                pm.alpha, pm.omega, pm.u = override
                self._alignment_pinned.add(name)
            else:
                pm.alpha = alpha_val
                pm.omega = omega_val
                pm.u = u_val

    @property
    def param_groups(self) -> list[dict]:
        return self._param_groups

    # ------------------------------------------------------------------
    # Phase 2: dynamic alignment
    # ------------------------------------------------------------------

    def _capture_activations(self, sample_input: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run a forward pass and capture pre-activation inputs for each PM."""
        captured: dict[str, torch.Tensor] = {}
        hooks: list[torch.utils.hooks.RemovableHook] = []
        sample_size = self._sample_size

        for name, pm in self._pms:
            if pm.inner is None or isinstance(pm.inner, nn.Embedding):
                continue

            def _hook(mod, inp, out, _name=name):
                captured[_name] = inp[0].detach().clone()[:sample_size]

            hooks.append(pm.inner.register_forward_hook(_hook))

        with torch.no_grad():
            self.model(sample_input)

        for h in hooks:
            h.remove()

        return captured

    def capture_initial(self, sample_input: torch.Tensor) -> None:
        """Capture initial (z_0, w_0) for alignment measurement.

        Must be called before training starts (after ``__init__``).
        Stores ``_z0`` and ``_w0`` on each :class:`ParametrizedModule`.

        Args:
            sample_input: A batch of inputs to run through the model.
                Only the first ``sample_size`` samples are kept.
        """
        captured = self._capture_activations(sample_input)

        for name, pm in self._pms:
            if pm.inner is not None and name in captured:
                pm._z0 = captured[name]
                pm._w0 = pm.weight.detach().clone()

    def _sync_lrs(self, optimizer: torch.optim.Optimizer | None) -> None:
        """Recompute LRs from current lr_prefactor + c, sync to optimizer.

        When ``c_ema > 0``, blends each group's ``c`` toward ``c_target``
        every call: ``c = c_ema * c + (1 - c_ema) * c_target``.
        """
        ema = self._c_ema
        for group in self._param_groups:
            if not group.get("maxp_managed", False):
                continue
            if ema > 0:
                name = group["layer_name"]
                target = self._c_target.get(name, group["c"])
                group["c"] = ema * group["c"] + (1 - ema) * target
            fan_in = group["fan_in"]
            c = group["c"]
            group["lr"] = self.lr_prefactor * (fan_in ** (-c))

        if optimizer is not None:
            for our_group, opt_group in zip(self._param_groups, optimizer.param_groups):
                opt_group["lr"] = our_group["lr"]

    def step(
        self,
        sample_input: torch.Tensor,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Measure alignment, re-solve LP, update optimizer LRs.

        Call this after ``optimizer.step()`` each training step.

        LRs are always recomputed from the current ``lr_prefactor`` and
        per-layer ``c`` values, so external changes to ``lr_prefactor``
        (e.g. from a schedule) take effect every step.  The LP is only
        re-solved after warmup and on solve-interval boundaries.

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
            self._sync_lrs(optimizer)
            return
        if self._step_count % self._solve_interval != 0:
            self._sync_lrs(optimizer)
            return

        # 1. Capture current (z, w) via hooks
        current = self._capture_activations(sample_input)

        # 2. Compute alignment per PM, write back to PM
        from maxp.alignment import compute_alignment

        for name, pm in self._pms:
            if name in self._alignment_pinned:
                continue
            if pm._z0 is None or name not in current:
                continue
            z0, w0 = pm._z0, pm._w0
            z = current[name]
            w = pm.weight.detach().clone()
            pm.alpha, pm.omega, pm.u = compute_alignment(
                z0, w0, z, w, fan_in=pm.width_dim, norm_mode=self._norm_mode
            )

        # 3. Re-solve LP (skip c update if infeasible with current alignment)
        try:
            c_by_name = self._resolve()
        except ValueError:
            self._sync_lrs(optimizer)
            return

        # 4. Update c targets (blending happens in _sync_lrs)
        for group in self._param_groups:
            if not group.get("maxp_managed", False):
                continue
            name = group["layer_name"]
            if name in c_by_name:
                self._c_target[name] = float(c_by_name[name])
                if self._c_ema == 0:
                    group["c"] = float(c_by_name[name])

        # 5. Recompute LRs and sync
        self._sync_lrs(optimizer)

    def _resolve(self) -> dict[str, float]:
        """Re-solve LP with current per-PM alignment values."""
        for name, pm in self._pms:
            if name in self._graph.nodes:
                node = self._graph.nodes[name]
                node.alpha = pm.alpha
                node.omega = pm.omega
                node.u = pm.u
        return _solve_graph(self._graph, self._optimizer_type, c_fixed=self._c_fixed)

    # ------------------------------------------------------------------
    # Graph builders (used at init time)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chain_graph(pms, ab, alignment) -> OpGraph:
        """Build a synthetic linear-chain OpGraph from ordered PMs.

        Wires each weight-bearing PM's successor to the next in discovery
        order, so the DAG solver sees a simple chain.
        """
        preset = _ALIGNMENT_PRESETS.get(alignment)
        if preset is None:
            raise ValueError(f"Unknown alignment '{alignment}'. Supported: {list(_ALIGNMENT_PRESETS)}")

        alpha_val, omega_val, u_val = preset

        weighted = [(name, pm) for name, pm in pms if pm.weight is not None]
        nodes: dict[str, DagNode] = {}

        for idx, (name, pm) in enumerate(weighted):
            a = pm.a if pm.a is not None else ab[pm.layer_type][0]
            b = pm.b if pm.b is not None else ab[pm.layer_type][1]
            preds = [weighted[idx - 1][0]] if idx > 0 else []
            succs = [weighted[idx + 1][0]] if idx < len(weighted) - 1 else []
            nodes[name] = DagNode(
                name=name, a=a, b=b,
                layer_type=pm.layer_type,
                has_weight=True,
                width_dim=pm.width_dim,
                predecessors=preds,
                successors=succs,
                alpha=alpha_val, omega=omega_val, u=u_val,
            )

        return OpGraph(nodes)

    @staticmethod
    def _trace_graph(model, sample_input, ab, alignment) -> OpGraph:
        """Trace data flow graph from model execution."""
        from maxp.dag import trace_pm_dag

        preset = _ALIGNMENT_PRESETS.get(alignment)
        if preset is None:
            raise ValueError(f"Unknown alignment '{alignment}'. Supported: {list(_ALIGNMENT_PRESETS)}")

        alpha_val, omega_val, u_val = preset

        graph = trace_pm_dag(model, sample_input, ab=ab)

        for node in graph.nodes.values():
            node.alpha = alpha_val
            node.omega = omega_val
            node.u = u_val

        return graph
