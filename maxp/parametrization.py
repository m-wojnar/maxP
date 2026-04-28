"""Parametrization — main entry point for maxp."""

from __future__ import annotations

import torch
import torch.nn as nn

from maxp.solver import find_c
from maxp.module import ParametrizedModule
from maxp.dag import DagNode, OpGraph, _DEFAULT_AB


# Alignment assumptions: (align_z0_dW, align_dZ_w0, align_dZ_dW) per layer
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
    c_prev: dict[str, float] | None = None,
    solver: "plp.LpSolver | None" = None,
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
    result = find_c(
        graph, optimizer_type=optimizer_type, c_fixed=c_fixed,
        c_prev=c_prev, solver=solver,
    )
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
        alignment_overrides: Optional dict mapping layer name, name suffix,
            or layer_type → ``(align_z0_dW, align_dZ_w0, align_dZ_dW)`` to
            pin alignment for specific layers.  Pinned layers skip dynamic
            measurement in :meth:`step`.  Matching priority: exact name >
            leaf name (e.g. ``"fc2"`` matches ``"blocks.0.ff.fc2"``) > layer_type.
        sample_input: Optional example input for DAG tracing.  When provided,
            the solver assigns per-PM c values based on the actual data flow
            graph instead of collapsing by layer type.
        warmup_steps: Steps before first dynamic re-solve (default 0).
        solve_interval: Re-solve every N steps (default 1).
        sample_size: Max batch samples kept for alignment measurement (default 32).
        c_ema: EMA factor for smoothing ``c`` values toward solver targets.
            Each step: ``c = c_ema * c + (1 - c_ema) * c_target``.
            0.0 means instant updates (default), 0.99 means very slow blending.
        alignment_ema: EMA factor for smoothing raw alignment measurements.
            Each step: ``align = ema * align_old + (1 - ema) * align_new``.
            0.0 means no smoothing (default).  Useful when alignment
            measurements are noisy (e.g. with ``use_training_activations``).
        resample_w0: If True, store a random seed per layer instead of
            cloning ``w_0``.  Regenerates ``w_0`` on-the-fly during alignment
            measurement, saving memory proportional to the total weight size.
        use_training_activations: If True, register persistent forward hooks
            on the model to capture activations during the normal training
            forward pass.  Eliminates the extra forward pass in ``step()``.
            When enabled, ``sample_input`` is not required in ``step()``.
            Call ``register_hooks()`` (or ``capture_initial()``) to install
            the hooks before training.
        solver: PuLP solver instance to use for the LP.  Defaults to
            ``PULP_CBC_CMD(msg=False)``.  Pass any PuLP solver, e.g.
            ``pulp.CPLEX_CMD(msg=False, warmStart=True)`` for CPLEX.
            The solver is reused across dynamic re-solves.
        warm_start: If True, enable LP warm start — seeds LP variables
            with the previous solution and configures the default CBC
            solver with ``warmStart=True``.  Ignored when a custom
            ``solver`` is provided (configure warm start on it directly).
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
        c_ema: float = 0.0,
        alignment_ema: float = 0.0,
        resample_w0: bool = False,
        use_training_activations: bool = False,
        solver: "plp.LpSolver | None" = None,
        warm_start: bool = False,
    ):
        self.model = model
        self.lr_prefactor = lr_prefactor
        self._std_prefactor = std_prefactor

        ab = dict(_DEFAULT_AB)
        if ab_overrides:
            ab.update(ab_overrides)
        self._ab = ab

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
                override = alignment_overrides.get(name)
                if override is None:
                    leaf = name.split(".")[-1]
                    override = alignment_overrides.get(leaf)
                if override is None:
                    override = alignment_overrides.get(pm.layer_type)
                if override is not None:
                    node = graph.nodes[name]
                    node.align_z0_dW, node.align_dZ_w0, node.align_dZ_dW = override

        # Collect user-provided c values (per-PM overrides take priority)
        c_fixed: dict[str, float] = {}
        for name, pm in pms:
            if pm.weight is None:
                continue
            if pm.c is not None:
                c_fixed[name] = pm.c
            elif c_overrides and pm.layer_type in c_overrides:
                c_fixed[name] = c_overrides[pm.layer_type]

        c_by_name = _solve_graph(graph, optimizer_type, c_fixed=c_fixed or None,
                                  solver=solver)

        # Phase 2 state
        self._pms = pms
        self._c_fixed = c_fixed or None
        self._optimizer_type = optimizer_type
        self._warmup_steps = warmup_steps
        self._solve_interval = solve_interval
        self._sample_size = sample_size
        self._c_ema = c_ema
        self._alignment_ema = alignment_ema
        self._resample_w0 = resample_w0
        self._warm_start = warm_start
        self._use_training_activations = use_training_activations
        self._persistent_hooks: list[torch.utils.hooks.RemovableHook] = []
        self._latest_activations: dict[str, torch.Tensor] = {}
        self._step_count = 0
        self._graph = graph
        self._c_prev: dict[str, float] = dict(c_by_name)
        self._solver = solver  # user-provided solver (reused for warm start)
        self.refresh()  # init weights, build param groups

        # Set initial alignment on each PM from the preset, with overrides
        a0_dW_val, dZ_w0_val, dZ_dW_val = _ALIGNMENT_PRESETS[alignment]
        self._alignment_pinned: set[str] = set()
        for name, pm in pms:
            # Check overrides: exact name > name suffix > layer_type
            override = None
            if alignment_overrides:
                if name in alignment_overrides:
                    override = alignment_overrides[name]
                else:
                    leaf = name.split(".")[-1]
                    if leaf in alignment_overrides:
                        override = alignment_overrides[leaf]
                    elif pm.layer_type in alignment_overrides:
                        override = alignment_overrides[pm.layer_type]
            if override is not None:
                pm.align_z0_dW, pm.align_dZ_w0, pm.align_dZ_dW = override
                self._alignment_pinned.add(name)
            else:
                pm.align_z0_dW = a0_dW_val
                pm.align_dZ_w0 = dZ_w0_val
                pm.align_dZ_dW = dZ_dW_val

    @property
    def param_groups(self) -> list[dict]:
        return self._param_groups

    def refresh(self, optimizer: torch.optim.Optimizer | None = None) -> None:
        """Re-run weight init and rebuild ``param_groups`` against live params.

        Use this when ``Parametrization`` was built on a meta-device model
        (so ``__init__`` couldn't populate real weights or capture live
        param tensor refs).
        """
        ab = self._ab
        std_prefactor = self._std_prefactor

        # Apply: init weights, set scale, build param groups
        parametrized_ids: set[int] = set()
        groups: list[dict] = []

        for name, pm in self._pms:
            lt = pm.layer_type
            a_default, b_default = ab[lt]
            a = pm.a if pm.a is not None else a_default
            b = pm.b if pm.b is not None else b_default
            fan_in = pm.width_dim
            has_params = pm.weight is not None

            # Set output scale
            pm.scale = fan_in ** (-a) if a != 0.0 else 1.0

            if has_params:
                c = self._c_prev[name]

                # Re-initialise weights
                inner = pm.inner
                assert inner is not None
                init_std = std_prefactor * (fan_in ** (-b))
                with torch.no_grad():
                    if self._resample_w0:
                        seed = torch.randint(0, 2**31, (1,)).item()
                        pm._w0_seed = seed
                        pm._w0_std = init_std
                        gen = torch.Generator(device=pm.weight.device)
                        gen.manual_seed(seed)
                        pm.weight.normal_(mean=0.0, std=init_std, generator=gen)
                    else:
                        pm.weight.normal_(mean=0.0, std=init_std)
                    for pname, param in inner.named_parameters():
                        if "bias" in pname:
                            param.zero_()

                # Build param group
                params = list(inner.parameters())
                for p in params:
                    parametrized_ids.add(id(p))
                groups.append({
                    "params": params,
                    "lr": self.lr_prefactor * (fan_in ** (-c)),
                    "layer_name": name,
                    "fan_in": fan_in,
                    "c": float(c),
                    "maxp_managed": True,
                })

        # Collect all other parameters (LayerNorm, etc.)
        other = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in parametrized_ids
        ]
        if other:
            groups.append({
                "params": other,
                "lr": self.lr_prefactor,
                "layer_name": "_other",
                "maxp_managed": False,
            })

        self._param_groups = groups

        # c_target tracks the latest solver output; c blends toward it each step
        self._c_target: dict[str, float] = {
            group["layer_name"]: group["c"]
            for group in groups if group.get("maxp_managed", False)
        }

        if optimizer is not None:
            non_lr_defaults = {
                k: v for k, v in optimizer.param_groups[0].items()
                if k not in ("lr", "params")
            }
            optimizer.param_groups[:] = [{**g, **non_lr_defaults} for g in groups]
            # Reset state that referenced the old (now-dead) parameter ids
            optimizer.state.clear()

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
                captured[_name] = inp[0][:sample_size].detach().clone()

            hooks.append(pm.inner.register_forward_hook(_hook))

        with torch.no_grad():
            self.model(sample_input)

        for h in hooks:
            h.remove()

        return captured

    def register_hooks(self) -> None:
        """Install persistent forward hooks to capture activations during training.

        Called automatically by :meth:`capture_initial` when
        ``use_training_activations=True``.  The hooks populate
        ``_latest_activations`` on every forward pass so that ``step()``
        can skip its own forward pass.
        """
        self.remove_hooks()
        sample_size = self._sample_size
        activations = self._latest_activations

        # Pre-forward hook on the model clears old activations so they
        # don't coexist in memory with the new autograd graph.
        def _clear_hook(mod, inp):
            activations.clear()

        self._persistent_hooks.append(
            self.model.register_forward_pre_hook(_clear_hook)
        )

        for name, pm in self._pms:
            if pm.inner is None or isinstance(pm.inner, nn.Embedding):
                continue

            def _hook(mod, inp, out, _name=name):
                activations[_name] = inp[0][:sample_size].detach().clone()

            self._persistent_hooks.append(pm.inner.register_forward_hook(_hook))

    def remove_hooks(self) -> None:
        """Remove any persistent forward hooks."""
        for h in self._persistent_hooks:
            h.remove()
        self._persistent_hooks.clear()
        self._latest_activations.clear()

    def capture_initial(self, sample_input: torch.Tensor) -> None:
        """Capture initial (z_0, w_0) for alignment measurement.

        Must be called before training starts (after ``__init__``).
        Stores ``_z0`` and ``_w0`` on each :class:`ParametrizedModule`.

        When ``use_training_activations=True``, also installs persistent
        forward hooks via :meth:`register_hooks`.

        Args:
            sample_input: A batch of inputs to run through the model.
                Only the first ``sample_size`` samples are kept.
        """
        captured = self._capture_activations(sample_input)

        for name, pm in self._pms:
            if pm.inner is not None and pm.weight is not None and name in captured:
                pm._z0 = captured[name]
                if not self._resample_w0:
                    _w = pm.weight.detach()
                    if hasattr(_w, "to_local"):
                        _w = _w.to_local()
                    pm._w0 = _w.clone()

        if self._use_training_activations and not self._persistent_hooks:
            self.register_hooks()

    def _regenerate_w0(self, pm: ParametrizedModule) -> torch.Tensor:
        """Regenerate initial weights from stored seed (resample_w0 mode)."""
        w0 = torch.empty_like(pm.weight)
        gen = torch.Generator(device=pm.weight.device)
        gen.manual_seed(pm._w0_seed)
        w0.normal_(mean=0.0, std=pm._w0_std, generator=gen)
        return w0

    def _sync_lrs(self, optimizer: torch.optim.Optimizer | None) -> None:
        """Recompute LRs from current lr_prefactor + c, sync to optimizer.

        When ``c_ema > 0``, blends each group's ``c`` toward ``c_target``
        every call: ``c = c_ema * c + (1 - c_ema) * c_target``.
        """
        ema = self._c_ema
        for group in self._param_groups:
            if not group.get("maxp_managed", False):
                group["lr"] = self.lr_prefactor
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
        sample_input: torch.Tensor | None = None,
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
                Not required when ``use_training_activations=True``.
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

        # 1. Capture current (z, w) — either from persistent hooks or a dedicated pass
        if self._use_training_activations:
            if not self._latest_activations:
                raise RuntimeError(
                    "No activations captured yet. Ensure register_hooks() "
                    "was called and the model has run at least one forward pass."
                )
            current = dict(self._latest_activations)
        else:
            if sample_input is None:
                raise ValueError(
                    "sample_input is required when use_training_activations=False."
                )
            current = self._capture_activations(sample_input)

        # 2. Compute alignment per PM, write back to PM
        from maxp.alignment import compute_alignment

        for name, pm in self._pms:
            if name in self._alignment_pinned:
                continue
            if pm._z0 is None or name not in current:
                continue
            z0 = pm._z0
            w0 = self._regenerate_w0(pm) if self._resample_w0 else pm._w0
            z = current[name]
            w = pm.weight.detach()
            if hasattr(w, "to_local"):
                w = w.to_local()
            w = w.clone()
            new_a0_dW, new_dZ_w0, new_dZ_dW = compute_alignment(
                z0, w0, z, w, fan_in=pm.width_dim
            )
            a_ema = self._alignment_ema
            if a_ema > 0 and pm.align_z0_dW is not None:
                pm.align_z0_dW = a_ema * pm.align_z0_dW + (1 - a_ema) * new_a0_dW
                pm.align_dZ_w0 = a_ema * pm.align_dZ_w0 + (1 - a_ema) * new_dZ_w0
                pm.align_dZ_dW = a_ema * pm.align_dZ_dW + (1 - a_ema) * new_dZ_dW
            else:
                pm.align_z0_dW, pm.align_dZ_w0, pm.align_dZ_dW = new_a0_dW, new_dZ_w0, new_dZ_dW

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
        import pulp as plp

        for name, pm in self._pms:
            if name in self._graph.nodes:
                node = self._graph.nodes[name]
                node.align_z0_dW = pm.align_z0_dW
                node.align_dZ_w0 = pm.align_dZ_w0
                node.align_dZ_dW = pm.align_dZ_dW

        if self._solver is None:
            self._solver = plp.PULP_CBC_CMD(msg=False, warmStart=self._warm_start)

        c_by_name = _solve_graph(
            self._graph, self._optimizer_type,
            c_fixed=self._c_fixed,
            c_prev=self._c_prev if self._warm_start else None,
            solver=self._solver,
        )
        self._c_prev = dict(c_by_name)
        return c_by_name

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

        a0_dW_val, dZ_w0_val, dZ_dW_val = preset

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
                align_z0_dW=a0_dW_val, align_dZ_w0=dZ_w0_val, align_dZ_dW=dZ_dW_val,
            )

        return OpGraph(nodes)

    @staticmethod
    def _trace_graph(model, sample_input, ab, alignment) -> OpGraph:
        """Trace data flow graph from model execution."""
        from maxp.dag import trace_pm_dag

        preset = _ALIGNMENT_PRESETS.get(alignment)
        if preset is None:
            raise ValueError(f"Unknown alignment '{alignment}'. Supported: {list(_ALIGNMENT_PRESETS)}")

        a0_dW_val, dZ_w0_val, dZ_dW_val = preset

        graph = trace_pm_dag(model, sample_input, ab=ab)

        for node in graph.nodes.values():
            node.align_z0_dW = a0_dW_val
            node.align_dZ_w0 = dZ_w0_val
            node.align_dZ_dW = dZ_dW_val

        return graph
