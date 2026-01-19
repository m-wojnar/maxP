"""
MaxP Learning Rate Scheduler.

Dynamically adjusts per-layer learning rates using alignment measurements
between initial and current weights/activations, solved via linear programming.
"""
import math
from typing import Literal
import numpy as np
import pulp as plp
import torch
import torch.nn as nn
import torch.optim

from maxp.tracer import Tracer
from maxp.alignment import compute_alignment
from maxp.solver import find_c
from maxp.utils import AlignmentType, ParametrizationType, get_abc_parametrization, get_linear_layers


class MaxPScheduler:
    """
    Maximal muP parametrization learning rate scheduler.
    
    Dynamically adjusts per-layer learning rates for nn.Linear weights based on
    alignment measurements between initial and current weights/activations.
    Uses LP solver to find optimal learning rate exponents that maximize
    training speed while maintaining stability.
    
    Other parameters (LayerNorm, Attention, etc.) are not adjusted;
    their learning rates remain constant throughout training.
    
    The scheduler must be used with an optimizer whose param groups were created
    by `create_param_groups()`. Groups with "maxp_managed": True are adjusted;
    groups with "maxp_managed": False keep their initial LR.
    
    There are two ways to specify the ABC parametrization:
    
    1. **Named parametrization**: Pass `parametrization="mup"` (or "sp", "ntk", "mfp")
       and the scheduler will automatically generate appropriate al/bl values.
       
    2. **Custom values**: Pass explicit `al` and `bl` lists for fine-grained control.
    
    Args:
        optimizer: PyTorch optimizer with param groups from create_param_groups().
        model: PyTorch model being optimized.
        al: List of a_l exponents (layer multipliers), one per Linear layer.
            Required if `parametrization` is not provided.
        bl: List of b_l exponents (initialization variance), one per Linear layer.
            Required if `parametrization` is not provided.
        parametrization: Named parametrization type. One of:
            - "mup": Maximal Update Parametrization (μP)
            - "sp": Standard Parametrization (PyTorch default)
            - "ntk": Neural Tangent Kernel parametrization  
            - "mfp": Mean Field Parametrization
            If provided, al and bl are automatically generated based on the
            optimizer type and alignment assumption.
        alignment_assumption: Alignment assumption for named parametrizations.
            One of "full" or "no", used when `parametrization` is provided. Default: "full".
        lr_prefactor: Base learning rate multiplier.
        solver_warmup_steps: Number of steps before engaging the LP solver.
            During solver warmup, base learning rate remains at its initial value
            (but WSD warmup is still applied). Default: 100.
        sample_size: Number of samples to use for activation capture.
            Default: 32.
        solve_interval: How often to solve the LP (every N steps).
            Set to 1 for every step. Default: 1.
        solver: PuLP solver instance for the LP.
            Examples: pulp.PULP_CBC_CMD(), pulp.CPLEX_CMD().
            If None, uses default CBC solver.
        resample_w0: If True, resample initial weights during alignment
            computation instead of storing them. Saves memory but introduces
            variance. Default: False.
        feature_learning: If True, enforce feature learning constraint
            (r_{L-1} = 0) in LP. Default: False.
        alignment_norm: Norm to use for alignment computation. Either "spectral" or "rms".
            Default: "rms".
        wsd_warmup_steps: Number of steps for linear LR warmup from wsd_min_factor to 1.0.
            Independent of solver_warmup_steps. Default: 0 (no LR warmup).
        wsd_stable_steps: Number of steps for stable LR phase (multiplier = 1.0).
            Required when wsd_decay_type != "none". Default: None.
        wsd_decay_steps: Number of steps for LR decay phase.
            Required when wsd_decay_type != "none". During decay, the LP solver
            stops and per-layer LRs are frozen at their last computed values,
            with only the decay multiplier applied. Default: None.
        wsd_decay_type: Type of decay schedule. One of "none", "cosine", or "linear".
            Default: "none" (WSD disabled).
        wsd_min_factor: Minimum LR as fraction of lr_prefactor, used for warmup start
            and decay end. Default: 0.0.
    
    Example using named parametrization:
        >>> model = MLP(width=128, depth=4)
        >>> cl = get_abc_parametrization(4, "mup", "adam", "full").cl
        >>> param_groups = create_param_groups(model, lr_prefactor=0.1, cl=cl)
        >>> optimizer = torch.optim.AdamW(param_groups)
        >>> scheduler = MaxPScheduler(
        ...     optimizer, model, parametrization="mup", lr_prefactor=0.1
        ... )
    
    Example using custom values:
        >>> model = MLP(width=128, depth=4)
        >>> al = [0.0, 0.5, 0.5, 0.5]
        >>> bl = [0.0, 0.5, 0.5, 0.5]
        >>> cl = [0.0, 0.5, 0.5, 0.5]
        >>> param_groups = create_param_groups(model, lr_prefactor=0.1, cl=cl)
        >>> optimizer = torch.optim.AdamW(param_groups)
        >>> scheduler = MaxPScheduler(
        ...     optimizer, model, al=al, bl=bl, lr_prefactor=0.1
        ... )
        >>> 
        >>> # Before training
        >>> X_init = next(iter(train_loader))[0]
        >>> scheduler.capture_initial(X_init)
        >>> 
        >>> # Training loop
        >>> for X, y in train_loader:
        ...     optimizer.zero_grad()
        ...     loss = criterion(model(X), y)
        ...     loss.backward()
        ...     optimizer.step()
        ...     scheduler.step(X)  # Pass input for alignment computation
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        lr_prefactor: float,
        al: list[float] | None = None,
        bl: list[float] | None = None,
        parametrization: ParametrizationType | None = None,
        alignment_assumption: AlignmentType = "full",
        solver_warmup_steps: int = 100,
        sample_size: int = 32,
        solve_interval: int = 1,
        solver: plp.LpSolver | None = None,
        resample_w0: bool = False,
        feature_learning: bool = False,
        alignment_norm: str = "rms",
        wsd_warmup_steps: int = 0,
        wsd_stable_steps: int | None = None,
        wsd_decay_steps: int | None = None,
        wsd_decay_type: Literal["none", "cosine", "linear"] = "none",
        wsd_min_factor: float = 0.0,
    ):
        self.optimizer = optimizer
        self.model = model
        self.lr_prefactor = lr_prefactor
        self.optimizer_type = self._infer_optimizer_type(optimizer)
        
        # Resolve parametrization
        n_layers = len(get_linear_layers(model))
        self.al, self.bl, self._param_name = self._resolve_parametrization(
            al=al,
            bl=bl,
            parametrization=parametrization,
            alignment_assumption=alignment_assumption,
            optimizer_type=self.optimizer_type,
            n_layers=n_layers,
        )
        self.solver_warmup_steps = solver_warmup_steps
        self.sample_size = sample_size
        self.solve_interval = solve_interval
        self.solver = solver
        self.resample_w0 = resample_w0
        self.feature_learning = feature_learning
        self.alignment_norm = alignment_norm
        
        # WSD parameters
        self.wsd_warmup_steps = wsd_warmup_steps
        self.wsd_stable_steps = wsd_stable_steps
        self.wsd_decay_steps = wsd_decay_steps
        self.wsd_decay_type = wsd_decay_type
        self.wsd_min_factor = wsd_min_factor
        
        # Validate WSD parameters
        if wsd_decay_type != "none":
            if wsd_stable_steps is None or wsd_decay_steps is None:
                raise ValueError(
                    f"When wsd_decay_type is '{wsd_decay_type}', both wsd_stable_steps and "
                    f"wsd_decay_steps must be provided."
                )
        
        # Identify which param groups are managed by maxP (Linear weights)
        self._managed_indices: list[int] = []
        self._fan_in: list[int] = []

        for i, group in enumerate(optimizer.param_groups):
            if group.get("maxp_managed", False):
                self._managed_indices.append(i)
                self._fan_in.append(int(group["fan_in"]))
        
        # Validate managed param groups match al/bl
        n_managed = len(self._managed_indices)
        if n_managed != len(self.al):
            raise ValueError(
                f"Number of maxp_managed param groups ({n_managed}) must match "
                f"length of al ({len(self.al)}). Use create_param_groups() to set up "
                f"the optimizer correctly."
            )
        
        # Create tracer
        self.tracer = Tracer(
            model,
            sample_size=sample_size,
            bl=self.bl if resample_w0 else None,
            resample_w0=resample_w0,
        )
        
        # Verify tracer found same number of layers
        if len(self.tracer.layer_names) != len(self.al):
            raise ValueError(
                f"Tracer found {len(self.tracer.layer_names)} Linear layers, "
                f"but al has {len(self.al)} elements. Ensure al/bl have one entry "
                f"per nn.Linear layer (excluding attention)."
            )
        
        # State
        self._step_count = 0
        self._cached_cl: list[float] | None = None
        self._cached_rl: list[float] | None = None
        self._initialized = False
        
        # Cached alignment values (updated each solve step)
        self._cached_alpha: list[float] | None = None
        self._cached_omega: list[float] | None = None
        self._cached_u: list[float] | None = None
        
        # WSD state: frozen LRs (before WSD multiplier) for decay phase
        self._frozen_lrs: list[float] | None = None
        self._in_decay_phase = False
        
        # Store initial learning rates for managed groups
        self._initial_lrs = [self.optimizer.param_groups[i]["lr"] for i in self._managed_indices]

    @staticmethod
    def _infer_optimizer_type(optimizer: torch.optim.Optimizer) -> str:
        if isinstance(optimizer, torch.optim.SGD):
            return "sgd"

        if isinstance(optimizer, (
            torch.optim.Adam, torch.optim.AdamW, torch.optim.Adamax,
            torch.optim.NAdam, torch.optim.RAdam
        )):
            return "adam"

        raise ValueError(
            "Could not infer optimizer family. Supported families: Adam (Adam/AdamW/Adamax/NAdam/RAdam) and SGD."
        )
    
    @staticmethod
    def _resolve_parametrization(
        al: list[float] | None,
        bl: list[float] | None,
        parametrization: ParametrizationType | None,
        alignment_assumption: AlignmentType,
        optimizer_type: str,
        n_layers: int,
    ) -> tuple[list[float], list[float], str | None]:
        """
        Resolve ABC parametrization from explicit values or named parametrization.
        
        Returns:
            Tuple of (al, bl, param_name) where param_name is None if custom values used.
        """
        if parametrization is not None:
            if al is not None or bl is not None:
                raise ValueError(
                    "Cannot specify both 'parametrization' and explicit 'al'/'bl' values. "
                    "Use either named parametrization or custom values, not both."
                )
            abc = get_abc_parametrization(
                n_layers=n_layers,
                parametrization=parametrization,
                optimizer=optimizer_type,  # type: ignore
                alignment=alignment_assumption,
            )
            return abc.al, abc.bl, abc.name
        
        # Explicit values
        if al is None or bl is None:
            raise ValueError(
                "Must provide either 'parametrization' or both 'al' and 'bl'. "
                "For common parametrizations, use parametrization='mup', 'sp', 'ntk', or 'mfp'."
            )
        
        if len(al) != len(bl):
            raise ValueError(
                f"Length of al ({len(al)}) must match length of bl ({len(bl)})"
            )
        
        if len(al) != n_layers:
            raise ValueError(
                f"Length of al/bl ({len(al)}) must match number of Linear layers ({n_layers})"
            )
        
        return list(al), list(bl), None
    
    def capture_initial(self, X: torch.Tensor) -> None:
        """
        Capture initial state before training begins.
        
        Must be called once before the first call to step().
        
        Args:
            X: Input batch for forward pass to capture initial activations.
        """
        self.tracer.capture_initial(X)
        self._initialized = True
    
    def _get_wsd_multiplier(self) -> float:
        """
        Compute WSD (Warmup-Stable-Decay) learning rate multiplier.
        
        Returns a multiplier in [wsd_min_factor, 1.0] based on current step:
        - Warmup phase: linear ramp from wsd_min_factor to 1.0
        - Stable phase: constant 1.0
        - Decay phase: cosine or linear decay from 1.0 to wsd_min_factor
        
        Returns:
            Learning rate multiplier to apply on top of base LRs.
        """
        step = self._step_count
        
        # Warmup phase: linear ramp from wsd_min_factor to 1.0
        if step <= self.wsd_warmup_steps:
            if self.wsd_warmup_steps == 0:
                return 1.0
            t = step / self.wsd_warmup_steps
            return self.wsd_min_factor + (1.0 - self.wsd_min_factor) * t
        
        # If WSD decay is disabled, always return 1.0 after warmup
        if self.wsd_decay_type == "none":
            return 1.0
        
        # Stable phase: constant 1.0
        stable_end = self.wsd_warmup_steps + self.wsd_stable_steps
        if step <= stable_end:
            return 1.0
        
        # Decay phase: cosine or linear decay from 1.0 to wsd_min_factor
        decay_progress = (step - stable_end) / self.wsd_decay_steps
        decay_progress = min(decay_progress, 1.0)  # Clamp to [0, 1]
        
        if self.wsd_decay_type == "cosine":
            # Cosine decay: 1.0 -> wsd_min_factor
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return self.wsd_min_factor + (1.0 - self.wsd_min_factor) * cosine_factor
        elif self.wsd_decay_type == "linear":
            # Linear decay: 1.0 -> wsd_min_factor
            return 1.0 - (1.0 - self.wsd_min_factor) * decay_progress
        else:
            return 1.0
    
    def _is_in_decay_phase(self) -> bool:
        """Check if we're in the decay phase of WSD schedule."""
        if self.wsd_decay_type == "none":
            return False
        stable_end = self.wsd_warmup_steps + self.wsd_stable_steps
        return self._step_count > stable_end
    
    def step(self, X: torch.Tensor | None = None) -> None:
        """
        Update learning rates based on current alignment and WSD schedule.
        
        Should be called after each optimizer.step().
        
        If X is provided and we're past solver warmup and on a solve interval step,
        computes alignment and solves LP for new learning rates.
        Otherwise, uses cached learning rates from last solve.
        
        During WSD decay phase, the LP solver stops and per-layer LRs are frozen
        at their last computed values. Only the decay multiplier is applied.
        
        Args:
            X: Input batch for alignment computation. If None, skips
                alignment computation and uses cached values.
        """
        self._step_count += 1
        
        # Get WSD multiplier for this step
        wsd_mult = self._get_wsd_multiplier()
        
        # Check if we're entering decay phase
        if self._is_in_decay_phase() and not self._in_decay_phase:
            # Freeze LRs at current values (before WSD multiplier)
            self._frozen_lrs = [
                self.optimizer.param_groups[i]["lr"] for i in self._managed_indices
            ]
            self._in_decay_phase = True
        
        # During decay phase: apply only WSD multiplier to frozen LRs
        if self._in_decay_phase and self._frozen_lrs is not None:
            for group_idx, frozen_lr in zip(self._managed_indices, self._frozen_lrs):
                self.optimizer.param_groups[group_idx]["lr"] = frozen_lr * wsd_mult
            return
        
        # During solver warmup or if no input provided, apply WSD multiplier to initial LRs
        if self._step_count <= self.solver_warmup_steps or X is None:
            for group_idx, initial_lr in zip(self._managed_indices, self._initial_lrs):
                self.optimizer.param_groups[group_idx]["lr"] = initial_lr * wsd_mult
            return
        
        # Check if we should solve this step
        if self._step_count % self.solve_interval != 0:
            # Use cached values if available, with WSD multiplier
            if self._cached_cl is not None:
                self._apply_cl(self._cached_cl, wsd_mult)
            else:
                # No cached cl yet, apply WSD multiplier to initial LRs
                for group_idx, initial_lr in zip(self._managed_indices, self._initial_lrs):
                    self.optimizer.param_groups[group_idx]["lr"] = initial_lr * wsd_mult
            return
        
        # Ensure initialized
        if not self._initialized:
            raise RuntimeError(
                "MaxPScheduler not initialized. Call capture_initial() before training."
            )
        
        # Capture current state and compute alignment
        with torch.no_grad():
            current = self.tracer.capture(X, step=self._step_count)
            window = self.tracer.window(current)
            alpha, omega, u = compute_alignment(window, norm_mode=self.alignment_norm)
        
        # Cache alignment values
        self._cached_alpha = alpha
        self._cached_omega = omega
        self._cached_u = u
        
        # Solve LP for optimal c_l
        cl, rl = find_c(
            al=self.al,
            bl=self.bl,
            alpha=alpha,
            omega=omega,
            u=u,
            optimizer_type=self.optimizer_type,
            solver=self.solver,
            feature_learning=self.feature_learning,
        )
        
        # Check if solve was successful
        if any(np.isnan(c) for c in cl):
            # Keep previous LRs if solve failed
            return
        
        # Cache and apply with WSD multiplier
        self._cached_cl = cl
        self._cached_rl = rl
        self._apply_cl(cl, wsd_mult)
    
    def _apply_cl(self, cl: list[float], wsd_mult: float = 1.0) -> None:
        """Apply learning rate exponents to managed optimizer param groups.
        
        Args:
            cl: List of c_l exponents from LP solver.
            wsd_mult: WSD multiplier to apply on top of base LRs.
        """
        for idx, (group_idx, c) in enumerate(zip(self._managed_indices, cl)):
            fan_in = self._fan_in[idx]
            lr = self.lr_prefactor * (fan_in ** (-c)) * wsd_mult
            self.optimizer.param_groups[group_idx]["lr"] = lr
    
    def get_last_lr(self) -> list[float]:
        """
        Return last computed learning rate for each managed param group.
        
        Returns:
            List of current learning rates for Linear weight groups.
        """
        return [self.optimizer.param_groups[i]["lr"] for i in self._managed_indices]
    
    def get_alignment(self) -> tuple[list[float] | None, list[float] | None, list[float] | None]:
        """
        Return last computed alignment values (alpha, omega, u).
        
        Returns:
            Tuple of (alpha, omega, u) lists, or (None, None, None) if not computed yet.
            Each list has one entry per managed Linear layer.
        """
        return self._cached_alpha, self._cached_omega, self._cached_u
    
    def get_layer_names(self) -> list[str]:
        """
        Return names of managed Linear layers.
        
        Returns:
            List of layer names in order matching other scheduler outputs.
        """
        return self.tracer.layer_names
    
    @property
    def n_layers(self) -> int:
        """Return number of managed Linear layers."""
        return len(self._managed_indices)
    
    def state_dict(self) -> dict:
        """
        Return scheduler state for checkpointing.
        
        Returns:
            Dict containing scheduler state.
        """
        return {
            "step_count": self._step_count,
            "cached_cl": self._cached_cl,
            "cached_rl": self._cached_rl,
            "cached_alpha": self._cached_alpha,
            "cached_omega": self._cached_omega,
            "cached_u": self._cached_u,
            "initialized": self._initialized,
            "frozen_lrs": self._frozen_lrs,
            "in_decay_phase": self._in_decay_phase,
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load scheduler state from checkpoint.
        
        Args:
            state_dict: State dict from state_dict().
        """
        self._step_count = state_dict["step_count"]
        self._cached_cl = state_dict["cached_cl"]
        self._cached_rl = state_dict["cached_rl"]
        self._cached_alpha = state_dict.get("cached_alpha")
        self._cached_omega = state_dict.get("cached_omega")
        self._cached_u = state_dict.get("cached_u")
        self._initialized = state_dict["initialized"]
        
        # WSD state
        self._frozen_lrs = state_dict.get("frozen_lrs")
        self._in_decay_phase = state_dict.get("in_decay_phase", False)
        
        # Apply cached cl if available, with current WSD multiplier
        if self._cached_cl is not None:
            wsd_mult = self._get_wsd_multiplier()
            self._apply_cl(self._cached_cl, wsd_mult)


class ChainedMaxPScheduler:
    """
    Chains MaxPScheduler with standard PyTorch LR schedulers.
    
    Combines the per-layer learning rate adjustment from MaxPScheduler with
    global decay schedules (e.g., CosineAnnealingLR, LinearLR). On each step:
    
    1. Chained schedulers apply their decay schedules to a reference param group
    2. The relative LR change is computed and applied to MaxPScheduler's lr_prefactor
    3. MaxPScheduler computes per-layer LRs using the updated lr_prefactor
    
    This ensures that external schedulers control the overall learning rate scale
    while MaxP handles per-layer LR ratios based on alignment measurements.
    
    If the schedulers list is empty, this wrapper behaves identically to the
    underlying MaxPScheduler.
    
    Args:
        maxp_scheduler: The MaxPScheduler instance to wrap.
        schedulers: List of PyTorch LR schedulers to chain. All must use the
            same optimizer as maxp_scheduler. Can be empty.
    
    Example:
        >>> # Create MaxP scheduler
        >>> maxp_sched = MaxPScheduler(optimizer, model, parametrization="mup", lr_prefactor=0.1)
        >>> 
        >>> # Create cosine decay scheduler
        >>> cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        >>> 
        >>> # Chain them together
        >>> scheduler = ChainedMaxPScheduler(maxp_sched, [cosine_sched])
        >>> 
        >>> # Use like MaxPScheduler
        >>> scheduler.capture_initial(X_init)
        >>> for X, y in train_loader:
        ...     optimizer.zero_grad()
        ...     loss = criterion(model(X), y)
        ...     loss.backward()
        ...     optimizer.step()
        ...     scheduler.step(X)
    """
    
    def __init__(
        self,
        maxp_scheduler: MaxPScheduler,
        schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    ):
        self.maxp_scheduler = maxp_scheduler
        self.schedulers = list(schedulers)
        self.optimizer = maxp_scheduler.optimizer
        
        # Validate all schedulers share the same optimizer
        for i, sched in enumerate(self.schedulers):
            if sched.optimizer is not self.optimizer:
                raise ValueError(
                    f"Scheduler at index {i} uses a different optimizer. "
                    f"All schedulers must share the same optimizer as MaxPScheduler."
                )
        
        # Store initial lr_prefactor
        self._initial_lr_prefactor = maxp_scheduler.lr_prefactor
        self._last_lr: list[float] = []
        
        # Track managed indices from MaxPScheduler
        self._managed_indices = maxp_scheduler._managed_indices
    
    def capture_initial(self, X: torch.Tensor) -> None:
        """
        Capture initial state before training begins.
        
        Delegates to MaxPScheduler.
        
        Args:
            X: Input batch for forward pass to capture initial activations.
        """
        self.maxp_scheduler.capture_initial(X)
        self._last_lr = self.maxp_scheduler.get_last_lr()
    
    def step(self, X: torch.Tensor | None = None) -> None:
        """
        Update learning rates using chained schedulers and MaxP.
        
        1. Capture current LR from first managed param group
        2. Step all chained schedulers to get the LR change
        3. Apply the relative change to MaxPScheduler's lr_prefactor
        4. MaxPScheduler computes per-layer LRs using the updated lr_prefactor
        
        Args:
            X: Input batch for alignment computation. If None, MaxP skips
                alignment computation and uses cached values.
        """
        # If no chained schedulers, just step MaxP
        if not self.schedulers or len(self._managed_indices) == 0:
            self.maxp_scheduler.step(X)
            self._last_lr = self.maxp_scheduler.get_last_lr()
            return
        
        # Step 1: Capture LR before chained schedulers step
        first_idx = self._managed_indices[0]
        lr_before = self.optimizer.param_groups[first_idx]["lr"]
        
        # Step 2: Step all chained schedulers
        for sched in self.schedulers:
            sched.step()
        
        # Step 3: Compute LR change and apply to lr_prefactor
        lr_after = self.optimizer.param_groups[first_idx]["lr"]
        
        if lr_before != 0:
            lr_change_factor = lr_after / lr_before
            self.maxp_scheduler.lr_prefactor *= lr_change_factor
        
        # Step 4: Let MaxP compute per-layer LRs using updated lr_prefactor
        self.maxp_scheduler.step(X)
        self._last_lr = self.maxp_scheduler.get_last_lr()
    
    def get_last_lr(self) -> list[float]:
        """
        Return last computed learning rate for each managed param group.
        
        Returns:
            List of current learning rates for Linear weight groups.
        """
        return list(self._last_lr)
    
    def get_alignment(self) -> tuple[list[float] | None, list[float] | None, list[float] | None]:
        """
        Return last computed alignment values (alpha, omega, u).
        
        Delegates to the underlying MaxPScheduler.
        
        Returns:
            Tuple of (alpha, omega, u) lists, or (None, None, None) if not computed yet.
        """
        return self.maxp_scheduler.get_alignment()
    
    def get_layer_names(self) -> list[str]:
        """
        Return names of managed Linear layers.
        
        Delegates to the underlying MaxPScheduler.
        
        Returns:
            List of layer names in order matching other scheduler outputs.
        """
        return self.maxp_scheduler.get_layer_names()
    
    @property
    def n_layers(self) -> int:
        """Return number of managed Linear layers."""
        return self.maxp_scheduler.n_layers
    
    def state_dict(self) -> dict:
        """
        Return scheduler state for checkpointing.
        
        Includes state from MaxPScheduler and all chained schedulers.
        
        Returns:
            Dict containing scheduler state.
        """
        return {
            "maxp_scheduler": self.maxp_scheduler.state_dict(),
            "chained_schedulers": [s.state_dict() for s in self.schedulers],
            "initial_lr_prefactor": self._initial_lr_prefactor,
            "current_lr_prefactor": self.maxp_scheduler.lr_prefactor,
            "last_lr": self._last_lr,
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load scheduler state from checkpoint.
        
        Restores state of MaxPScheduler and all chained schedulers.
        
        Args:
            state_dict: State dict from state_dict().
        """
        self.maxp_scheduler.load_state_dict(state_dict["maxp_scheduler"])
        
        chained_states = state_dict.get("chained_schedulers", [])
        for sched, sched_state in zip(self.schedulers, chained_states):
            sched.load_state_dict(sched_state)
        
        self._initial_lr_prefactor = state_dict.get("initial_lr_prefactor", self._initial_lr_prefactor)
        if "current_lr_prefactor" in state_dict:
            self.maxp_scheduler.lr_prefactor = state_dict["current_lr_prefactor"]
        self._last_lr = state_dict.get("last_lr", [])
