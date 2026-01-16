"""
MaxP Learning Rate Scheduler.

Dynamically adjusts per-layer learning rates using alignment measurements
between initial and current weights/activations, solved via linear programming.
"""

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
        warmup_steps: Number of steps before engaging the LP solver.
            During warmup, learning rates remain at their initial values.
            Default: 100.
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
        warmup_steps: int = 100,
        sample_size: int = 32,
        solve_interval: int = 1,
        solver: plp.LpSolver | None = None,
        resample_w0: bool = False,
        feature_learning: bool = False,
        alignment_norm: str = "rms",
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
        self.warmup_steps = warmup_steps
        self.sample_size = sample_size
        self.solve_interval = solve_interval
        self.solver = solver
        self.resample_w0 = resample_w0
        self.feature_learning = feature_learning
        self.alignment_norm = alignment_norm
        
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
    
    def step(self, X: torch.Tensor | None = None) -> None:
        """
        Update learning rates based on current alignment.
        
        Should be called after each optimizer.step().
        
        If X is provided and we're past warmup and on a solve interval step,
        computes alignment and solves LP for new learning rates.
        Otherwise, uses cached learning rates from last solve.
        
        Args:
            X: Input batch for alignment computation. If None, skips
                alignment computation and uses cached values.
        """

        self._step_count += 1
        
        # During warmup or if no input provided, keep current LRs
        if self._step_count <= self.warmup_steps or X is None:
            return
        
        # Check if we should solve this step
        if self._step_count % self.solve_interval != 0:
            # Use cached values if available
            if self._cached_cl is not None:
                self._apply_cl(self._cached_cl)
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
        
        # Cache and apply
        self._cached_cl = cl
        self._cached_rl = rl
        self._apply_cl(cl)
    
    def _apply_cl(self, cl: list[float]) -> None:
        """Apply learning rate exponents to managed optimizer param groups."""
        for idx, (group_idx, c) in enumerate(zip(self._managed_indices, cl)):
            fan_in = self._fan_in[idx]
            lr = self.lr_prefactor * (fan_in ** (-c))
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
        
        # Apply cached cl if available
        if self._cached_cl is not None:
            self._apply_cl(self._cached_cl)


class ChainedMaxPScheduler:
    """
    Chains MaxPScheduler with standard PyTorch LR schedulers.
    
    Combines the per-layer learning rate adjustment from MaxPScheduler with
    global decay schedules (e.g., CosineAnnealingLR, LinearLR). On each step:
    
    1. MaxPScheduler computes per-layer LRs based on alignment measurements
    2. Chained schedulers apply multiplicative decay factors
    3. Final LRs preserve MaxP's per-layer ratios while applying global decay
    
    The decay factor is computed from the first managed param group and applied
    uniformly to all managed groups, ensuring per-layer LR ratios are preserved.
    
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
        
        # Store base LRs (will be captured after capture_initial)
        self._base_lrs: list[float] | None = None
        self._maxp_lrs: list[float] | None = None
        self._last_lr: list[float] = []
        
        # Track managed indices from MaxPScheduler
        self._managed_indices = maxp_scheduler._managed_indices
    
    def capture_initial(self, X: torch.Tensor) -> None:
        """
        Capture initial state before training begins.
        
        Delegates to MaxPScheduler and captures base LRs for decay computation.
        
        Args:
            X: Input batch for forward pass to capture initial activations.
        """
        self.maxp_scheduler.capture_initial(X)
        # Capture base LRs after MaxP initialization (before any decay)
        self._base_lrs = [self.optimizer.param_groups[i]["lr"] for i in self._managed_indices]
        self._last_lr = list(self._base_lrs)
    
    def step(self, X: torch.Tensor | None = None) -> None:
        """
        Update learning rates using MaxP and apply decay from chained schedulers.
        
        1. MaxPScheduler computes per-layer LRs based on alignment
        2. Chained schedulers apply their decay schedules
        3. Decay factor is computed and applied uniformly to preserve LR ratios
        
        Args:
            X: Input batch for alignment computation. If None, MaxP skips
                alignment computation and uses cached values.
        """
        # Step 1: Let MaxP compute per-layer LRs
        self.maxp_scheduler.step(X)
        self._maxp_lrs = self.maxp_scheduler.get_last_lr()
        
        # If no chained schedulers, we're done
        if not self.schedulers:
            self._last_lr = list(self._maxp_lrs)
            return
        
        # Step 2: Step all chained schedulers
        for sched in self.schedulers:
            sched.step()
        
        # Step 3: Compute decay factor from first managed param group
        # After chained schedulers step, the optimizer has decayed LRs
        if self._base_lrs is None or len(self._managed_indices) == 0:
            self._last_lr = list(self._maxp_lrs)
            return
        
        first_idx = self._managed_indices[0]
        current_lr = self.optimizer.param_groups[first_idx]["lr"]
        base_lr = self._base_lrs[0]
        
        # Avoid division by zero
        if base_lr == 0:
            decay_factor = 1.0
        else:
            decay_factor = current_lr / base_lr
        
        # Step 4: Apply decay factor to MaxP's per-layer LRs
        final_lrs = []
        for idx, (group_idx, maxp_lr) in enumerate(zip(self._managed_indices, self._maxp_lrs)):
            final_lr = maxp_lr * decay_factor
            self.optimizer.param_groups[group_idx]["lr"] = final_lr
            final_lrs.append(final_lr)
        
        self._last_lr = final_lrs
    
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
            "base_lrs": self._base_lrs,
            "maxp_lrs": self._maxp_lrs,
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
        
        self._base_lrs = state_dict.get("base_lrs")
        self._maxp_lrs = state_dict.get("maxp_lrs")
        self._last_lr = state_dict.get("last_lr", [])
