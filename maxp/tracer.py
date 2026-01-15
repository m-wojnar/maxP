"""
Tracer module for capturing activations and weights from nn.Linear layers.

Provides forward hooks to record initial and current states for alignment computation.
"""

import contextlib
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class LayerSnapshot:
    """Snapshot of a single layer's state at a point in time."""
    
    input: Tensor | None = None
    """Input activations to the layer (batch, features)."""
    
    output: Tensor | None = None
    """Output activations from the layer (batch, features)."""
    
    weight: Tensor | None = None
    """Current weight matrix (out_features, in_features)."""


@dataclass
class StepTrace:
    """Collection of layer snapshots at a specific training step."""
    
    step: int
    """Training step number."""
    
    layers: dict[str, LayerSnapshot] = field(default_factory=dict)
    """Mapping from layer name to its snapshot."""


@dataclass
class TraceWindow:
    """Window containing initial and current traces for alignment computation."""
    
    init: StepTrace
    """Trace from initialization (step 0)."""
    
    current: StepTrace
    """Trace from the current step."""
    
    n_layers: int
    """Number of tracked layers."""
    
    fan_in: list[int]
    """Per-layer fan-in (in_features) for each tracked layer."""
    
    layer_names: list[str]
    """Ordered list of layer names."""
    
    bl: list[float] | None = None
    """b_l exponents for ABC parametrization."""
    
    resample_w0: bool = False
    """Whether to resample w0 instead of storing it."""


class Tracer:
    """
    Tracer for capturing activations and weights from nn.Linear layers.
    
    Registers forward hooks on all nn.Linear modules (excluding attention layers)
    to capture input/output activations and weights for alignment computation.
    
    Args:
        model: PyTorch model to trace.
        sample_size: Maximum number of samples to keep from each batch.
            If None, keeps all samples. Default: 32.
        bl: List of b_l exponents for ABC parametrization (one per layer).
        resample_w0: If True, don't store initial weights; instead store bl
            for resampling during alignment computation. Saves memory but
            introduces variance in alignment estimates. Default: False.
    """
    
    def __init__(
        self,
        model: nn.Module,
        sample_size: int | None = 32,
        bl: list[float] | None = None,
        resample_w0: bool = False,
    ):
        self.model = model
        self.sample_size = sample_size
        self.bl = bl
        self.resample_w0 = resample_w0
        
        # Collect linear modules in definition order
        self.layer_names: list[str] = []
        self.modules: list[nn.Linear] = []
        
        self._collect_linear_layers(model, prefix="model")
        
        if not self.modules:
            raise RuntimeError("Tracer: no nn.Linear layers found in model.")
        
        # Compute fan_in once (static per layer)
        self.fan_in: list[int] = [mod.in_features for mod in self.modules]
        
        # Validate bl length if provided
        if bl is not None and len(bl) != len(self.modules):
            raise ValueError(
                f"Length of bl ({len(bl)}) must match number of Linear layers ({len(self.modules)})"
            )

        if self.resample_w0 and self.bl is None:
            raise ValueError("Tracer requires bl when resample_w0=True")
        
        # Hook state
        self._armed = False
        self._snap: dict[str, LayerSnapshot] = {}

        # Whether we're currently capturing the initialization trace
        self._capturing_initial = False
        
        # Register hooks
        self._handles = [
            mod.register_forward_hook(self._make_hook(name, idx))
            for idx, (name, mod) in enumerate(zip(self.layer_names, self.modules))
        ]
        
        # Initial trace storage
        self.initial: StepTrace | None = None
    
    def _collect_linear_layers(self, module: nn.Module, prefix: str) -> None:
        """Recursively collect nn.Linear layers, skipping attention."""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}"
            
            # Skip attention layers
            if "attn" in name.lower():
                continue
            
            if isinstance(child, nn.Linear):
                self.layer_names.append(full_name)
                self.modules.append(child)
            else:
                self._collect_linear_layers(child, full_name)
    
    def _make_hook(self, name: str, layer_idx: int):
        """Create a forward hook for a specific layer."""
        def hook(module: nn.Linear, inputs: tuple[Tensor, ...], output: Tensor):
            if not self._armed:
                return
            
            x = inputs[0]
            out = output
            
            # Limit sample size
            if self.sample_size is not None:
                x = x[:self.sample_size]
                out = out[:self.sample_size]
            
            weight: Tensor | None
            if self.resample_w0 and self._capturing_initial:
                # When resampling, we don't need to store initial weights at all.
                weight = None
            else:
                weight = module.weight.detach().cpu().clone().contiguous()

            self._snap[name] = LayerSnapshot(
                input=x.detach().cpu().clone().contiguous(),
                output=out.detach().cpu().clone().contiguous(),
                weight=weight,
            )
        
        return hook
    
    @contextlib.contextmanager
    def _trace_context(self):
        """Context manager to arm/disarm tracing."""
        self._armed = True
        self._snap = {}
        try:
            yield
        finally:
            self._armed = False
    
    @torch.no_grad()
    def capture_initial(self, X: Tensor) -> StepTrace:
        """
        Capture initial state before training begins.
        
        Must be called once before training starts. Stores initial weights
        (unless resample_w0=True) and runs a forward pass to capture
        initial activations.
        
        Args:
            X: Input batch for the forward pass.
        
        Returns:
            StepTrace containing the initial layer snapshots.
        """
        # Capture initial activations (and weights only if resample_w0=False)
        self._capturing_initial = True
        try:
            with self._trace_context():
                _ = self.model(X)
        finally:
            self._capturing_initial = False
        
        self.initial = StepTrace(step=0, layers=dict(self._snap))
        return self.initial
    
    @torch.no_grad()
    def capture(self, X: Tensor, step: int = 0) -> StepTrace:
        """
        Capture current state at a training step.
        
        Args:
            X: Input batch for the forward pass.
            step: Current training step number.
        
        Returns:
            StepTrace containing the current layer snapshots.
        """
        with self._trace_context():
            _ = self.model(X)
        
        return StepTrace(step=step, layers=dict(self._snap))
    
    def window(self, current: StepTrace) -> TraceWindow:
        """
        Create a TraceWindow from initial and current traces.
        
        Args:
            current: Current step trace from capture().
        
        Returns:
            TraceWindow containing both initial and current traces.
        
        Raises:
            RuntimeError: If capture_initial() hasn't been called.
        """
        
        if self.initial is None:
            raise RuntimeError("Tracer.initial not set; call capture_initial() first.")
        
        return TraceWindow(
            init=self.initial,
            current=current,
            n_layers=len(self.layer_names),
            fan_in=self.fan_in,
            layer_names=self.layer_names,
            bl=self.bl,
            resample_w0=self.resample_w0,
        )
    
    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
    
    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_hooks()
