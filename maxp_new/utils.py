"""
Utility functions for maxP scheduler.
"""

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn


# Type aliases for parametrization options
ParametrizationType = Literal["sp", "mup", "ntk", "mfp"]
OptimizerType = Literal["adam", "sgd"]
AlignmentType = Literal["full", "no"]


@dataclass
class ABCParametrization:
    """
    Container for ABC parametrization values.
    
    Attributes:
        al: List of a_l exponents (layer multipliers).
        bl: List of b_l exponents (initialization variance).
        cl: List of c_l exponents (learning rate scaling).
        name: Name of the parametrization (e.g., "muP-adam-full").
    """
    
    al: list[float]
    bl: list[float]
    cl: list[float]
    name: str


def get_abc_parametrization(
    n_layers: int,
    parametrization: ParametrizationType | str = "mup",
    optimizer: OptimizerType | str = "adam",
    alignment: AlignmentType | str = "full",
) -> ABCParametrization:
    """
    Generate ABC parametrization values for common neural network scaling schemes.
    
    Supports four classic parametrizations:
    - "sp": Standard Parametrization (PyTorch default)
    - "mup": Maximal Update Parametrization (μP)
    - "ntk": Neural Tangent Kernel parametrization
    - "mfp": Mean Field Parametrization
    
    Each parametrization can be configured for different optimizers and 
    alignment assumptions.
    
    Args:
        n_layers: Number of nn.Linear layers in the model.
        parametrization: Parametrization type. One of:
            - "sp": Standard Parametrization
            - "mup": Maximal Update Parametrization  
            - "ntk": Neural Tangent Kernel
            - "mfp": Mean Field Parametrization
        optimizer: Optimizer family. One of:
            - "adam": Adam/AdamW/Adamax/NAdam/RAdam family
            - "sgd": SGD family
        alignment: Alignment assumption. One of "full" or "no".
    
    Returns:
        ABCParametrization dataclass with al, bl, cl lists and name.
    
    Raises:
        ValueError: If n_layers < 2, or invalid parametrization/optimizer/alignment.
    
    Example:
        >>> params = get_abc_parametrization(n_layers=4, parametrization="mup")
        >>> params.al
        [-0.5, 0.0, 0.0, 0.5]
        >>> params.bl  
        [0.5, 0.5, 0.5, 0.5]
        >>> params.cl
        [0.5, 1.0, 1.0, 0.5]
        >>> params.name
        'mup-adam-full'
    """

    if n_layers < 2:
        raise ValueError(f"n_layers must be >= 2, got {n_layers}")
    
    parametrization = parametrization.lower()  # type: ignore
    optimizer = optimizer.lower()  # type: ignore
    alignment = alignment.lower()  # type: ignore
    
    if parametrization not in ("sp", "mup", "ntk", "mfp"):
        raise ValueError(
            f"Unknown parametrization '{parametrization}'. "
            f"Supported: 'sp', 'mup', 'ntk', 'mfp'"
        )
    
    if optimizer not in ("adam", "sgd"):
        raise ValueError(
            f"Unknown optimizer '{optimizer}'. Supported: 'adam', 'sgd'"
        )
    
    if alignment not in ("full", "no"):
        raise ValueError(
            f"Unknown alignment '{alignment}'. Supported: 'full', 'no'"
        )
    
    name = f"{parametrization}-{optimizer}-{alignment}"
    n_hidden = n_layers - 2  # Number of hidden layers (excluding first and last)
    
    if parametrization == "mup":
        al, bl, cl = _mup_abc(n_hidden, optimizer, alignment)
    elif parametrization == "ntk":
        al, bl, cl = _ntk_abc(n_hidden, optimizer, alignment)
    elif parametrization == "mfp":
        al, bl, cl = _mfp_abc(n_hidden, optimizer, alignment)
    else:  # "sp"
        al, bl, cl = _sp_abc(n_hidden, optimizer, alignment)
    
    return ABCParametrization(al=al, bl=bl, cl=cl, name=name)


def _mup_abc(
    n_hidden: int, optimizer: str, alignment: str
) -> tuple[list[float], list[float], list[float]]:
    """μP (Maximal Update Parametrization) ABC values."""
    al = [-0.5] + [0.0] * n_hidden + [0.5]
    bl = [0.5] + [0.5] * n_hidden + [0.5]
    
    if alignment == "full":
        if optimizer == "sgd":
            cl = [0.0] + [0.0] * n_hidden + [0.0]
        else:  # adam
            cl = [0.5] + [1.0] * n_hidden + [0.5]
    else:  # no alignment
        if optimizer == "sgd":
            cl = [0.0] + [-0.5] * n_hidden + [0.0]
        else:  # adam
            cl = [0.5] + [0.5] * n_hidden + [0.0]
    
    return al, bl, cl


def _ntk_abc(
    n_hidden: int, optimizer: str, alignment: str
) -> tuple[list[float], list[float], list[float]]:
    """NTK (Neural Tangent Kernel) ABC values."""
    al = [0.0] + [0.5] * n_hidden + [0.5]
    bl = [0.0] + [0.0] * n_hidden + [0.0]
    
    if alignment == "full":
        if optimizer == "sgd":
            cl = [-0.5] + [-0.5] * n_hidden + [0.0]
        else:  # adam
            cl = [0.0] + [0.5] * n_hidden + [0.5]
    else:  # no alignment
        if optimizer == "sgd":
            cl = [-0.5] + [-1.0] * n_hidden + [-0.5]
        else:  # adam
            cl = [0.0] + [0.0] * n_hidden + [0.0]
    
    return al, bl, cl


def _mfp_abc(
    n_hidden: int, optimizer: str, alignment: str
) -> tuple[list[float], list[float], list[float]]:
    """MFP (Mean Field Parametrization) ABC values."""
    al = [0.0] + [0.5] * n_hidden + [1.0]
    bl = [0.0] + [0.0] * n_hidden + [0.0]
    
    if alignment == "full":
        if optimizer == "sgd":
            cl = [-1.0] + [-1.0] * n_hidden + [-1.0]
        else:  # adam
            cl = [0.0] + [0.5] * n_hidden + [0.0]
    else:  # no alignment
        if optimizer == "sgd":
            cl = [-1.0] + [-1.5] * n_hidden + [-1.0]
        else:  # adam
            cl = [0.0] + [0.0] * n_hidden + [0.5]
    
    return al, bl, cl


def _sp_abc(
    n_hidden: int, optimizer: str, alignment: str
) -> tuple[list[float], list[float], list[float]]:
    """SP (Standard Parametrization) ABC values."""
    al = [0.0] + [0.0] * n_hidden + [0.0]
    bl = [0.0] + [0.5] * n_hidden + [0.5]
    
    if alignment == "full":
        if optimizer == "sgd":
            cl = [-0.5] + [0.5] * n_hidden + [1.0]
        else:  # adam
            cl = [0.0] + [1.0] * n_hidden + [1.0]
    else:  # no alignment
        if optimizer == "sgd":
            cl = [-0.5] + [0.0] * n_hidden + [0.5]
        else:  # adam
            cl = [0.0] + [0.5] * n_hidden + [0.5]
    
    return al, bl, cl


class ScaledModule(nn.Module):
    """Wraps any module, scaling its output by a fixed factor."""

    def __init__(self, module: nn.Module, scale: float):
        super().__init__()
        self.module = module
        self.scale = scale

    def forward(self, x):
        return self.scale * self.module(x)


def _set_nested_attr(model: nn.Module, path: str, value: nn.Module) -> None:
    """Set a nested attribute on a module using dot-separated path."""
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def initialize_abc_weights(
    model: nn.Module,
    al: list[float] | None = None,
    bl: list[float] | None = None,
    parametrization: ParametrizationType | str | None = None,
    optimizer: OptimizerType | str = "adam",
    alignment: AlignmentType | str = "full",
    std_prefactor: float = 1.0,
    apply_multipliers: bool = True,
) -> nn.Module:
    """
    Initialize Linear layer weights according to ABC parametrization.
    
    Re-initializes weights of all nn.Linear layers (excluding attention) using:
        std = std_prefactor * (n^{-2b_l})^0.5
    
    where n is the layer's fan-in (input features).
    
    Optionally applies layer multipliers by wrapping Linear layers with
    ScaledLinear modules that multiply outputs by n^{-a_l}.
    
    There are two ways to specify the parametrization:
    
    1. **Named parametrization**: Pass `parametrization="mup"` (or "sp", "ntk", "mfp")
       and the function will automatically generate appropriate al/bl values.
       
    2. **Custom values**: Pass explicit `al` and `bl` lists for fine-grained control.
    
    Args:
        model: PyTorch model to initialize. Modified in-place.
        al: List of a_l exponents (layer multipliers), one per Linear layer.
            Required if `parametrization` is not provided.
        bl: List of b_l exponents (initialization variance), one per Linear layer.
            Required if `parametrization` is not provided.
        parametrization: Named parametrization type. One of:
            - "mup": Maximal Update Parametrization (μP)
            - "sp": Standard Parametrization (PyTorch default)
            - "ntk": Neural Tangent Kernel parametrization  
            - "mfp": Mean Field Parametrization
            If provided, al and bl are automatically generated.
        optimizer: Optimizer family for named parametrization (only used when
            parametrization is provided). One of "adam" or "sgd".
        alignment: Alignment assumption for named parametrization (only used
            when parametrization is provided). One of "full" or "no".
        std_prefactor: Base standard deviation multiplier. Default: 1.0.
        apply_multipliers: If True, wrap Linear layers with ScaledLinear to
            apply layer multipliers (n^{-a_l}). If False, only re-initialize
            weights without applying multipliers. Default: True.
    
    Returns:
        The modified model (same object, returned for convenience).
    
    Raises:
        ValueError: If neither al/bl nor parametrization is provided.
        ValueError: If both al/bl and parametrization are provided.
        ValueError: If length of al/bl doesn't match number of Linear layers.
    
    Example using named parametrization:
        >>> model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
        >>> initialize_abc_weights(model, parametrization="mup")
        >>> # Weights are now initialized with muP scaling
    
    Example using custom values:
        >>> model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
        >>> al = [-0.5, 0.5]  # Layer multiplier exponents
        >>> bl = [0.5, 0.5]   # Initialization variance exponents
        >>> initialize_abc_weights(model, al=al, bl=bl)
    
    Example without multipliers (for models that handle multipliers internally):
        >>> model = MyCustomMLP()  # Has its own layer multiplier logic
        >>> initialize_abc_weights(model, parametrization="mup", apply_multipliers=False)
    """
    
    layers = get_linear_layers(model)
    n_layers = len(layers)
    
    # Resolve parametrization
    if parametrization is not None:
        if al is not None or bl is not None:
            raise ValueError(
                "Cannot specify both 'parametrization' and explicit 'al'/'bl' values. "
                "Use either named parametrization or custom values, not both."
            )
        abc = get_abc_parametrization(
            n_layers=n_layers,
            parametrization=parametrization,
            optimizer=optimizer,
            alignment=alignment,
        )
        al = abc.al
        bl = abc.bl
    elif al is None or bl is None:
        raise ValueError(
            "Must provide either 'parametrization' or both 'al' and 'bl'. "
            "For common parametrizations, use parametrization='mup', 'sp', 'ntk', or 'mfp'."
        )
    
    if len(al) != n_layers or len(bl) != n_layers:
        raise ValueError(
            f"Length of al ({len(al)}) and bl ({len(bl)}) must match "
            f"number of Linear layers ({n_layers})"
        )
    
    for (name, linear), a, b in zip(layers, al, bl):
        fan_in = linear.weight.shape[1]
        
        # Re-initialize weights with ABC scaling: std = std_prefactor * (n^{-2b})^0.5
        val_l = fan_in ** (-2 * b)
        std = std_prefactor * (val_l ** 0.5)
        torch.nn.init.normal_(linear.weight, mean=0.0, std=std)
        
        # Re-initialize bias to zero if present
        if linear.bias is not None:
            torch.nn.init.zeros_(linear.bias)
        
        # Apply layer multiplier by wrapping with ScaledLinear
        if apply_multipliers and a != 0.0:
            scale = fan_in ** (-a)
            scaled = ScaledLinear(linear, scale)
            _set_nested_attr(model, name, scaled)
    
    return model


def get_linear_layers(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    """
    Get all nn.Linear layers from a model, excluding attention layers.
    
    Args:
        model: PyTorch model to inspect.
    
    Returns:
        List of (name, module) tuples for each nn.Linear layer.
    """

    layers: list[tuple[str, nn.Linear]] = []
    
    def traverse(module: nn.Module, prefix: str) -> None:
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Skip attention layers
            if "attn" in name.lower():
                continue
            
            if isinstance(child, nn.Linear):
                layers.append((full_name, child))
            else:
                traverse(child, full_name)
    
    traverse(model, "")
    return layers


def create_param_groups(
    model: nn.Module,
    lr_prefactor: float,
    cl: list[float] | None = None,
    parametrization: ParametrizationType | str | None = None,
    optimizer: OptimizerType | str = "adam",
    alignment: AlignmentType | str = "full",
    other_lr: float | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Create optimizer parameter groups with per-layer learning rates.
    
    Creates one parameter group per nn.Linear layer with initial learning rate
    set according to ABC parametrization using that layer's fan-in n_l:
        lr_l = lr_prefactor * n_l^{-c_l}
    
    All other parameters (biases, LayerNorm, embeddings, etc.) are collected
    into a single "other" group with a constant learning rate that is not
    adjusted by the maxP scheduler.
    
    There are two ways to specify the initial c_l values:
    
    1. **Named parametrization**: Pass `parametrization="mup"` (or "sp", "ntk", "mfp")
       and the function will automatically generate appropriate cl values.
       
    2. **Custom values**: Pass explicit `cl` list for fine-grained control.
    
    Args:
        model: PyTorch model.
        lr_prefactor: Base learning rate multiplier.
        cl: List of initial c_l learning-rate exponents, one per nn.Linear layer.
            Required if `parametrization` is not provided.
        parametrization: Named parametrization type. One of:
            - "mup": Maximal Update Parametrization (μP)
            - "sp": Standard Parametrization (PyTorch default)
            - "ntk": Neural Tangent Kernel parametrization  
            - "mfp": Mean Field Parametrization
            If provided, cl is automatically generated.
        optimizer: Optimizer family for named parametrization. One of:
            - "adam": Adam/AdamW/Adamax/NAdam/RAdam family
            - "sgd": SGD family
            Only used when `parametrization` is provided.
        alignment: Alignment assumption for named parametrization. One of "full" or "no".
            Only used when `parametrization` is provided.
        other_lr: Learning rate for non-Linear parameters. If None, defaults
            to lr_prefactor. This LR remains constant throughout training.
        **kwargs: Additional arguments passed to each param group
            (e.g., weight_decay, betas).
    
    Returns:
        List of parameter group dicts suitable for torch.optim constructors.
        Linear weight groups have "maxp_managed": True; the "other" group
        (if non-empty) has "maxp_managed": False.
    
    Raises:
        ValueError: If neither cl nor parametrization is provided, or both are.
        ValueError: If length of cl doesn't match number of Linear layers.
    
    Example using named parametrization:
        >>> model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
        >>> param_groups = create_param_groups(model, lr_prefactor=0.1, parametrization="mup")
        >>> optimizer = torch.optim.AdamW(param_groups)
    
    Example using custom cl values:
        >>> model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
        >>> cl = [0.0, 0.5]  # c_l for each Linear layer
        >>> param_groups = create_param_groups(model, lr_prefactor=0.1, cl=cl)
        >>> optimizer = torch.optim.AdamW(param_groups)
    """

    layers = get_linear_layers(model)
    n_layers = len(layers)
    
    # Resolve cl from parametrization or explicit values
    if parametrization is not None:
        if cl is not None:
            raise ValueError(
                "Cannot specify both 'parametrization' and explicit 'cl' values. "
                "Use either named parametrization or custom values, not both."
            )
        abc = get_abc_parametrization(
            n_layers=n_layers,
            parametrization=parametrization,  # type: ignore
            optimizer=optimizer,  # type: ignore
            alignment=alignment,  # type: ignore
        )
        cl = abc.cl
    elif cl is None:
        raise ValueError(
            "Must provide either 'cl' or 'parametrization'. "
            "For common parametrizations, use parametrization='mup', 'sp', 'ntk', or 'mfp'."
        )
    
    if n_layers != len(cl):
        raise ValueError(
            f"Length of cl ({len(cl)}) must match number of Linear layers ({n_layers})"
        )
    
    if other_lr is None:
        other_lr = lr_prefactor
    
    # Collect Linear weight parameter ids for exclusion from "other" group
    linear_weight_ids: set[int] = set()
    param_groups: list[dict[str, Any]] = []

    for (name, module), c in zip(layers, cl):
        fan_in = int(module.weight.shape[1])
        lr = lr_prefactor * (fan_in ** (-c))
        linear_weight_ids.add(id(module.weight))
        group = {
            "params": [module.weight],
            "lr": lr,
            "layer_name": name,
            "fan_in": fan_in,
            "c": float(c),
            "maxp_managed": True,
            **kwargs,
        }
        param_groups.append(group)
    
    # Collect all other parameters (biases, LayerNorm, embeddings, etc.)
    other_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in linear_weight_ids
    ]
    
    if other_params:
        other_group = {
            "params": other_params,
            "lr": other_lr,
            "layer_name": "_other",
            "maxp_managed": False,
            **kwargs,
        }
        param_groups.append(other_group)
    
    return param_groups
