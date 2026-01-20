"""
Utility functions for maxP scheduler.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import torch
import torch.nn as nn


# Type aliases for parametrization options
ParametrizationType = Literal["sp", "mup", "ntk", "mfp"]
OptimizerType = Literal["adam", "sgd"]
AlignmentType = Literal["full", "no"]


class SemanticRole(Enum):
    """
    Semantic roles for layers in the network, determining their parametrization.

    Based on the paper:
    - EMBEDDING: embeddings, positional embeddings, LayerNorm scale parameters
    - HIDDEN: MLP layers, Q/K/V projections, attention output projections
    - READOUT: the final output layer (last LINEAR in the network)
    """
    EMBEDDING = "embedding"   # al + bl = 0.0 constraint
    HIDDEN = "hidden"         # al + bl = 0.5 constraint
    READOUT = "readout"       # al + bl >= 0.5 constraint


def get_semantic_roles(layer_infos: list["LayerInfo"]) -> list[SemanticRole]:
    """
    Extract semantic roles from LayerInfo objects.

    This is a convenience function that extracts the semantic_role attribute
    from each LayerInfo. The roles are assigned during get_managed_layers().

    Args:
        layer_infos: List of LayerInfo objects from get_managed_layers().

    Returns:
        List of SemanticRole for each layer.

    Example:
        >>> layers = get_managed_layers(model)
        >>> roles = get_semantic_roles(layers)
        >>> for l, r in zip(layers, roles):
        ...     print(f"{l.name}: {r.value}")
    """
    return [info.semantic_role for info in layer_infos]


@dataclass
class LayerInfo:
    """
    Information about a managed layer.

    Attributes:
        name: Fully qualified name of the layer (e.g., "transformer.block.0.attn.q").
        module: The nn.Linear, nn.Embedding module, or nn.Parameter tensor.
        semantic_role: Semantic role of the layer (EMBEDDING, HIDDEN, or READOUT).
        fan_in: Input dimension (in_features for Linear, embedding_dim for Embedding,
            last dim for Parameter).
        fan_out: Output dimension (out_features for Linear, num_embeddings for Embedding,
            product of other dims for Parameter).
    """
    name: str
    module: nn.Module | nn.Parameter
    semantic_role: SemanticRole
    fan_in: int
    fan_out: int


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
    semantic_roles: list[SemanticRole],
    parametrization: ParametrizationType | str = "mup",
    optimizer: OptimizerType | str = "adam",
    alignment: AlignmentType | str = "full",
) -> ABCParametrization:
    """
    Generate ABC parametrization values based on semantic roles.

    Supports four classic parametrizations:
    - "sp": Standard Parametrization (PyTorch default)
    - "mup": Maximal Update Parametrization (μP)
    - "ntk": Neural Tangent Kernel parametrization
    - "mfp": Mean Field Parametrization
    
    Each parametrization can be configured for different optimizers and 
    alignment assumptions.
    
    Args:
        semantic_roles: List of SemanticRole for each layer (from get_semantic_roles()).
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
        ValueError: If semantic_roles is empty, has no READOUT, or invalid options.
    
    Example:
        >>> from maxp.utils import SemanticRole, get_abc_parametrization
        >>> roles = [SemanticRole.EMBEDDING, SemanticRole.HIDDEN, SemanticRole.HIDDEN, SemanticRole.READOUT]
        >>> params = get_abc_parametrization(roles, parametrization="mup")
        >>> params.al
        [-0.5, 0.0, 0.0, 0.5]
        >>> params.bl  
        [0.5, 0.5, 0.5, 0.5]
    """
    if not semantic_roles:
        raise ValueError("semantic_roles cannot be empty")
    
    if SemanticRole.READOUT not in semantic_roles:
        raise ValueError("At least one layer must have SemanticRole.READOUT")
    
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
    
    # Get (a, b, c) values for each semantic role
    if parametrization == "mup":
        role_values = _mup_values(optimizer, alignment)
    elif parametrization == "ntk":
        role_values = _ntk_values(optimizer, alignment)
    elif parametrization == "mfp":
        role_values = _mfp_values(optimizer, alignment)
    else:  # "sp"
        role_values = _sp_values(optimizer, alignment)
    
    # Build al, bl, cl lists by iterating over semantic roles
    al: list[float] = []
    bl: list[float] = []
    cl: list[float] = []
    
    for role in semantic_roles:
        a, b, c = role_values[role]
        al.append(a)
        bl.append(b)
        cl.append(c)
    
    return ABCParametrization(al=al, bl=bl, cl=cl, name=name)


def _mup_values(
    optimizer: str, alignment: str
) -> dict[SemanticRole, tuple[float, float, float]]:
    """
    μP (Maximal Update Parametrization) ABC values by semantic role.
    
    Returns:
        Dict mapping SemanticRole to (a, b, c) tuple.
    """
    # a, b values (same for all alignment/optimizer combinations)
    # EMBEDDING: al + bl = 0.0 (stability at init for first layer)
    # HIDDEN: al + bl = 0.5 (stability at init for hidden layers)
    # READOUT: al + bl = 1.0 (>= 0.5, stability at init for output layer)
    
    if alignment == "full":
        if optimizer == "sgd":
            return {
                SemanticRole.EMBEDDING: (-0.5, 0.5, 0.0),
                SemanticRole.HIDDEN: (0.0, 0.5, 0.0),
                SemanticRole.READOUT: (0.5, 0.5, 0.0),
            }
        else:  # adam
            return {
                SemanticRole.EMBEDDING: (-0.5, 0.5, 0.5),
                SemanticRole.HIDDEN: (0.0, 0.5, 1.0),
                SemanticRole.READOUT: (0.5, 0.5, 0.5),
            }
    else:  # no alignment
        if optimizer == "sgd":
            return {
                SemanticRole.EMBEDDING: (-0.5, 0.5, 0.0),
                SemanticRole.HIDDEN: (0.0, 0.5, -0.5),
                SemanticRole.READOUT: (0.5, 0.5, 0.0),
            }
        else:  # adam
            return {
                SemanticRole.EMBEDDING: (-0.5, 0.5, 0.5),
                SemanticRole.HIDDEN: (0.0, 0.5, 0.5),
                SemanticRole.READOUT: (0.5, 0.5, 0.0),
            }


def _ntk_values(
    optimizer: str, alignment: str
) -> dict[SemanticRole, tuple[float, float, float]]:
    """
    NTK (Neural Tangent Kernel) ABC values by semantic role.
    
    Returns:
        Dict mapping SemanticRole to (a, b, c) tuple.
    """
    # NTK: al + bl = 0.0 for embedding, 0.5 for hidden, 0.5 for readout
    
    if alignment == "full":
        if optimizer == "sgd":
            return {
                SemanticRole.EMBEDDING: (0.0, 0.0, -0.5),
                SemanticRole.HIDDEN: (0.5, 0.0, -0.5),
                SemanticRole.READOUT: (0.5, 0.0, 0.0),
            }
        else:  # adam
            return {
                SemanticRole.EMBEDDING: (0.0, 0.0, 0.0),
                SemanticRole.HIDDEN: (0.5, 0.0, 0.5),
                SemanticRole.READOUT: (0.5, 0.0, 0.5),
            }
    else:  # no alignment
        if optimizer == "sgd":
            return {
                SemanticRole.EMBEDDING: (0.0, 0.0, -0.5),
                SemanticRole.HIDDEN: (0.5, 0.0, -1.0),
                SemanticRole.READOUT: (0.5, 0.0, -0.5),
            }
        else:  # adam
            return {
                SemanticRole.EMBEDDING: (0.0, 0.0, 0.0),
                SemanticRole.HIDDEN: (0.5, 0.0, 0.0),
                SemanticRole.READOUT: (0.5, 0.0, 0.0),
            }


def _mfp_values(
    optimizer: str, alignment: str
) -> dict[SemanticRole, tuple[float, float, float]]:
    """
    MFP (Mean Field Parametrization) ABC values by semantic role.
    
    Returns:
        Dict mapping SemanticRole to (a, b, c) tuple.
    """
    # MFP: al + bl = 0.0 for embedding, 0.5 for hidden, 1.0 for readout
    
    if alignment == "full":
        if optimizer == "sgd":
            return {
                SemanticRole.EMBEDDING: (0.0, 0.0, -1.0),
                SemanticRole.HIDDEN: (0.5, 0.0, -1.0),
                SemanticRole.READOUT: (1.0, 0.0, -1.0),
            }
        else:  # adam
            return {
                SemanticRole.EMBEDDING: (0.0, 0.0, 0.0),
                SemanticRole.HIDDEN: (0.5, 0.0, 0.5),
                SemanticRole.READOUT: (1.0, 0.0, 0.0),
            }
    else:  # no alignment
        if optimizer == "sgd":
            return {
                SemanticRole.EMBEDDING: (0.0, 0.0, -1.0),
                SemanticRole.HIDDEN: (0.5, 0.0, -1.5),
                SemanticRole.READOUT: (1.0, 0.0, -1.0),
            }
        else:  # adam
            return {
                SemanticRole.EMBEDDING: (0.0, 0.0, 0.0),
                SemanticRole.HIDDEN: (0.5, 0.0, 0.0),
                SemanticRole.READOUT: (1.0, 0.0, 0.5),
            }


def _sp_values(
    optimizer: str, alignment: str
) -> dict[SemanticRole, tuple[float, float, float]]:
    """
    SP (Standard Parametrization) ABC values by semantic role.
    
    Returns:
        Dict mapping SemanticRole to (a, b, c) tuple.
    """
    # SP: al + bl = 0.0 for embedding, 0.5 for hidden, 0.5 for readout
    
    if alignment == "full":
        if optimizer == "sgd":
            return {
                SemanticRole.EMBEDDING: (0.0, 0.0, -0.5),
                SemanticRole.HIDDEN: (0.0, 0.5, 0.5),
                SemanticRole.READOUT: (0.0, 0.5, 1.0),
            }
        else:  # adam
            return {
                SemanticRole.EMBEDDING: (0.0, 0.0, 0.0),
                SemanticRole.HIDDEN: (0.0, 0.5, 1.0),
                SemanticRole.READOUT: (0.0, 0.5, 1.0),
            }
    else:  # no alignment
        if optimizer == "sgd":
            return {
                SemanticRole.EMBEDDING: (0.0, 0.0, -0.5),
                SemanticRole.HIDDEN: (0.0, 0.5, 0.0),
                SemanticRole.READOUT: (0.0, 0.5, 0.5),
            }
        else:  # adam
            return {
                SemanticRole.EMBEDDING: (0.0, 0.0, 0.0),
                SemanticRole.HIDDEN: (0.0, 0.5, 0.5),
                SemanticRole.READOUT: (0.0, 0.5, 0.5),
            }


class ScaledLinear(nn.Module):
    """
    A wrapper around nn.Linear that applies a fixed output scaling factor.
    
    This is used to implement the layer multiplier (n^{-a_l}) from ABC
    parametrization without modifying the underlying Linear layer.
    
    The forward pass computes: scale * linear(x)
    
    Args:
        linear: The nn.Linear module to wrap.
        scale: Fixed scaling factor applied to output.
    
    Attributes:
        linear: The wrapped nn.Linear module.
        scale: The output scaling factor.
    """
    
    def __init__(self, linear: nn.Linear, scale: float):
        super().__init__()
        self.linear = linear
        self.scale = scale
    
    def forward(self, x):
        return self.scale * self.linear(x)
    
    @property
    def weight(self):
        """Access wrapped layer's weight for compatibility."""
        return self.linear.weight
    
    @property
    def bias(self):
        """Access wrapped layer's bias for compatibility."""
        return self.linear.bias
    
    @property
    def in_features(self) -> int:
        """Access wrapped layer's in_features for compatibility."""
        return self.linear.in_features
    
    @property
    def out_features(self) -> int:
        """Access wrapped layer's out_features for compatibility."""
        return self.linear.out_features
    
    def __repr__(self):
        return (
            f"ScaledLinear(in_features={self.linear.in_features}, "
            f"out_features={self.linear.out_features}, "
            f"bias={self.linear.bias is not None}, scale={self.scale:.4g})"
        )


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

    Re-initializes weights of all nn.Linear layers using:
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

    # Get all nn.Linear layers for initialization
    layer_infos = get_managed_layers(model)
    linear_infos = [l for l in layer_infos if isinstance(l.module, nn.Linear)]
    linear_layers = [(l.name, l.module, l.semantic_role) for l in linear_infos]
    n_layers = len(linear_layers)

    # Resolve parametrization
    if parametrization is not None:
        if al is not None or bl is not None:
            raise ValueError(
                "Cannot specify both 'parametrization' and explicit 'al'/'bl' values. "
                "Use either named parametrization or custom values, not both."
            )
        if n_layers == 0:
            raise ValueError("No Linear layers found in model")

        # Get semantic roles for Linear layers only
        linear_roles = [role for (name, module, role) in linear_layers]

        abc = get_abc_parametrization(
            semantic_roles=linear_roles,
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

    for (name, linear, role), a, b in zip(linear_layers, al, bl):
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


def get_managed_layers(
    model: nn.Module,
    embedding_patterns: tuple[str, ...] = ("embed", "embedding", "patch_embed", "token_embed", "pos_embed", "cls_token", "class_token"),
) -> list[LayerInfo]:
    """
    Get all managed layers from a model with their semantic roles.

    Collects nn.Linear, nn.Embedding, nn.Parameter (matching embedding patterns),
    and nn.LayerNorm scale parameters, and assigns semantic roles based on the paper:
    - EMBEDDING: embeddings, positional embeddings, LayerNorm scale parameters
    - HIDDEN: all other nn.Linear layers (MLP, attention projections, etc.)
    - READOUT: the last nn.Linear layer in the network

    Special case: If no EMBEDDING layers exist, the first nn.Linear layer is
    assigned EMBEDDING role (it serves as the input projection).

    Args:
        model: PyTorch model to inspect.
        embedding_patterns: Tuple of substrings that identify embedding layers.
            If a layer name contains one of these patterns (case-insensitive),
            it's classified as EMBEDDING. Also used to identify nn.Parameter embeddings.

    Returns:
        List of LayerInfo objects for each managed layer, in definition order.

    Example:
        >>> layers = get_managed_layers(model)
        >>> for l in layers:
        ...     print(f"{l.name}: {l.semantic_role.value}")
    """
    # First pass: collect all layers with temporary role markers
    _TempInfo = tuple[str, nn.Module | nn.Parameter, bool, int, int]  # name, module, is_embedding, fan_in, fan_out
    temp_layers: list[_TempInfo] = []

    def _is_embedding_layer(name: str) -> bool:
        """Check if the layer name matches embedding patterns."""
        name_lower = name.lower()
        return any(pat.lower() in name_lower for pat in embedding_patterns)

    def traverse(module: nn.Module, prefix: str) -> None:
        # Check for nn.Parameter attributes that match embedding patterns
        for name, param in module.named_parameters(recurse=False):
            full_name = f"{prefix}.{name}" if prefix else name
            if _is_embedding_layer(full_name) and param.requires_grad:
                # nn.Parameter embedding (e.g., pos_embed, cls_token)
                if param.ndim >= 2:
                    fan_in = param.shape[-1]
                    fan_out = param.numel() // fan_in
                else:
                    fan_in = param.shape[0]
                    fan_out = 1
                temp_layers.append((full_name, param, True, fan_in, fan_out))

        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):
                fan_in = child.in_features
                fan_out = child.out_features
                is_embed = _is_embedding_layer(full_name)
                temp_layers.append((full_name, child, is_embed, fan_in, fan_out))
            elif isinstance(child, nn.Embedding):
                temp_layers.append((full_name, child, True, child.embedding_dim, child.num_embeddings))
            elif isinstance(child, nn.LayerNorm):
                # LayerNorm scale parameters are EMBEDDING per the paper
                fan_in = child.normalized_shape[-1] if isinstance(child.normalized_shape, (list, tuple)) else child.normalized_shape
                temp_layers.append((full_name + '.weight', child.weight, True, fan_in, 1))
            else:
                traverse(child, full_name)

    traverse(model, "")

    # Second pass: determine semantic roles
    # Find first/last linear indices and check for embedding layers
    first_linear_idx = None
    last_linear_idx = None
    has_embedding_layer = False

    for i, (name, module, is_embed, fan_in, fan_out) in enumerate(temp_layers):
        if isinstance(module, nn.Linear) and not is_embed:
            if first_linear_idx is None:
                first_linear_idx = i
            last_linear_idx = i
        if is_embed:
            has_embedding_layer = True

    # Build final LayerInfo list with semantic roles
    layers: list[LayerInfo] = []
    for i, (name, module, is_embed, fan_in, fan_out) in enumerate(temp_layers):
        if is_embed:
            role = SemanticRole.EMBEDDING
        elif isinstance(module, nn.Linear):
            if i == last_linear_idx:
                role = SemanticRole.READOUT
            elif i == first_linear_idx and not has_embedding_layer:
                role = SemanticRole.EMBEDDING
            else:
                role = SemanticRole.HIDDEN
        else:
            role = SemanticRole.HIDDEN

        layers.append(LayerInfo(
            name=name,
            module=module,
            semantic_role=role,
            fan_in=fan_in,
            fan_out=fan_out,
        ))

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

    Creates one parameter group per managed layer with initial learning rate
    set according to ABC parametrization using that layer's fan-in n_l:
        lr_l = lr_prefactor * n_l^{-c_l}

    All other parameters are collected into a single "other" group with a constant
    learning rate that is not adjusted by the maxP scheduler.

    There are two ways to specify the initial c_l values:

    1. **Named parametrization**: Pass `parametrization="mup"` (or "sp", "ntk", "mfp")
       and the function will automatically generate appropriate cl values based
       on each layer's semantic role (EMBEDDING, HIDDEN, READOUT).

    2. **Custom values**: Pass explicit `cl` list for fine-grained control.

    Args:
        model: PyTorch model.
        lr_prefactor: Base learning rate multiplier.
        cl: List of initial c_l learning-rate exponents, one per managed layer.
            Required if `parametrization` is not provided.
        parametrization: Named parametrization type. One of:
            - "mup": Maximal Update Parametrization (μP)
            - "sp": Standard Parametrization (PyTorch default)
            - "ntk": Neural Tangent Kernel parametrization
            - "mfp": Mean Field Parametrization
            If provided, cl values are automatically generated based on
            each layer's semantic role.
        optimizer: Optimizer family for named parametrization. One of:
            - "adam": Adam/AdamW/Adamax/NAdam/RAdam family
            - "sgd": SGD family
            Only used when `parametrization` is provided.
        alignment: Alignment assumption for named parametrization. One of "full" or "no".
            Only used when `parametrization` is provided.
        other_lr: Learning rate for non-managed parameters. If None, defaults
            to lr_prefactor. This LR remains constant throughout training.
        **kwargs: Additional arguments passed to each param group
            (e.g., weight_decay, betas).

    Returns:
        List of parameter group dicts suitable for torch.optim constructors.
        Managed layer groups have "maxp_managed": True and "semantic_role": SemanticRole;
        the "other" group (if non-empty) has "maxp_managed": False.

    Raises:
        ValueError: If neither cl nor parametrization is provided, or both are.
        ValueError: If length of cl doesn't match number of managed layers.

    Example using named parametrization:
        >>> model = ViT(...)  # Vision Transformer
        >>> param_groups = create_param_groups(
        ...     model, lr_prefactor=0.1, parametrization="mup"
        ... )
        >>> optimizer = torch.optim.AdamW(param_groups)

    Example using custom cl values:
        >>> model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
        >>> cl = [0.0, 0.5]  # c_l for each Linear layer
        >>> param_groups = create_param_groups(model, lr_prefactor=0.1, cl=cl)
        >>> optimizer = torch.optim.AdamW(param_groups)
    """
    layer_infos = get_managed_layers(model)
    n_total = len(layer_infos)

    if n_total == 0:
        raise ValueError("No managed layers found in model")

    # Get semantic roles from layer infos
    semantic_roles = get_semantic_roles(layer_infos)

    # Resolve cl values
    if parametrization is not None:
        if cl is not None:
            raise ValueError(
                "Cannot specify both 'parametrization' and explicit 'cl' values. "
                "Use either named parametrization or custom values, not both."
            )

        # Get cl for all layers using semantic roles
        abc = get_abc_parametrization(
            semantic_roles=semantic_roles,
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

    if n_total != len(cl):
        raise ValueError(
            f"Length of cl ({len(cl)}) must match number of managed layers ({n_total})."
        )

    if other_lr is None:
        other_lr = lr_prefactor

    # Collect managed parameter ids for exclusion from "other" group
    managed_param_ids: set[int] = set()
    param_groups: list[dict[str, Any]] = []

    for layer_info, c in zip(layer_infos, cl):
        fan_in = layer_info.fan_in
        lr = lr_prefactor * (fan_in ** (-c))

        # Get the weight parameter
        if isinstance(layer_info.module, nn.Linear):
            weight = layer_info.module.weight
        elif isinstance(layer_info.module, nn.Embedding):
            weight = layer_info.module.weight
        elif isinstance(layer_info.module, nn.Parameter):
            # nn.Parameter (e.g., pos_embed, cls_token)
            weight = layer_info.module
        else:
            continue  # Skip unknown module types

        managed_param_ids.add(id(weight))
        group = {
            "params": [weight],
            "lr": lr,
            "layer_name": layer_info.name,
            "fan_in": fan_in,
            "c": float(c),
            "semantic_role": layer_info.semantic_role,
            "maxp_managed": True,
            **kwargs,
        }
        param_groups.append(group)

    # Collect all other parameters
    other_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in managed_param_ids
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
