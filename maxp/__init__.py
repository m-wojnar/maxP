"""
maxP: Maximal Parametrization LR Scheduler for PyTorch

A learning rate scheduler that dynamically adjusts per-layer learning rates
using alignment measurements between initial and current weights/activations,
solved via linear programming.
"""

from maxp.scheduler import MaxPScheduler
from maxp.tracer import Tracer, LayerSnapshot, StepTrace, TraceWindow
from maxp.alignment import compute_alignment
from maxp.solver import find_c, find_c_adam, find_c_sgd
from maxp.utils import (
    ABCParametrization, ScaledLinear, 
    create_param_groups, get_abc_parametrization, get_linear_layers, initialize_abc_weights
)

__version__ = "0.1.0"

__all__ = [
    "MaxPScheduler",
    "Tracer",
    "LayerSnapshot",
    "StepTrace",
    "TraceWindow",
    "compute_alignment",
    "find_c",
    "find_c_adam",
    "find_c_sgd",
    "ABCParametrization",
    "ScaledLinear",
    "create_param_groups",
    "get_abc_parametrization",
    "get_linear_layers",
    "initialize_abc_weights",
]
