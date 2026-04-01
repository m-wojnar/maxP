"""
Chain (flat-list) LP solver for computing optimal learning rate exponents.

This is a reference implementation kept for comparison testing against the
graph-based solver. The graph solver (maxp.solver.find_c) is a strict
generalization that handles linear chains as a special case.

Originally lived in maxp/solver.py; moved here when the main package was
simplified to use only the graph-based solver.
"""

import numpy as np
import pulp as plp

from maxp.solver import _min2_lp, _min_lp


def find_c_adam(
    al: list[float],
    bl: list[float],
    align_z0_dW: list[float],
    align_dZ_w0: list[float],
    align_dZ_dW: list[float],
    solver: plp.LpSolver | None = None,
    feature_learning: bool = False,
    M: float = 10.0,
) -> tuple[list[float], list[float]]:
    """
    Find optimal learning rate exponents c_l for Adam-family optimizers.

    Solves an LP to minimize sum(c_l) (maximize learning rates) subject to
    stability constraints derived from alignment measurements.

    Args:
        al: List of a_l exponents (layer multipliers).
        bl: List of b_l exponents (initialization variance).
        align_z0_dW: List of alignment values for the z0 @ dW^T term.
        align_dZ_w0: List of alignment values for the dZ @ W0^T term.
        align_dZ_dW: List of alignment values for the dZ @ dW^T term.
        solver: PuLP solver instance. If None, uses CBC solver.
            Examples: pulp.PULP_CBC_CMD(), pulp.CPLEX_CMD(), pulp.GLPK_CMD().
        feature_learning: If True, enforce feature learning constraint (r_{L-1} = 0).
        M: Big-M constant for min/max encoding. Default: 10.0.

    Returns:
        Tuple of (cl, rl) where:
            - cl: List of optimal c_l values (learning rate exponents)
            - rl: List of r_l values (stability residuals)

    Raises:
        ValueError if (a,b) violate stability-at-initialization.
        ValueError if LP is infeasible.
    """

    assert len(al) == len(bl) == len(align_z0_dW) == len(align_dZ_w0) == len(align_dZ_dW)
    n = len(al)

    # Validate stability-at-initialization conditions for provided (a,b).
    if not np.isclose(al[0] + bl[0], 0.0):
        raise ValueError("Invalid (a,b): al[0] + bl[0] must equal 0.0 for stability at initialization.")
    for i in range(1, n - 1):
        if not np.isclose(al[i] + bl[i], 0.5):
            raise ValueError("Invalid (a,b): al[i] + bl[i] must equal 0.5 for stability at initialization.")
    if al[n - 1] + bl[n - 1] < 0.5 - 1e-12:
        raise ValueError("Invalid (a,b): al[n-1] + bl[n-1] must be at least 0.5 for stability at initialization.")

    if solver is None:
        solver = plp.PULP_CBC_CMD(msg=False)

    # Variable ID counter for unique naming
    var_id = [0]

    lp = plp.LpProblem("maxp_adam", plp.LpMinimize)
    c = plp.LpVariable.dicts("c", range(n))
    r = plp.LpVariable.dicts("r", range(n))
    lp.c = c
    lp.r = r

    # Stable activations during training (first layer)
    lp += r[0] == al[0] + c[0]
    lp += r[0] >= 0

    # Hidden layers
    for i in range(1, n - 1):
        x1 = plp.LpVariable(f"min_x1_{i}")
        x2 = plp.LpVariable(f"min_x2_{i}")
        x3 = plp.LpVariable(f"min_x3_{i}")

        lp += x1 == al[i] + c[i] - align_z0_dW[i]
        lp += x2 == al[i] + c[i] + r[i - 1] - align_dZ_dW[i]
        lp += x3 == 0.5 + r[i - 1] - align_dZ_w0[i]
        lp += r[i] == _min_lp(lp, x1, x2, x3, M=M, var_id=var_id)
        lp += r[i] >= 0

    # Output layer (stable logits)
    x1 = plp.LpVariable(f"min_x1_{n - 1}")
    x2 = plp.LpVariable(f"min_x2_{n - 1}")
    x3 = plp.LpVariable(f"min_x3_{n - 1}")

    lp += x1 == al[n - 1] + bl[n - 1] + r[n - 2] - align_dZ_w0[n - 1]
    lp += x2 == al[n - 1] + c[n - 1] - align_z0_dW[n - 1]
    lp += x3 == al[n - 1] + c[n - 1] + r[n - 2] - align_dZ_dW[n - 1]
    lp += r[n - 1] == _min_lp(lp, x1, x2, x3, M=M, var_id=var_id)
    lp += r[n - 1] >= 0

    # Feature learning constraint
    if feature_learning:
        lp += r[n - 2] == 0

    # Objective: minimize sum of c_l (maximize learning rates)
    lp += plp.lpSum(c)
    lp.solve(solver)

    if lp.status != plp.LpStatusOptimal:
        raise ValueError("LP solver did not find an optimal solution; problem may be infeasible.")

    cl = [lp.c[i].varValue for i in range(n)]
    rl = [lp.r[i].varValue for i in range(n)]
    return cl, rl


def find_c_sgd(
    al: list[float],
    bl: list[float],
    align_z0_dW: list[float],
    align_dZ_w0: list[float],
    align_dZ_dW: list[float],
    solver: plp.LpSolver | None = None,
    feature_learning: bool = False,
    M: float = 10.0,
) -> tuple[list[float], list[float]]:
    """
    Find optimal learning rate exponents c_l for SGD optimizer.

    Similar to find_c_adam but with different constraint formulation
    accounting for SGD's gradient scaling behavior.

    Args:
        al: List of a_l exponents (layer multipliers).
        bl: List of b_l exponents (initialization variance).
        align_z0_dW: List of alignment values for the z0 @ dW^T term.
        align_dZ_w0: List of alignment values for the dZ @ W0^T term.
        align_dZ_dW: List of alignment values for the dZ @ dW^T term.
        solver: PuLP solver instance. If None, uses CBC solver.
        feature_learning: If True, enforce feature learning constraint.
        M: Big-M constant for min/max encoding.

    Returns:
        Tuple of (cl, rl) where:
            - cl: List of optimal c_l values
            - rl: List of r_l values

    Raises:
        ValueError if (a,b) violate stability-at-initialization.
        ValueError if LP is infeasible.
    """

    assert len(al) == len(bl) == len(align_z0_dW) == len(align_dZ_w0) == len(align_dZ_dW)
    n = len(al)

    # Validate stability-at-initialization conditions for provided (a,b).
    if not np.isclose(al[0] + bl[0], 0.0):
        raise ValueError("Invalid (a,b): al[0] + bl[0] must equal 0.0 for stability at initialization.")
    for i in range(1, n - 1):
        if not np.isclose(al[i] + bl[i], 0.5):
            raise ValueError("Invalid (a,b): al[i] + bl[i] must equal 0.5 for stability at initialization.")
    if (al[n - 1] + bl[n - 1]) < 0.5 - 1e-12:
        raise ValueError("Invalid (a,b): al[n-1] + bl[n-1] must be at least 0.5 for stability at initialization.")

    if solver is None:
        solver = plp.PULP_CBC_CMD(msg=False)

    var_id = [0]

    lp = plp.LpProblem("maxp_sgd", plp.LpMinimize)
    c = plp.LpVariable.dicts("c", range(n))
    r = plp.LpVariable.dicts("r", range(n))
    g = plp.LpVariable.dicts("g", range(n - 1))
    lp.c = c
    lp.r = r

    # Gradient scaling terms
    for i in range(n - 1):
        lp += g[i] == _min_lp(lp, al[n - 1] + bl[n - 1], 2 * al[n - 1] + c[n - 1], M=M, var_id=var_id) + al[i]

    # First layer
    lp += r[0] == g[0] + al[0] + c[0]
    lp += r[0] >= 0

    # Hidden layers
    for i in range(1, n - 1):
        x1 = plp.LpVariable(f"min_x1_{i}")
        x2 = plp.LpVariable(f"min_x2_{i}")
        x3 = plp.LpVariable(f"min_x3_{i}")

        lp += x1 == g[i] + al[i] + c[i] - align_z0_dW[i]
        lp += x2 == g[i] + al[i] + c[i] + r[i - 1] - align_dZ_dW[i]
        lp += x3 == 0.5 + r[i - 1] - align_dZ_w0[i]
        lp += r[i] == _min_lp(lp, x1, x2, x3, M=M, var_id=var_id)
        lp += r[i] >= 0

    # Output layer
    x1 = plp.LpVariable(f"min_x1_{n - 1}")
    x2 = plp.LpVariable(f"min_x2_{n - 1}")
    x3 = plp.LpVariable(f"min_x3_{n - 1}")

    lp += x1 == al[n - 1] + bl[n - 1] + r[n - 2] - align_dZ_w0[n - 1]
    lp += x2 == 2 * al[n - 1] + c[n - 1] - align_z0_dW[n - 1]
    lp += x3 == 2 * al[n - 1] + c[n - 1] + r[n - 2] - align_dZ_dW[n - 1]
    lp += r[n - 1] == _min_lp(lp, x1, x2, x3, M=M, var_id=var_id)
    lp += r[n - 1] >= 0

    # Feature learning constraint
    if feature_learning:
        lp += r[n - 2] == 0

    # Objective: minimize sum of c_l (maximize learning rates)
    lp += plp.lpSum(c)
    lp.solve(solver)

    if lp.status != plp.LpStatusOptimal:
        raise ValueError("LP solver did not find an optimal solution; problem may be infeasible.")

    cl = [lp.c[i].varValue for i in range(n)]
    rl = [lp.r[i].varValue for i in range(n)]
    return cl, rl


def find_c(
    al: list[float],
    bl: list[float],
    align_z0_dW: list[float],
    align_dZ_w0: list[float],
    align_dZ_dW: list[float],
    optimizer_type: str = "adam",
    solver: plp.LpSolver | None = None,
    feature_learning: bool = False,
    M: float = 10.0,
) -> tuple[list[float], list[float]]:
    """
    Find optimal learning rate exponents c_l.

    Unified interface that dispatches to find_c_adam or find_c_sgd
    based on optimizer_type.

    Args:
        al: List of a_l exponents.
        bl: List of b_l exponents.
        align_z0_dW: List of alignment values for the z0 @ dW^T term.
        align_dZ_w0: List of alignment values for the dZ @ W0^T term.
        align_dZ_dW: List of alignment values for the dZ @ dW^T term.
        optimizer_type: Either "adam" or "sgd".
        solver: PuLP solver instance.
        feature_learning: If True, enforce feature learning constraint.
        M: Big-M constant.

    Returns:
        Tuple of (cl, rl).

    Raises:
        ValueError if optimizer_type is not "adam" or "sgd".
        ValueError if (a,b) violate stability-at-initialization.
        ValueError if LP is infeasible.
    """

    if optimizer_type.lower() == "adam":
        return find_c_adam(al, bl, align_z0_dW, align_dZ_w0, align_dZ_dW, solver, feature_learning, M)
    elif optimizer_type.lower() == "sgd":
        return find_c_sgd(al, bl, align_z0_dW, align_dZ_w0, align_dZ_dW, solver, feature_learning, M)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}. Must be 'adam' or 'sgd'.")
