"""
LP solver for computing optimal learning rate exponents.

Uses linear programming to find c_l values that maximize learning rates
while maintaining stability constraints based on alignment measurements.
"""

import numpy as np
import pulp as plp

from maxp.utils import SemanticRole


def _min2_lp(lp: plp.LpProblem, a, b, M: float, var_id: list[int]):
    """
    Encode min(a, b) as LP constraints using big-M method.
    
    Reference: https://or.stackexchange.com/a/1174
    """
    
    X = plp.LpVariable(f"min2_X_{var_id[0]}")
    y = plp.LpVariable(f"min2_y_{var_id[0]}", cat=plp.LpBinary)
    var_id[0] += 1
    
    lp += b - a <= M * y
    lp += a - b <= M * (1 - y)
    lp += X <= a
    lp += X <= b
    lp += X >= a - M * (1 - y)
    lp += X >= b - M * y
    
    return X


def _min_lp(lp: plp.LpProblem, *args, M: float, var_id: list[int]):
    """Encode min(a, b, c, ...) as LP constraints."""

    if len(args) == 1:
        return args[0]
    
    X = _min2_lp(lp, args[0], args[1], M, var_id)
    
    for a in args[2:]:
        X = _min2_lp(lp, X, a, M, var_id)
    
    return X


def _validate_stability_at_init(
    al: list[float], bl: list[float], semantic_roles: list[SemanticRole]
) -> None:
    """
    Validate stability-at-initialization conditions based on semantic roles.
    
    - EMBEDDING: al + bl must equal 0.0
    - HIDDEN: al + bl must equal 0.5
    - READOUT: al + bl must be >= 0.5
    """
    for i, role in enumerate(semantic_roles):
        ab_sum = al[i] + bl[i]
        if role == SemanticRole.EMBEDDING:
            if not np.isclose(ab_sum, 0.0):
                raise ValueError(
                    f"Invalid (a,b) for layer {i} (EMBEDDING role): "
                    f"al[{i}] + bl[{i}] = {ab_sum}, must equal 0.0 for stability at initialization."
                )
        elif role == SemanticRole.HIDDEN:
            if not np.isclose(ab_sum, 0.5):
                raise ValueError(
                    f"Invalid (a,b) for layer {i} (HIDDEN role): "
                    f"al[{i}] + bl[{i}] = {ab_sum}, must equal 0.5 for stability at initialization."
                )
        elif role == SemanticRole.READOUT:
            if ab_sum < 0.5 - 1e-12:
                raise ValueError(
                    f"Invalid (a,b) for layer {i} (READOUT role): "
                    f"al[{i}] + bl[{i}] = {ab_sum}, must be >= 0.5 for stability at initialization."
                )


def find_c_adam(
    al: list[float],
    bl: list[float],
    alpha: list[float],
    omega: list[float],
    u: list[float],
    semantic_roles: list[SemanticRole],
    solver: plp.LpSolver | None = None,
    feature_learning: bool = False,
    M: float = 10.0,
) -> tuple[list[float], list[float]]:
    """
    Find optimal learning rate exponents c_l for Adam-family optimizers.
    
    Solves an LP to minimize sum(c_l) (maximize learning rates) subject to
    stability constraints derived from alignment measurements.
    
    Constraints are applied based on semantic roles:
    - EMBEDDING: Simple stability bound (no alignment terms, like first layer)
    - HIDDEN: Three alignment-based constraints (like middle layers)
    - READOUT: Output layer constraints with bl term (like last layer)
    
    Args:
        al: List of a_l exponents (layer multipliers).
        bl: List of b_l exponents (initialization variance).
        alpha: List of alpha alignment values (z_0 @ Δw term).
        omega: List of omega alignment values (Δz @ w_0 term).
        u: List of u alignment values (Δz @ Δw term).
        semantic_roles: List of SemanticRole for each layer.
        solver: PuLP solver instance. If None, uses CBC solver.
            Examples: pulp.PULP_CBC_CMD(), pulp.CPLEX_CMD(), pulp.GLPK_CMD().
        feature_learning: If True, enforce feature learning constraint 
            (r = 0 for last HIDDEN layer before READOUT).
        M: Big-M constant for min/max encoding. Default: 10.0.
    
    Returns:
        Tuple of (cl, rl) where:
            - cl: List of optimal c_l values (learning rate exponents)
            - rl: List of r_l values (stability residuals)
    
    Raises:
        ValueError if (a,b) violate stability-at-initialization for any role.
        ValueError if LP is infeasible.
    """
    assert len(al) == len(bl) == len(alpha) == len(omega) == len(u) == len(semantic_roles)
    n = len(al)

    # Validate stability-at-initialization conditions by semantic role
    _validate_stability_at_init(al, bl, semantic_roles)
    
    if solver is None:
        solver = plp.PULP_CBC_CMD(msg=False)
    
    # Variable ID counter for unique naming
    var_id = [0]
    
    lp = plp.LpProblem("maxp_adam", plp.LpMinimize)
    c = plp.LpVariable.dicts("c", range(n))
    r = plp.LpVariable.dicts("r", range(n))
    lp.c = c
    lp.r = r
    
    # Find the readout layer index (should be exactly one)
    readout_idx = None
    for i, role in enumerate(semantic_roles):
        if role == SemanticRole.READOUT:
            readout_idx = i
            break
    
    if readout_idx is None:
        raise ValueError("No READOUT layer found in semantic_roles")
    
    # Apply constraints based on semantic role
    for i, role in enumerate(semantic_roles):
        if role == SemanticRole.EMBEDDING:
            # EMBEDDING: simple stability bound (no alignment terms)
            # Same as old "first layer" constraint
            lp += r[i] == al[i] + c[i]
            lp += r[i] >= 0
            
        elif role == SemanticRole.HIDDEN:
            # HIDDEN: three competing constraints based on alignment
            # Same as old "hidden layer" constraints
            x1 = plp.LpVariable(f"min_x1_{i}")
            x2 = plp.LpVariable(f"min_x2_{i}")
            x3 = plp.LpVariable(f"min_x3_{i}")
            
            # Find the previous layer's r (or use 0 if this is first)
            if i == 0:
                r_prev = 0
            else:
                r_prev = r[i - 1]
            
            lp += x1 == al[i] + c[i] - alpha[i]
            lp += x2 == al[i] + c[i] + r_prev - u[i]
            lp += x3 == 0.5 + r_prev - omega[i]
            lp += r[i] == _min_lp(lp, x1, x2, x3, M=M, var_id=var_id)
            lp += r[i] >= 0
            
        elif role == SemanticRole.READOUT:
            # READOUT: output layer constraints with bl term
            # Same as old "last layer" constraints
            x1 = plp.LpVariable(f"min_x1_{i}")
            x2 = plp.LpVariable(f"min_x2_{i}")
            x3 = plp.LpVariable(f"min_x3_{i}")
            
            # Find the previous layer's r
            if i == 0:
                r_prev = 0
            else:
                r_prev = r[i - 1]
            
            lp += x1 == al[i] + bl[i] + r_prev - omega[i]
            lp += x2 == al[i] + c[i] - alpha[i]
            lp += x3 == al[i] + c[i] + r_prev - u[i]
            lp += r[i] == _min_lp(lp, x1, x2, x3, M=M, var_id=var_id)
            lp += r[i] >= 0
    
    # Feature learning constraint: force r=0 on last HIDDEN before READOUT
    if feature_learning:
        # Find the last HIDDEN layer
        last_hidden_idx = None
        for i in range(n - 1, -1, -1):
            if semantic_roles[i] == SemanticRole.HIDDEN:
                last_hidden_idx = i
                break
        
        if last_hidden_idx is not None:
            lp += r[last_hidden_idx] == 0
    
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
    alpha: list[float],
    omega: list[float],
    u: list[float],
    semantic_roles: list[SemanticRole],
    solver: plp.LpSolver | None = None,
    feature_learning: bool = False,
    M: float = 10.0,
) -> tuple[list[float], list[float]]:
    """
    Find optimal learning rate exponents c_l for SGD optimizer.
    
    Similar to find_c_adam but with different constraint formulation
    accounting for SGD's gradient scaling behavior.
    
    Constraints are applied based on semantic roles:
    - EMBEDDING: Simple stability bound with gradient scaling
    - HIDDEN: Three alignment-based constraints with gradient scaling
    - READOUT: Output layer constraints
    
    Args:
        al: List of a_l exponents (layer multipliers).
        bl: List of b_l exponents (initialization variance).
        alpha: List of alpha alignment values.
        omega: List of omega alignment values.
        u: List of u alignment values.
        semantic_roles: List of SemanticRole for each layer.
        solver: PuLP solver instance. If None, uses CBC solver.
        feature_learning: If True, enforce feature learning constraint.
        M: Big-M constant for min/max encoding.
    
    Returns:
        Tuple of (cl, rl) where:
            - cl: List of optimal c_l values
            - rl: List of r_l values

    Raises:
        ValueError if (a,b) violate stability-at-initialization for any role.
        ValueError if LP is infeasible.
    """
    assert len(al) == len(bl) == len(alpha) == len(omega) == len(u) == len(semantic_roles)
    n = len(al)

    # Validate stability-at-initialization conditions by semantic role
    _validate_stability_at_init(al, bl, semantic_roles)
    
    if solver is None:
        solver = plp.PULP_CBC_CMD(msg=False)
    
    var_id = [0]
    
    lp = plp.LpProblem("maxp_sgd", plp.LpMinimize)
    c = plp.LpVariable.dicts("c", range(n))
    r = plp.LpVariable.dicts("r", range(n))
    lp.c = c
    lp.r = r
    
    # Find the readout layer index
    readout_idx = None
    for i, role in enumerate(semantic_roles):
        if role == SemanticRole.READOUT:
            readout_idx = i
            break
    
    if readout_idx is None:
        raise ValueError("No READOUT layer found in semantic_roles")
    
    # Gradient scaling terms (depend on readout layer)
    # g[i] for non-readout layers
    g = {}
    for i, role in enumerate(semantic_roles):
        if role != SemanticRole.READOUT:
            g[i] = plp.LpVariable(f"g_{i}")
            lp += g[i] == _min_lp(
                lp, al[readout_idx] + bl[readout_idx], 2 * al[readout_idx] + c[readout_idx], 
                M=M, var_id=var_id
            ) + al[i]
    
    # Apply constraints based on semantic role
    for i, role in enumerate(semantic_roles):
        if role == SemanticRole.EMBEDDING:
            # EMBEDDING: simple stability bound with gradient scaling
            lp += r[i] == g[i] + al[i] + c[i]
            lp += r[i] >= 0
            
        elif role == SemanticRole.HIDDEN:
            # HIDDEN: three competing constraints with gradient scaling
            x1 = plp.LpVariable(f"min_x1_{i}")
            x2 = plp.LpVariable(f"min_x2_{i}")
            x3 = plp.LpVariable(f"min_x3_{i}")
            
            if i == 0:
                r_prev = 0
            else:
                r_prev = r[i - 1]
            
            lp += x1 == g[i] + al[i] + c[i] - alpha[i]
            lp += x2 == g[i] + al[i] + c[i] + r_prev - u[i]
            lp += x3 == 0.5 + r_prev - omega[i]
            lp += r[i] == _min_lp(lp, x1, x2, x3, M=M, var_id=var_id)
            lp += r[i] >= 0
            
        elif role == SemanticRole.READOUT:
            # READOUT: output layer constraints (no gradient scaling term)
            x1 = plp.LpVariable(f"min_x1_{i}")
            x2 = plp.LpVariable(f"min_x2_{i}")
            x3 = plp.LpVariable(f"min_x3_{i}")
            
            if i == 0:
                r_prev = 0
            else:
                r_prev = r[i - 1]
            
            lp += x1 == al[i] + bl[i] + r_prev - omega[i]
            lp += x2 == 2 * al[i] + c[i] - alpha[i]
            lp += x3 == 2 * al[i] + c[i] + r_prev - u[i]
            lp += r[i] == _min_lp(lp, x1, x2, x3, M=M, var_id=var_id)
            lp += r[i] >= 0
    
    # Feature learning constraint: force r=0 on last HIDDEN before READOUT
    if feature_learning:
        last_hidden_idx = None
        for i in range(n - 1, -1, -1):
            if semantic_roles[i] == SemanticRole.HIDDEN:
                last_hidden_idx = i
                break
        
        if last_hidden_idx is not None:
            lp += r[last_hidden_idx] == 0
    
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
    alpha: list[float],
    omega: list[float],
    u: list[float],
    semantic_roles: list[SemanticRole],
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
        alpha: List of alpha alignment values.
        omega: List of omega alignment values.
        u: List of u alignment values.
        semantic_roles: List of SemanticRole for each layer.
        optimizer_type: Either "adam" or "sgd".
        solver: PuLP solver instance.
        feature_learning: If True, enforce feature learning constraint.
        M: Big-M constant.
    
    Returns:
        Tuple of (cl, rl).
    
    Raises:
        ValueError if optimizer_type is not "adam" or "sgd".
        ValueError if (a,b) violate stability-at-initialization for any role.
        ValueError if LP is infeasible.
    """

    if optimizer_type.lower() == "adam":
        return find_c_adam(al, bl, alpha, omega, u, semantic_roles, solver, feature_learning, M)
    elif optimizer_type.lower() == "sgd":
        return find_c_sgd(al, bl, alpha, omega, u, semantic_roles, solver, feature_learning, M)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}. Must be 'adam' or 'sgd'.")
