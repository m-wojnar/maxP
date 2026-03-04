"""
LP solver for computing optimal learning rate exponents.

Uses linear programming to find c_l values that maximize learning rates
while maintaining stability constraints based on alignment measurements.
"""

import numpy as np
import pulp as plp


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


# ---------------------------------------------------------------------------
# Graph-based solver: one c variable per weight-bearing PM
# ---------------------------------------------------------------------------

def find_c_adam(
    graph: "OpGraph",
    solver: plp.LpSolver | None = None,
    feature_learning: bool = False,
    M: float = 10.0,
    c_fixed: dict[str, float] | None = None,
) -> dict[str, tuple[float | None, float]]:
    """Find optimal per-op c values for Adam on an OpGraph.

    Args:
        graph: OpGraph describing PM data-flow (from trace_pm_dag() or
            constructed manually).
        solver: PuLP solver instance.
        feature_learning: If True, enforce r=0 for predecessor(s) of sink nodes.
        M: Big-M constant.

    Returns:
        Dict mapping node name -> (c or None for activation-only, r).
    """
    from maxp.dag import MergeType

    if solver is None:
        solver = plp.PULP_CBC_CMD(msg=False)

    graph.validate()
    topo = graph.topological_order()
    var_id = [0]

    lp = plp.LpProblem("maxp_adam", plp.LpMinimize)

    # Create variables for each node
    c_vars: dict[str, plp.LpVariable | float | None] = {}
    r_vars: dict[str, plp.LpVariable] = {}

    for node in topo:
        r_vars[node.name] = plp.LpVariable(f"r_{node.name}")
        if node.has_weight:
            if c_fixed and node.name in c_fixed:
                c_vars[node.name] = c_fixed[node.name]
            else:
                c_vars[node.name] = plp.LpVariable(f"c_{node.name}")
        else:
            c_vars[node.name] = None

    # Validate (a,b) and build constraints in topological order
    sinks = {n.name for n in graph.sinks()}

    for node in topo:
        is_source = len(node.predecessors) == 0
        is_sink = node.name in sinks
        ab_sum = node.a + node.b

        # Validate stability-at-initialization
        if is_source and node.has_weight:
            if not np.isclose(ab_sum, 0.0):
                raise ValueError(
                    f"Source '{node.name}': a+b={ab_sum}, must be 0 for stability at init.")
        elif not is_source and not is_sink and node.has_weight:
            if not np.isclose(ab_sum, 0.5):
                raise ValueError(
                    f"Hidden '{node.name}': a+b={ab_sum}, must be 0.5 for stability at init.")
        elif is_sink and node.has_weight:
            if ab_sum < 0.5 - 1e-12:
                raise ValueError(
                    f"Sink '{node.name}': a+b={ab_sum}, must be >= 0.5 for stability at init.")

        # Compute r_in from predecessors
        r_in = _compute_r_in(lp, node, r_vars, var_id, M)

        r = r_vars[node.name]
        c = c_vars[node.name]

        if is_source:
            if node.has_weight:
                # r = a + c
                lp += r == node.a + c
            else:
                # Activation-only source: r = a (no learning rate)
                lp += r == node.a
        else:
            # Interior or sink node
            if node.has_weight:
                x1 = plp.LpVariable(f"x1_{node.name}")
                x2 = plp.LpVariable(f"x2_{node.name}")
                x3 = plp.LpVariable(f"x3_{node.name}")

                lp += x1 == node.a + c - node.alpha
                lp += x2 == node.a + c + r_in - node.u
                lp += x3 == (node.a + node.b) + r_in - node.omega
                lp += r == _min_lp(lp, x1, x2, x3, M=M, var_id=var_id)
            else:
                # Activation-only: r = r_in + a
                lp += r == r_in + node.a

        lp += r >= 0

    # Feature learning: r=0 for predecessors of sinks
    if feature_learning:
        for sink_node in graph.sinks():
            for pred_name in sink_node.predecessors:
                lp += r_vars[pred_name] == 0

    # Objective: minimize sum of c for weight-bearing nodes (exclude fixed constants)
    weight_c = [c for c in c_vars.values()
                if c is not None and not isinstance(c, (int, float))]
    if weight_c:
        lp += plp.lpSum(weight_c)

    lp.solve(solver)

    if lp.status != plp.LpStatusOptimal:
        raise ValueError("LP solver did not find an optimal solution; problem may be infeasible.")

    result: dict[str, tuple[float | None, float]] = {}
    for node in topo:
        cv = c_vars[node.name]
        if cv is None:
            c_val = None
        elif isinstance(cv, (int, float)):
            c_val = float(cv)
        else:
            c_val = cv.varValue
        r_val = r_vars[node.name].varValue
        result[node.name] = (c_val, r_val)

    return result


def find_c_sgd(
    graph: "OpGraph",
    solver: plp.LpSolver | None = None,
    feature_learning: bool = False,
    M: float = 10.0,
    c_fixed: dict[str, float] | None = None,
) -> dict[str, tuple[float | None, float]]:
    """Find optimal per-op c values for SGD on an OpGraph.

    Same structure as Adam but with gradient scaling terms from the sink nodes.
    """
    from maxp.dag import MergeType

    if solver is None:
        solver = plp.PULP_CBC_CMD(msg=False)

    graph.validate()
    topo = graph.topological_order()
    var_id = [0]

    lp = plp.LpProblem("maxp_sgd", plp.LpMinimize)

    c_vars: dict[str, plp.LpVariable | float | None] = {}
    r_vars: dict[str, plp.LpVariable] = {}

    for node in topo:
        r_vars[node.name] = plp.LpVariable(f"r_{node.name}")
        if node.has_weight:
            if c_fixed and node.name in c_fixed:
                c_vars[node.name] = c_fixed[node.name]
            else:
                c_vars[node.name] = plp.LpVariable(f"c_{node.name}")
        else:
            c_vars[node.name] = None

    sinks = graph.sinks()
    sink_names = {n.name for n in sinks}

    # Validate (a,b) same as Adam
    for node in topo:
        is_source = len(node.predecessors) == 0
        is_sink = node.name in sink_names
        ab_sum = node.a + node.b

        if is_source and node.has_weight:
            if not np.isclose(ab_sum, 0.0):
                raise ValueError(
                    f"Source '{node.name}': a+b={ab_sum}, must be 0.")
        elif not is_source and not is_sink and node.has_weight:
            if not np.isclose(ab_sum, 0.5):
                raise ValueError(
                    f"Hidden '{node.name}': a+b={ab_sum}, must be 0.5.")
        elif is_sink and node.has_weight:
            if ab_sum < 0.5 - 1e-12:
                raise ValueError(
                    f"Sink '{node.name}': a+b={ab_sum}, must be >= 0.5.")

    # Compute gradient scaling: g = min across sinks of
    # min(a_sink + b_sink, 2*a_sink + c_sink)
    # For each non-sink node, g_i = g_base + a_i
    g_per_sink = []
    for sink in sinks:
        if sink.has_weight:
            c_sink = c_vars[sink.name]
            g_per_sink.append(
                _min_lp(lp, sink.a + sink.b, 2 * sink.a + c_sink, M=M, var_id=var_id)
            )
        else:
            # Activation-only sink — use a+b as the bound
            g_per_sink.append(sink.a + sink.b)

    if len(g_per_sink) == 1:
        g_base = g_per_sink[0]
    else:
        g_base = _min_lp(lp, *g_per_sink, M=M, var_id=var_id)

    # Build constraints
    for node in topo:
        is_source = len(node.predecessors) == 0
        is_sink = node.name in sink_names
        r_in = _compute_r_in(lp, node, r_vars, var_id, M)
        r = r_vars[node.name]
        c = c_vars[node.name]

        if is_source:
            if node.has_weight:
                g_i = g_base + node.a
                lp += r == g_i + node.a + c
            else:
                lp += r == node.a
        elif is_sink:
            if node.has_weight:
                x1 = plp.LpVariable(f"x1_{node.name}")
                x2 = plp.LpVariable(f"x2_{node.name}")
                x3 = plp.LpVariable(f"x3_{node.name}")

                lp += x1 == node.a + node.b + r_in - node.omega
                lp += x2 == 2 * node.a + c - node.alpha
                lp += x3 == 2 * node.a + c + r_in - node.u
                lp += r == _min_lp(lp, x1, x2, x3, M=M, var_id=var_id)
            else:
                lp += r == r_in + node.a
        else:
            # Interior (non-source, non-sink)
            if node.has_weight:
                g_i = g_base + node.a
                x1 = plp.LpVariable(f"x1_{node.name}")
                x2 = plp.LpVariable(f"x2_{node.name}")
                x3 = plp.LpVariable(f"x3_{node.name}")

                lp += x1 == g_i + node.a + c - node.alpha
                lp += x2 == g_i + node.a + c + r_in - node.u
                lp += x3 == 0.5 + r_in - node.omega
                lp += r == _min_lp(lp, x1, x2, x3, M=M, var_id=var_id)
            else:
                lp += r == r_in + node.a

        lp += r >= 0

    if feature_learning:
        for sink_node in sinks:
            for pred_name in sink_node.predecessors:
                lp += r_vars[pred_name] == 0

    weight_c = [c for c in c_vars.values()
                if c is not None and not isinstance(c, (int, float))]
    if weight_c:
        lp += plp.lpSum(weight_c)

    lp.solve(solver)

    if lp.status != plp.LpStatusOptimal:
        raise ValueError("LP solver did not find an optimal solution; problem may be infeasible.")

    result: dict[str, tuple[float | None, float]] = {}
    for node in topo:
        cv = c_vars[node.name]
        if cv is None:
            c_val = None
        elif isinstance(cv, (int, float)):
            c_val = float(cv)
        else:
            c_val = cv.varValue
        r_val = r_vars[node.name].varValue
        result[node.name] = (c_val, r_val)

    return result


def find_c(
    graph: "OpGraph",
    optimizer_type: str = "adam",
    solver: plp.LpSolver | None = None,
    feature_learning: bool = False,
    M: float = 10.0,
    c_fixed: dict[str, float] | None = None,
) -> dict[str, tuple[float | None, float]]:
    """Find optimal per-op c values on an OpGraph.

    Dispatches to find_c_adam or find_c_sgd.

    Args:
        c_fixed: Optional dict mapping node name → fixed c value.
            Nodes in this dict use a float constant instead of an LP variable.

    Returns:
        Dict mapping node name -> (c or None for activation-only, r).
    """
    if optimizer_type.lower() == "adam":
        return find_c_adam(graph, solver, feature_learning, M, c_fixed=c_fixed)
    elif optimizer_type.lower() == "sgd":
        return find_c_sgd(graph, solver, feature_learning, M, c_fixed=c_fixed)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}. Must be 'adam' or 'sgd'.")


def _compute_r_in(lp, node, r_vars, var_id, M):
    """Compute r_in for a node from its predecessors, respecting merge type."""
    from maxp.dag import MergeType

    if not node.predecessors:
        return None

    pred_rs = [r_vars[p] for p in node.predecessors]

    if len(pred_rs) == 1:
        return pred_rs[0]

    if node.merge_type == MergeType.SUM:
        return plp.lpSum(pred_rs)
    else:  # MIN
        return _min_lp(lp, *pred_rs, M=M, var_id=var_id)
