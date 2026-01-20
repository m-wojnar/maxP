import numpy as np

from maxp.solver import find_c_adam, find_c_sgd
from maxp.utils import SemanticRole


def test_solver_adam_returns_valid_lengths_and_nonneg_r():
    # 3 layers: EMBEDDING, HIDDEN, READOUT (typical MLP without embedding layer type)
    semantic_roles = [SemanticRole.EMBEDDING, SemanticRole.HIDDEN, SemanticRole.READOUT]
    al = [-0.5, 0.0, 0.5]
    bl = [0.5, 0.5, 0.5]
    alpha = [0.0, 0.0, 0.0]
    omega = [0.0, 0.0, 0.0]
    u = [0.0, 0.0, 0.0]

    cl, rl = find_c_adam(al, bl, alpha, omega, u, semantic_roles=semantic_roles)

    assert len(cl) == 3
    assert len(rl) == 3
    assert all(np.isfinite(x) for x in cl)
    assert all(r >= -1e-8 for r in rl)


def test_solver_sgd_returns_valid_lengths_and_nonneg_r():
    # 3 layers: EMBEDDING, HIDDEN, READOUT (typical MLP without embedding layer type)
    semantic_roles = [SemanticRole.EMBEDDING, SemanticRole.HIDDEN, SemanticRole.READOUT]
    al = [-0.5, 0.0, 0.5]
    bl = [0.5, 0.5, 0.5]
    alpha = [0.0, 0.0, 0.0]
    omega = [0.0, 0.0, 0.0]
    u = [0.0, 0.0, 0.0]

    cl, rl = find_c_sgd(al, bl, alpha, omega, u, semantic_roles=semantic_roles)

    assert len(cl) == 3
    assert len(rl) == 3
    assert all(np.isfinite(x) for x in cl)
    assert all(r >= -1e-8 for r in rl)


def test_solver_with_embedding_layer():
    """Test solver with an embedding layer (simple constraints)."""
    # 3 layers: EMBEDDING, HIDDEN, READOUT
    semantic_roles = [SemanticRole.EMBEDDING, SemanticRole.HIDDEN, SemanticRole.READOUT]
    al = [-0.5, 0.0, 0.5]
    bl = [0.5, 0.5, 0.5]
    alpha = [0.0, 0.0, 0.0]
    omega = [0.0, 0.0, 0.0]
    u = [0.0, 0.0, 0.0]

    cl, rl = find_c_adam(al, bl, alpha, omega, u, semantic_roles=semantic_roles)

    assert len(cl) == 3
    assert len(rl) == 3
    assert all(np.isfinite(x) for x in cl)
