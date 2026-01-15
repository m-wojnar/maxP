import numpy as np

from maxp.solver import find_c_adam, find_c_sgd


def test_solver_adam_returns_valid_lengths_and_nonneg_r():
    al = [0.0, 0.5, 0.5]
    bl = [0.0, 0.0, 0.0]
    alpha = [0.0, 0.0, 0.0]
    omega = [0.0, 0.0, 0.0]
    u = [0.0, 0.0, 0.0]

    cl, rl = find_c_adam(al, bl, alpha, omega, u)

    assert len(cl) == 3
    assert len(rl) == 3
    assert all(np.isfinite(x) for x in cl)
    assert all(r >= -1e-8 for r in rl)


def test_solver_sgd_returns_valid_lengths_and_nonneg_r():
    al = [0.0, 0.5, 0.5]
    bl = [0.0, 0.0, 0.0]
    alpha = [0.0, 0.0, 0.0]
    omega = [0.0, 0.0, 0.0]
    u = [0.0, 0.0, 0.0]

    cl, rl = find_c_sgd(al, bl, alpha, omega, u)

    assert len(cl) == 3
    assert len(rl) == 3
    assert all(np.isfinite(x) for x in cl)
    assert all(r >= -1e-8 for r in rl)
