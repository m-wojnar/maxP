import numpy as np
import torch
import torch.nn as nn

from maxp.tracer import Tracer
from maxp.alignment import compute_alignment


class TwoLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 4, bias=False)
        self.l2 = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        return x


def test_alignment_zero_when_unchanged():
    torch.manual_seed(0)
    model = TwoLinear()
    tracer = Tracer(model, sample_size=None, resample_w0=False)

    X = torch.randn(8, 4)
    tracer.capture_initial(X)
    cur = tracer.capture(X, step=1)
    window = tracer.window(cur)

    alpha, omega, u = compute_alignment(window)

    assert len(alpha) == 2 and len(omega) == 2 and len(u) == 2
    assert all(np.isfinite(v) for v in alpha)
    assert all(np.isfinite(v) for v in omega)
    assert all(np.isfinite(v) for v in u)

    # no change => all should be 0 after sanitization
    assert all(abs(v) < 1e-9 for v in alpha)
    assert all(abs(v) < 1e-9 for v in omega)
    assert all(abs(v) < 1e-9 for v in u)


def test_alignment_nonzero_after_weight_change():
    torch.manual_seed(0)
    model = TwoLinear()
    tracer = Tracer(model, sample_size=None, resample_w0=False)

    X = torch.randn(8, 4)
    tracer.capture_initial(X)

    # perturb second layer weights
    with torch.no_grad():
        model.l2.weight.add_(0.01)

    cur = tracer.capture(X, step=2)
    window = tracer.window(cur)

    alpha, omega, u = compute_alignment(window)

    assert any(abs(v) > 0 for v in alpha)


def test_alignment_resample_w0_runs_and_is_finite():
    torch.manual_seed(0)
    model = TwoLinear()
    tracer = Tracer(model, sample_size=None, bl=[0.1, 0.2], resample_w0=True)

    X = torch.randn(8, 4)
    tracer.capture_initial(X)

    cur = tracer.capture(X, step=1)
    window = tracer.window(cur)

    alpha, omega, u = compute_alignment(window)

    assert all(np.isfinite(v) for v in alpha)
    assert all(np.isfinite(v) for v in omega)
    assert all(np.isfinite(v) for v in u)
