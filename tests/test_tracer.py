import torch
import torch.nn as nn

from maxp.tracer import Tracer


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(4, 8, bias=False)
        self.ln = nn.LayerNorm(8)
        self.lin2 = nn.Linear(8, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.ln(x)
        x = torch.relu(x)
        x = self.lin2(x)
        return x


def test_tracer_collects_only_linear_layers():
    model = ToyModel()
    tracer = Tracer(model, sample_size=3, resample_w0=False)

    assert len(tracer.modules) == 2
    assert all(isinstance(m, nn.Linear) for m in tracer.modules)


def test_capture_initial_and_capture_shapes():
    torch.manual_seed(0)
    model = ToyModel()
    tracer = Tracer(model, sample_size=2, resample_w0=False)

    X = torch.randn(5, 4)
    init = tracer.capture_initial(X)
    cur = tracer.capture(X, step=1)

    assert init.step == 0
    assert cur.step == 1
    assert set(init.layers.keys()) == set(cur.layers.keys())

    # Sample size should be applied
    for name, snap in cur.layers.items():
        assert snap.input is not None
        assert snap.output is not None
        assert snap.weight is not None
        assert snap.input.shape[0] == 2
        assert snap.output.shape[0] == 2
        assert snap.weight.ndim == 2


def test_resample_w0_does_not_store_initial_weights():
    model = ToyModel()
    tracer = Tracer(model, sample_size=2, bl=[0.1, 0.2], resample_w0=True)

    X = torch.randn(3, 4)
    init = tracer.capture_initial(X)
    cur = tracer.capture(X, step=1)

    # When resampling, init weights are not stored at all.
    for name, snap in init.layers.items():
        assert snap.weight is None

    # Current captures still include weights.
    for name, snap in cur.layers.items():
        assert snap.weight is not None
