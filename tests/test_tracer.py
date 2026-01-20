import torch
import torch.nn as nn

from maxp.tracer import Tracer


class ToyModel(nn.Module):
    """Simple model with Linear and LayerNorm layers."""
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


def test_tracer_collects_linear_layers():
    """Test that tracer collects Linear layers and LayerNorm scale parameters."""
    model = ToyModel()
    tracer = Tracer(model, sample_size=3, resample_w0=False)

    # Tracer should find all managed layers (nn.Linear)
    # LayerNorm now exposes its scale parameter as a managed entry (e.g. 'ln.weight')
    assert len(tracer.layer_names) == 3
    assert any("lin1" in n for n in tracer.layer_names)
    assert any("ln.weight" in n for n in tracer.layer_names)
    assert any("lin2" in n for n in tracer.layer_names)


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
        # nn.Parameter entries (e.g., 'ln.weight') have input=None
        if snap.input is not None:
            assert snap.input.shape[0] == 2
            assert snap.output is not None
            assert snap.output.shape[0] == 2
        else:
            # Parameter-only entries: output is the parameter value
            assert snap.output is not None

        # Weight snapshot present for all managed entries (may be 1D for norm scales)
        assert snap.weight is not None
        assert snap.weight.ndim >= 1


def test_resample_w0_does_not_store_initial_weights():
    model = ToyModel()
    # Provide bl for each managed layer (lin1, ln.weight, lin2)
    tracer = Tracer(model, sample_size=2, bl=[0.1, 0.2, 0.3], resample_w0=True)

    X = torch.randn(3, 4)
    init = tracer.capture_initial(X)
    cur = tracer.capture(X, step=1)

    # When resampling, init weights are not stored at all.
    for name, snap in init.layers.items():
        assert snap.weight is None

    # Current captures still include weights.
    for name, snap in cur.layers.items():
        assert snap.weight is not None


def test_tracer_has_semantic_roles():
    """Test that tracer computes semantic roles for layers."""
    model = ToyModel()
    tracer = Tracer(model, sample_size=2, resample_w0=False)
    
    # Should have semantic roles for each layer
    assert len(tracer.semantic_roles) == len(tracer.layer_names)
    
    # Check window includes semantic_roles
    X = torch.randn(3, 4)
    init = tracer.capture_initial(X)
    cur = tracer.capture(X, step=1)
    window = tracer.window(cur)
    
    assert window.semantic_roles is not None
    assert len(window.semantic_roles) == window.n_layers
