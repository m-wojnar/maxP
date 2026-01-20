import math

import torch
import torch.nn as nn

from maxp.scheduler import MaxPScheduler
from maxp.utils import create_param_groups


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 8, bias=False)
        self.l2 = nn.Linear(8, 8, bias=False)
        self.l3 = nn.Linear(8, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.l3(x)


class TinyNetWithNorm(nn.Module):
    """Model with LayerNorm and biases to test 'other' param group handling."""
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 8, bias=True)
        self.ln = nn.LayerNorm(8)
        self.l2 = nn.Linear(8, 8, bias=True)
        self.l3 = nn.Linear(8, 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.ln(self.l1(x)))
        x = torch.relu(self.l2(x))
        return self.l3(x)


def test_smoke_tiny_training_loop_no_crash_and_lr_updates_after_warmup():
    torch.manual_seed(0)

    model = TinyNet()

    # For TinyNet with 3 LINEAR layers (no LayerNorm):
    # - l1 and l2 are HIDDEN, l3 is READOUT
    al = [-0.5, 0.0, 0.5]
    bl = [0.5, 0.5, 0.5]
    cl = [0.5, 0.5, 0.5]

    groups = create_param_groups(model, lr_prefactor=0.1, cl=cl)
    opt = torch.optim.AdamW(groups)

    sched = MaxPScheduler(
        opt,
        model,
        al=al,
        bl=bl,
        lr_prefactor=0.1,
        solver_warmup_steps=2,
        solve_interval=1,
        resample_w0=False,
    )

    X0 = torch.randn(16, 4)
    sched.capture_initial(X0)

    initial_lrs = list(sched.get_last_lr())
    lr_history = []

    for step in range(6):
        X = torch.randn(16, 4)
        y = torch.randn(16, 2)

        opt.zero_grad()
        pred = model(X)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        opt.step()
        sched.step(X)

        lrs = list(sched.get_last_lr())
        lr_history.append(lrs)

        # LRs should always be finite
        assert all(math.isfinite(v) and v > 0 for v in lrs)

    # Warmup: first 2 scheduler steps should not change LRs
    assert lr_history[0] == initial_lrs
    assert lr_history[1] == initial_lrs

    # In most realistic cases LR changes; allow equality but ensure we computed something.
    changed_after_warmup = any(lr_history[i] != initial_lrs for i in range(2, len(lr_history)))
    assert changed_after_warmup


def test_smoke_model_with_layernorm_and_biases():
    """Test that all parameters are optimized, but only Linear weight LRs are adjusted."""
    torch.manual_seed(0)

    model = TinyNetWithNorm()

    # For TinyNetWithNorm: l1, ln.weight, l2, l3 are managed entries.
    # Semantic roles (with LayerNorm as EMBEDDING):
    #   l1 -> HIDDEN (has_embedding_layer=True, so first LINEAR is HIDDEN)
    #   ln.weight -> EMBEDDING
    #   l2 -> HIDDEN
    #   l3 -> READOUT
    # Using muP-adam-full values:
    al = [0.0, -0.5, 0.0, 0.5]
    bl = [0.5, 0.5, 0.5, 0.5]
    cl = [1.0, 0.5, 1.0, 0.5]

    other_lr = 0.01
    groups = create_param_groups(model, lr_prefactor=0.1, cl=cl, other_lr=other_lr)
    opt = torch.optim.AdamW(groups)

    # Verify all parameters are in the optimizer
    total_params_in_optimizer = sum(len(g["params"]) for g in opt.param_groups)
    total_model_params = sum(1 for p in model.parameters() if p.requires_grad)
    assert total_params_in_optimizer == total_model_params

    # Verify we have 3 managed groups + 1 other group
    managed_groups = [g for g in opt.param_groups if g.get("maxp_managed", False)]
    other_groups = [g for g in opt.param_groups if not g.get("maxp_managed", False)]

    # l1, ln.weight, l2, l3 are now managed
    assert len(managed_groups) == 4
    assert len(other_groups) == 1
    assert other_groups[0]["lr"] == other_lr

    sched = MaxPScheduler(
        opt,
        model,
        al=al,
        bl=bl,
        lr_prefactor=0.1,
        solver_warmup_steps=1,
        solve_interval=1,
        resample_w0=False,
    )

    X0 = torch.randn(16, 4)
    sched.capture_initial(X0)

    initial_managed_lrs = list(sched.get_last_lr())

    for step in range(4):
        X = torch.randn(16, 4)
        y = torch.randn(16, 2)

        opt.zero_grad()
        pred = model(X)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        opt.step()
        sched.step(X)

    # Other group LR should remain constant
    assert other_groups[0]["lr"] == other_lr

    # Managed LRs should be finite and positive
    final_lrs = sched.get_last_lr()
    assert all(math.isfinite(v) and v > 0 for v in final_lrs)
