import torch
import torch.nn as nn

from maxp.scheduler import MaxPScheduler
from maxp.utils import create_param_groups


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 8, bias=False)
        self.l2 = nn.Linear(8, 8, bias=False)
        self.l3 = nn.Linear(8, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.l3(x)


def test_scheduler_warmup_keeps_lrs_constant():
    torch.manual_seed(0)
    model = SmallMLP()
    al = [0.0, 0.5, 0.5]
    bl = [0.0, 0.0, 0.0]
    cl = [0.0, 0.5, 0.5]

    groups = create_param_groups(model, lr_prefactor=0.1, cl=cl)
    opt = torch.optim.AdamW(groups)
    sched = MaxPScheduler(
        opt, model, al=al, bl=bl, lr_prefactor=0.1, warmup_steps=3, solve_interval=1
    )

    X0 = torch.randn(16, 4)
    sched.capture_initial(X0)

    initial_lrs = sched.get_last_lr()

    # warmup steps
    for _ in range(3):
        opt.zero_grad()
        y = model(X0).sum()
        y.backward()
        opt.step()
        sched.step(X0)
        assert sched.get_last_lr() == initial_lrs


def test_scheduler_solve_interval_caches_lrs():
    torch.manual_seed(0)
    model = SmallMLP()
    al = [0.0, 0.5, 0.5]
    bl = [0.0, 0.0, 0.0]
    cl = [0.0, 0.5, 0.5]

    groups = create_param_groups(model, lr_prefactor=0.1, cl=cl)
    opt = torch.optim.SGD(groups)
    sched = MaxPScheduler(
        opt,
        model,
        al=al,
        bl=bl,
        lr_prefactor=0.1,
        warmup_steps=0,
        solve_interval=2,
    )

    X = torch.randn(16, 4)
    sched.capture_initial(X)

    # Step 1: no solve (interval=2) -> no cached cl yet, LR should stay as initialized
    opt.zero_grad(); model(X).sum().backward(); opt.step(); sched.step(X)
    lrs1 = sched.get_last_lr()

    # Step 2: solve -> cache and possibly update
    opt.zero_grad(); model(X).sum().backward(); opt.step(); sched.step(X)
    lrs2 = sched.get_last_lr()

    # Step 3: no solve -> reuse cached lrs2
    opt.zero_grad(); model(X).sum().backward(); opt.step(); sched.step(X)
    lrs3 = sched.get_last_lr()

    assert lrs3 == lrs2
    # Allow lrs2 == lrs1 in edge cases (solver might return same c)
    assert len(lrs1) == len(lrs2) == len(lrs3)


def test_scheduler_resample_w0_runs():
    torch.manual_seed(0)
    model = SmallMLP()
    al = [0.0, 0.5, 0.5]
    # Must satisfy stability-at-init constraints used by the LP solver:
    # a0+b0=0, a1+b1=0.5, a2+b2>=0.5
    bl = [0.0, 0.0, 0.0]
    cl = [0.0, 0.5, 0.5]

    groups = create_param_groups(model, lr_prefactor=0.1, cl=cl)
    opt = torch.optim.AdamW(groups)

    sched = MaxPScheduler(
        opt,
        model,
        al=al,
        bl=bl,
        lr_prefactor=0.1,
        warmup_steps=0,
        solve_interval=1,
        resample_w0=True,
    )

    X = torch.randn(16, 4)
    sched.capture_initial(X)

    opt.zero_grad(); model(X).sum().backward(); opt.step(); sched.step(X)
    assert len(sched.get_last_lr()) == 3
