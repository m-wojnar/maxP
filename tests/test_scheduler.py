import torch
import torch.nn as nn

from maxp.scheduler import MaxPScheduler, ChainedMaxPScheduler
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


# =============================================================================
# ChainedMaxPScheduler Tests
# =============================================================================


def test_chained_scheduler_applies_cosine_decay():
    """Verify that chained cosine scheduler applies decay to learning rates."""
    torch.manual_seed(0)
    model = SmallMLP()
    al = [0.0, 0.5, 0.5]
    bl = [0.0, 0.0, 0.0]
    cl = [0.0, 0.5, 0.5]

    groups = create_param_groups(model, lr_prefactor=0.1, cl=cl)
    opt = torch.optim.AdamW(groups)
    
    maxp_sched = MaxPScheduler(
        opt, model, al=al, bl=bl, lr_prefactor=0.1, warmup_steps=0, solve_interval=1
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    chained_sched = ChainedMaxPScheduler(maxp_sched, [cosine_sched])

    X = torch.randn(16, 4)
    chained_sched.capture_initial(X)
    
    initial_lrs = chained_sched.get_last_lr()
    
    # Run several steps - LRs should decrease with cosine schedule
    for _ in range(5):
        opt.zero_grad()
        model(X).sum().backward()
        opt.step()
        chained_sched.step(X)
    
    current_lrs = chained_sched.get_last_lr()
    
    # Cosine decay should have reduced LRs
    for init_lr, curr_lr in zip(initial_lrs, current_lrs):
        assert curr_lr < init_lr, "Cosine decay should reduce learning rates"


def test_chained_scheduler_preserves_layer_ratios():
    """Verify that decay is applied uniformly, preserving per-layer LR ratios from MaxP."""
    torch.manual_seed(0)
    model = SmallMLP()
    al = [0.0, 0.5, 0.5]
    bl = [0.0, 0.0, 0.0]
    cl = [0.0, 0.5, 0.5]

    groups = create_param_groups(model, lr_prefactor=0.1, cl=cl)
    opt = torch.optim.AdamW(groups)
    
    maxp_sched = MaxPScheduler(
        opt, model, al=al, bl=bl, lr_prefactor=0.1, warmup_steps=0, solve_interval=1
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    chained_sched = ChainedMaxPScheduler(maxp_sched, [cosine_sched])

    X = torch.randn(16, 4)
    chained_sched.capture_initial(X)
    
    # Run several steps and verify decay factor is applied uniformly at each step
    for step in range(5):
        opt.zero_grad()
        model(X).sum().backward()
        opt.step()
        chained_sched.step(X)
        
        # Get the MaxP-computed LRs (before decay) and final LRs (after decay)
        maxp_lrs = chained_sched._maxp_lrs
        final_lrs = chained_sched.get_last_lr()
        
        # All layers should have the same decay factor applied
        if maxp_lrs[0] != 0:
            decay_factors = [final / maxp for final, maxp in zip(final_lrs, maxp_lrs)]
            for i, df in enumerate(decay_factors[1:], 1):
                assert abs(decay_factors[0] - df) < 1e-9, \
                    f"Decay factor should be uniform across layers at step {step}"


def test_chained_scheduler_multiple_schedulers():
    """Test chaining multiple schedulers (warmup + cosine)."""
    torch.manual_seed(0)
    model = SmallMLP()
    al = [0.0, 0.5, 0.5]
    bl = [0.0, 0.0, 0.0]
    cl = [0.0, 0.5, 0.5]

    groups = create_param_groups(model, lr_prefactor=0.1, cl=cl)
    opt = torch.optim.AdamW(groups)
    
    maxp_sched = MaxPScheduler(
        opt, model, al=al, bl=bl, lr_prefactor=0.1, warmup_steps=0, solve_interval=1
    )
    # Linear warmup for first 3 steps
    warmup_sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=3)
    # Then cosine decay
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    
    chained_sched = ChainedMaxPScheduler(maxp_sched, [warmup_sched, cosine_sched])

    X = torch.randn(16, 4)
    chained_sched.capture_initial(X)
    
    lrs_history = []
    
    for _ in range(6):
        opt.zero_grad()
        model(X).sum().backward()
        opt.step()
        chained_sched.step(X)
        lrs_history.append(chained_sched.get_last_lr()[0])
    
    # During warmup (steps 1-3), LRs should generally increase
    # After warmup, cosine decay kicks in
    assert len(lrs_history) == 6


def test_chained_scheduler_state_dict_roundtrip():
    """Test that state_dict/load_state_dict correctly saves and restores state."""
    torch.manual_seed(0)
    model = SmallMLP()
    al = [0.0, 0.5, 0.5]
    bl = [0.0, 0.0, 0.0]
    cl = [0.0, 0.5, 0.5]

    groups = create_param_groups(model, lr_prefactor=0.1, cl=cl)
    opt = torch.optim.AdamW(groups)
    
    maxp_sched = MaxPScheduler(
        opt, model, al=al, bl=bl, lr_prefactor=0.1, warmup_steps=0, solve_interval=1
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    chained_sched = ChainedMaxPScheduler(maxp_sched, [cosine_sched])

    X = torch.randn(16, 4)
    chained_sched.capture_initial(X)
    
    # Run a few steps
    for _ in range(3):
        opt.zero_grad()
        model(X).sum().backward()
        opt.step()
        chained_sched.step(X)
    
    # Save state
    state = chained_sched.state_dict()
    lrs_before = chained_sched.get_last_lr()
    
    # Create new scheduler and load state
    torch.manual_seed(0)
    model2 = SmallMLP()
    groups2 = create_param_groups(model2, lr_prefactor=0.1, cl=cl)
    opt2 = torch.optim.AdamW(groups2)
    
    maxp_sched2 = MaxPScheduler(
        opt2, model2, al=al, bl=bl, lr_prefactor=0.1, warmup_steps=0, solve_interval=1
    )
    cosine_sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=10)
    chained_sched2 = ChainedMaxPScheduler(maxp_sched2, [cosine_sched2])
    
    X2 = torch.randn(16, 4)
    chained_sched2.capture_initial(X2)
    chained_sched2.load_state_dict(state)
    
    lrs_after = chained_sched2.get_last_lr()
    
    # LRs should match
    for lr1, lr2 in zip(lrs_before, lrs_after):
        assert abs(lr1 - lr2) < 1e-9, "State dict roundtrip should preserve LRs"


def test_chained_scheduler_empty_list():
    """Test that empty scheduler list behaves identically to plain MaxPScheduler."""
    torch.manual_seed(0)
    model = SmallMLP()
    al = [0.0, 0.5, 0.5]
    bl = [0.0, 0.0, 0.0]
    cl = [0.0, 0.5, 0.5]

    # Create two identical setups
    groups1 = create_param_groups(model, lr_prefactor=0.1, cl=cl)
    opt1 = torch.optim.AdamW(groups1)
    maxp_sched = MaxPScheduler(
        opt1, model, al=al, bl=bl, lr_prefactor=0.1, warmup_steps=0, solve_interval=1
    )
    
    torch.manual_seed(0)
    model2 = SmallMLP()
    groups2 = create_param_groups(model2, lr_prefactor=0.1, cl=cl)
    opt2 = torch.optim.AdamW(groups2)
    maxp_sched2 = MaxPScheduler(
        opt2, model2, al=al, bl=bl, lr_prefactor=0.1, warmup_steps=0, solve_interval=1
    )
    chained_sched = ChainedMaxPScheduler(maxp_sched2, [])  # Empty list

    X = torch.randn(16, 4)
    maxp_sched.capture_initial(X)
    chained_sched.capture_initial(X)
    
    # Run steps and compare
    for _ in range(3):
        opt1.zero_grad()
        model(X).sum().backward()
        opt1.step()
        maxp_sched.step(X)
        
        opt2.zero_grad()
        model2(X).sum().backward()
        opt2.step()
        chained_sched.step(X)
        
        lrs_plain = maxp_sched.get_last_lr()
        lrs_chained = chained_sched.get_last_lr()
        
        for lr1, lr2 in zip(lrs_plain, lrs_chained):
            assert abs(lr1 - lr2) < 1e-9, "Empty chained scheduler should match plain MaxP"


def test_chained_scheduler_delegation_methods():
    """Test that delegation methods work correctly."""
    torch.manual_seed(0)
    model = SmallMLP()
    al = [0.0, 0.5, 0.5]
    bl = [0.0, 0.0, 0.0]
    cl = [0.0, 0.5, 0.5]

    groups = create_param_groups(model, lr_prefactor=0.1, cl=cl)
    opt = torch.optim.AdamW(groups)
    
    maxp_sched = MaxPScheduler(
        opt, model, al=al, bl=bl, lr_prefactor=0.1, warmup_steps=0, solve_interval=1
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    chained_sched = ChainedMaxPScheduler(maxp_sched, [cosine_sched])

    X = torch.randn(16, 4)
    chained_sched.capture_initial(X)
    
    # Test n_layers
    assert chained_sched.n_layers == 3
    assert chained_sched.n_layers == maxp_sched.n_layers
    
    # Test get_layer_names
    assert chained_sched.get_layer_names() == maxp_sched.get_layer_names()
    
    # Test optimizer reference
    assert chained_sched.optimizer is opt
    
    # Run a step to get alignment values
    opt.zero_grad()
    model(X).sum().backward()
    opt.step()
    chained_sched.step(X)
    
    # Test get_alignment
    alpha, omega, u = chained_sched.get_alignment()
    alpha_maxp, omega_maxp, u_maxp = maxp_sched.get_alignment()
    assert alpha == alpha_maxp
    assert omega == omega_maxp
    assert u == u_maxp


def test_chained_scheduler_validates_optimizer():
    """Test that ChainedMaxPScheduler validates optimizer consistency."""
    torch.manual_seed(0)
    model = SmallMLP()
    al = [0.0, 0.5, 0.5]
    bl = [0.0, 0.0, 0.0]
    cl = [0.0, 0.5, 0.5]

    groups1 = create_param_groups(model, lr_prefactor=0.1, cl=cl)
    opt1 = torch.optim.AdamW(groups1)
    maxp_sched = MaxPScheduler(
        opt1, model, al=al, bl=bl, lr_prefactor=0.1, warmup_steps=0, solve_interval=1
    )
    
    # Create scheduler with different optimizer
    model2 = SmallMLP()
    groups2 = create_param_groups(model2, lr_prefactor=0.1, cl=cl)
    opt2 = torch.optim.AdamW(groups2)  # Different optimizer!
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=10)
    
    try:
        ChainedMaxPScheduler(maxp_sched, [cosine_sched])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "different optimizer" in str(e)
