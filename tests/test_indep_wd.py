"""Independent weight decay — maxp.parametrization.indep_wd_rescale."""

import pytest
import torch
import torch.nn as nn

from maxp import ParametrizedModule, Parametrization
from maxp.parametrization import indep_wd_rescale


def _groups(lr_ref=3e-2, wd=0.1):
    return [
        {"layer_name": "_other", "lr": lr_ref, "weight_decay": wd},
        {"layer_name": "emb", "lr": lr_ref, "weight_decay": wd},
        {"layer_name": "hidden", "lr": lr_ref * 0.0442, "weight_decay": wd},
        {"layer_name": "head", "lr": lr_ref * 0.2415, "weight_decay": wd},
    ]


def test_products_uniform_ref_unchanged_idempotent():
    lr_ref, wd = 3e-2, 0.1
    groups = _groups(lr_ref, wd)
    indep_wd_rescale(groups)
    for g in groups:
        assert g["lr"] * g["weight_decay"] == pytest.approx(lr_ref * wd, rel=1e-12)
    assert groups[0]["weight_decay"] == pytest.approx(wd, rel=1e-12)
    once = [g["weight_decay"] for g in groups]
    indep_wd_rescale(groups)  # absolute assignment → idempotent
    assert [g["weight_decay"] for g in groups] == once


def test_shared_schedule_factor_preserves_products():
    groups = _groups()
    indep_wd_rescale(groups)
    for s in (0.01, 0.37, 1.0):  # shared LambdaLR factor
        for g in groups:
            assert (g["lr"] * s) * g["weight_decay"] == pytest.approx(3e-2 * s * 0.1, rel=1e-12)


def test_errors_without_usable_reference():
    groups = _groups()
    groups[0]["layer_name"] = "not_other"
    with pytest.raises(RuntimeError, match="exactly one"):
        indep_wd_rescale(groups)
    groups = _groups()
    del groups[0]["weight_decay"]
    with pytest.raises(RuntimeError, match="weight_decay"):
        indep_wd_rescale(groups)


def test_adamw_decay_step_uniform():
    """Zero grads → one AdamW step is pure decay p*(1 - lr_g*wd_g); after the
    rescale that factor is identical for multiplier-1 and multiplier-0.1 groups."""
    torch.manual_seed(0)
    lr_ref, wd = 3e-2, 0.1
    p_ref, p_low = nn.Parameter(torch.randn(4, 4)), nn.Parameter(torch.randn(4, 4))
    opt = torch.optim.AdamW(
        [{"params": [p_ref], "layer_name": "_other", "lr": lr_ref},
         {"params": [p_low], "layer_name": "head", "lr": lr_ref * 0.1}],
        lr=lr_ref, weight_decay=wd, foreach=False)
    indep_wd_rescale(opt)
    before = (p_ref.detach().clone(), p_low.detach().clone())
    p_ref.grad, p_low.grad = torch.zeros_like(p_ref), torch.zeros_like(p_low)
    opt.step()
    factor = 1.0 - lr_ref * wd
    assert torch.allclose(p_ref.detach(), before[0] * factor, atol=1e-7)
    assert torch.allclose(p_low.detach(), before[1] * factor, atol=1e-7)


class _MLP(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.emb = ParametrizedModule(nn.Linear(8, d, bias=False), width_dim=d, layer_type="embedding")
        self.hidden = ParametrizedModule(nn.Linear(d, d, bias=False), width_dim=d, layer_type="hidden")
        self.norm = nn.LayerNorm(d)  # unparametrized → the "_other" reference group
        self.head = ParametrizedModule(nn.Linear(d, 4, bias=False), width_dim=d, layer_type="readout")

    def forward(self, x):
        return self.head(self.norm(torch.relu(self.hidden(torch.relu(self.emb(x))))))


def test_parametrization_flag_maintains_invariant():
    """Parametrization(indep_wd=True) applies the rescale through refresh()
    and keeps it true after LR syncs (dynamic re-solves included)."""
    torch.manual_seed(0)
    param = Parametrization(_MLP(), lr_prefactor=1e-2, indep_wd=True)
    opt = torch.optim.AdamW(param.param_groups, lr=1e-2, weight_decay=0.1)
    param.refresh(optimizer=opt)
    products = {g["lr"] * g["weight_decay"] for g in opt.param_groups}
    assert max(products) == pytest.approx(min(products), rel=1e-12)
    assert max(products) == pytest.approx(1e-2 * 0.1, rel=1e-9)
    param.lr_prefactor = 3e-3  # simulate a schedule/prefactor change + re-sync
    param._sync_lrs(opt)
    products = {g["lr"] * g["weight_decay"] for g in opt.param_groups}
    assert max(products) == pytest.approx(min(products), rel=1e-12)
    assert max(products) == pytest.approx(3e-3 * 0.1, rel=1e-9)


# ---------------------------------------------------------------------------
# End-to-end: real training loops in every mode (classic µP-baseline, µP-full,
# dynamic maxP, maxP-meas), with a WSD-style scheduler, flag on and off.
# ---------------------------------------------------------------------------

def _build(mode, indep):
    torch.manual_seed(0)
    model = _MLP()
    kw = dict(lr_prefactor=1e-2, indep_wd=indep,
              warmup_steps=0, solve_interval=2, sample_size=4)
    if mode == "mup-full":
        param = Parametrization(model, alignment="full", **kw)
    elif mode == "maxP-meas":
        param = Parametrization(model, alignment="no",
                                alignment_overrides={"readout": (0.5, 0.5, 0.75)}, **kw)
    else:  # "classic" (µP baseline, alignment="no") and dynamic "maxP"
        param = Parametrization(model, alignment="no", **kw)
    opt = torch.optim.AdamW(param.param_groups, lr=1e-2, weight_decay=0.1)
    param.refresh(optimizer=opt)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.1 + 0.9 * min(1.0, s / 5))
    return model, param, opt, sched


def _check_invariant(opt, indep):
    ref = next(g for g in opt.param_groups if g["layer_name"] == "_other")
    if indep:
        assert ref["weight_decay"] == pytest.approx(0.1, rel=1e-12)
        for g in opt.param_groups:
            assert g["weight_decay"] == pytest.approx(
                0.1 * ref["lr"] / g["lr"], rel=1e-9), g["layer_name"]
            assert g["lr"] * g["weight_decay"] == pytest.approx(
                ref["lr"] * 0.1, rel=1e-9), g["layer_name"]
    else:
        for g in opt.param_groups:
            assert g["weight_decay"] == pytest.approx(0.1, rel=1e-12)


@pytest.mark.parametrize("mode", ["classic", "mup-full", "maxP", "maxP-meas"])
@pytest.mark.parametrize("indep", [True, False])
def test_training_loop_invariant(mode, indep):
    model, param, opt, sched = _build(mode, indep)
    dynamic = mode == "maxP"
    X = torch.randn(8, 8)
    if dynamic:
        param.capture_initial(X)

    lrs_seen = set()
    for step in range(12):
        opt.zero_grad()
        model(X).pow(2).mean().backward()
        opt.step()
        sched.step()
        if dynamic:
            if step == 6:  # prefactor jump → _sync_lrs must re-apply the rescale
                for g in opt.param_groups:
                    if g["layer_name"] == "_other":
                        g["lr"] *= 0.5
            param.lr_prefactor = next(
                g["lr"] for g in opt.param_groups if g["layer_name"] == "_other")
            param.step(X, opt)
        _check_invariant(opt, indep)
        lrs_seen.add(round(next(g["lr"] for g in opt.param_groups
                                if g["layer_name"] == "_other"), 12))
    assert len(lrs_seen) > 3  # the schedule really moved the LRs

    if indep:  # coupled control must be distinguishable: per-layer LRs differ
        lrs = {round(g["lr"], 12) for g in opt.param_groups}
        assert len(lrs) > 1
    param.remove_hooks()


@pytest.mark.parametrize("mode", ["classic", "mup-full", "maxP-meas"])
def test_pure_decay_trajectory_matches_analytic(mode):
    """Zero gradients → the only update is decay. Under indep_wd every weight,
    including low-LR layers, must shrink by exactly prod(1 - lr_ref(t) * 0.1)."""
    model, param, opt, sched = _build(mode, indep=True)
    w0 = {g["layer_name"]: [p.detach().clone() for p in g["params"]]
          for g in opt.param_groups}
    expected = 1.0
    for _ in range(8):
        for g in opt.param_groups:
            for p in g["params"]:
                p.grad = torch.zeros_like(p)
        lr_ref = next(g["lr"] for g in opt.param_groups if g["layer_name"] == "_other")
        opt.step()
        expected *= 1.0 - lr_ref * 0.1
        sched.step()
    for g in opt.param_groups:
        for p, p0 in zip(g["params"], w0[g["layer_name"]]):
            assert torch.allclose(p.detach(), p0 * expected, atol=1e-7), g["layer_name"]
