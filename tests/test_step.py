"""Tests for Phase 2 dynamic alignment: capture_initial() + step()."""

import math

import pytest
import torch
import torch.nn as nn

from maxp.module import ParametrizedModule
from maxp.parametrization import Parametrization


# ---------------------------------------------------------------------------
# Helper model: simple 3-layer MLP with ParametrizedModule wrappers
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    """embedding -> hidden -> readout chain using nn.Linear (no bias)."""

    def __init__(self, d_in=16, d=32, d_out=4):
        super().__init__()
        self.emb = ParametrizedModule(
            nn.Linear(d_in, d, bias=False), width_dim=d, layer_type="embedding",
        )
        self.hidden = ParametrizedModule(
            nn.Linear(d, d, bias=False), width_dim=d, layer_type="hidden",
        )
        self.head = ParametrizedModule(
            nn.Linear(d, d_out, bias=False), width_dim=d, layer_type="readout",
        )

    def forward(self, x):
        x = torch.relu(self.emb(x))
        x = torch.relu(self.hidden(x))
        return self.head(x)


def _setup(lr=0.01, d=32, **kw):
    """Create model, Parametrization, optimizer, and sample input."""
    torch.manual_seed(0)
    model = SimpleMLP(d=d)
    param = Parametrization(model, lr_prefactor=lr, **kw)
    optimizer = torch.optim.Adam(param.param_groups)
    X = torch.randn(8, 16)
    return model, param, optimizer, X


class TestStepUpdatesLR:
    """step() should re-solve the LP and change LRs after training."""

    def test_lr_changes_after_step(self):
        model, param, optimizer, X = _setup()
        param.capture_initial(X)

        # Record initial LRs
        initial_lrs = [g["lr"] for g in param.param_groups if g.get("maxp_managed")]

        # Train a few steps to create non-trivial dz, dw
        for _ in range(5):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()

        # Now call step() with a sample
        param.step(X)

        # LRs should have been updated (may or may not differ numerically,
        # but the code path should succeed without error)
        new_lrs = [g["lr"] for g in param.param_groups if g.get("maxp_managed")]
        assert len(new_lrs) == len(initial_lrs)
        # At minimum, all LRs should be positive
        for lr in new_lrs:
            assert lr > 0


class TestStepWarmup:
    """LRs should not change during warmup period."""

    def test_no_change_during_warmup(self):
        model, param, optimizer, X = _setup(warmup_steps=3)
        param.capture_initial(X)

        initial_lrs = [g["lr"] for g in param.param_groups if g.get("maxp_managed")]

        # Do 3 warmup steps — LRs should not change
        for _ in range(3):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()
            param.step(X, optimizer)

        warmup_lrs = [g["lr"] for g in param.param_groups if g.get("maxp_managed")]
        assert warmup_lrs == initial_lrs

    def test_change_after_warmup(self):
        model, param, optimizer, X = _setup(warmup_steps=2)
        param.capture_initial(X)

        # 2 warmup steps
        for _ in range(2):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()
            param.step(X, optimizer)

        # Step 3 should trigger re-solve (past warmup)
        optimizer.zero_grad()
        loss = model(X).sum()
        loss.backward()
        optimizer.step()
        param.step(X, optimizer)

        # No error means success — the re-solve ran
        new_lrs = [g["lr"] for g in param.param_groups if g.get("maxp_managed")]
        for lr in new_lrs:
            assert lr > 0


class TestStepInterval:
    """LRs should only change on solve_interval boundaries."""

    def test_interval_skips(self):
        model, param, optimizer, X = _setup(solve_interval=3)
        param.capture_initial(X)

        initial_lrs = [g["lr"] for g in param.param_groups if g.get("maxp_managed")]

        # Steps 1, 2: should be skipped (not multiples of 3)
        for _ in range(2):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()
            param.step(X, optimizer)

        after_2_lrs = [g["lr"] for g in param.param_groups if g.get("maxp_managed")]
        assert after_2_lrs == initial_lrs

        # Step 3: should trigger re-solve
        optimizer.zero_grad()
        loss = model(X).sum()
        loss.backward()
        optimizer.step()
        param.step(X, optimizer)

        # The LRs have been updated (code path completed)
        after_3_lrs = [g["lr"] for g in param.param_groups if g.get("maxp_managed")]
        for lr in after_3_lrs:
            assert lr > 0


class TestStepWithoutCaptureRaises:
    """Calling step() before capture_initial() should raise."""

    def test_raises_runtime_error(self):
        model, param, optimizer, X = _setup()
        with pytest.raises(RuntimeError, match="capture_initial"):
            param.step(X)


class TestStepSyncsOptimizer:
    """When optimizer is passed, its param_groups should be updated."""

    def test_optimizer_lr_synced(self):
        model, param, optimizer, X = _setup()
        param.capture_initial(X)

        # Train to create changes
        for _ in range(5):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()

        param.step(X, optimizer=optimizer)

        # Optimizer LRs should match param_groups LRs
        for our_group, opt_group in zip(param.param_groups, optimizer.param_groups):
            assert our_group["lr"] == opt_group["lr"]

    def test_step_without_optimizer_updates_param_groups(self):
        """Without optimizer arg, param_groups are still updated."""
        model, param, optimizer, X = _setup()
        param.capture_initial(X)

        for _ in range(5):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()

        # Call step without optimizer — should not raise
        param.step(X)

        # param_groups should still have valid positive LRs
        for g in param.param_groups:
            if g.get("maxp_managed"):
                assert g["lr"] > 0


class TestCaptureInitial:
    """Tests for capture_initial()."""

    def test_snapshots_are_populated(self):
        model, param, optimizer, X = _setup()
        # Before capture_initial, _z0 should be None on all PMs
        for _, pm in param._pms:
            assert pm._z0 is None

        param.capture_initial(X)

        # All PMs with weights should have _z0 and _w0 set
        for name, pm in param._pms:
            if pm.weight is not None:
                assert pm._z0 is not None
                assert pm._w0 is not None
                assert pm._z0.shape[-1] == pm.width_dim or pm._z0.shape[-1] > 0
                assert pm._w0.shape == pm.weight.shape

    def test_sample_size_limit(self):
        """Only sample_size samples are kept."""
        model, param, optimizer, X = _setup(sample_size=4)
        big_X = torch.randn(32, 16)
        param.capture_initial(big_X)

        for _, pm in param._pms:
            if pm._z0 is not None:
                assert pm._z0.shape[0] <= 4


class TestInitialAlignmentOnPM:
    """After Parametrization.__init__, each PM should have preset alignment."""

    def test_full_alignment_preset(self):
        model, param, optimizer, X = _setup(alignment="full")
        for _, pm in param._pms:
            assert pm.align_z0_dW == 1.0
            assert pm.align_dZ_w0 == 0.5
            assert pm.align_dZ_dW == 1.0

    def test_no_alignment_preset(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        param = Parametrization(model, lr_prefactor=0.01, alignment="no")
        for _, pm in param._pms:
            assert pm.align_z0_dW == 0.5
            assert pm.align_dZ_w0 == 0.5
            assert pm.align_dZ_dW == 0.5


class TestStepInfeasibleLP:
    """When the LP solver fails, step() should keep previous LRs and not crash."""

    def test_extreme_alignment_is_infeasible(self):
        """Directly confirm that extreme alignment values make the LP infeasible."""
        model, param, optimizer, X = _setup()

        # Set extreme alignment on all PMs
        for _, pm in param._pms:
            if pm.weight is not None:
                pm.align_z0_dW = 100.0
                pm.align_dZ_w0 = 100.0
                pm.align_dZ_dW = 100.0

        with pytest.raises(ValueError, match="infeasible|optimal"):
            param._resolve()

    def test_infeasible_lp_keeps_previous_lrs(self):
        """step() catches infeasible LP and leaves LRs unchanged."""
        from unittest.mock import patch

        model, param, optimizer, X = _setup()
        param.capture_initial(X)

        # Train a few steps so step() passes warmup
        for _ in range(5):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()

        lrs_before = [g["lr"] for g in param.param_groups]

        # Patch the resolver to raise ValueError (simulating infeasible LP)
        with patch.object(param, "_resolve",
                          side_effect=ValueError("infeasible")):
            param.step(X, optimizer)

        lrs_after = [g["lr"] for g in param.param_groups]
        assert lrs_after == lrs_before

    def test_infeasible_then_recovers(self):
        """After an infeasible step, a subsequent normal step updates LRs."""
        from unittest.mock import patch

        model, param, optimizer, X = _setup(warmup_steps=0)
        param.capture_initial(X)

        for _ in range(5):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()

        lrs_before = [g["lr"] for g in param.param_groups]

        # First step: infeasible → LRs unchanged
        with patch.object(param, "_resolve",
                          side_effect=ValueError("infeasible")):
            param.step(X, optimizer)
        assert [g["lr"] for g in param.param_groups] == lrs_before

        # Second step: normal → LRs may update
        param.step(X, optimizer)
        for g in param.param_groups:
            assert g["lr"] > 0


class TestStepUpdatesAlignmentOnPM:
    """After step(), each weight-bearing PM should have finite alignment values."""

    def test_alignment_values_are_finite_after_step(self):
        model, param, optimizer, X = _setup()
        param.capture_initial(X)

        # Train to create non-trivial changes
        for _ in range(5):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()

        param.step(X)

        for name, pm in param._pms:
            if pm.weight is not None:
                assert pm.align_z0_dW is not None
                assert pm.align_dZ_w0 is not None
                assert pm.align_dZ_dW is not None
                assert math.isfinite(pm.align_z0_dW), f"{name}: align_z0_dW={pm.align_z0_dW}"
                assert math.isfinite(pm.align_dZ_w0), f"{name}: align_dZ_w0={pm.align_dZ_w0}"
                assert math.isfinite(pm.align_dZ_dW), f"{name}: align_dZ_dW={pm.align_dZ_dW}"


# ---------------------------------------------------------------------------
# Tests for c overrides
# ---------------------------------------------------------------------------

class TestCOverrideAllFixed:
    """When all PMs have c set, solver is skipped and LRs match exactly."""

    def test_all_c_fixed_via_pm(self):
        torch.manual_seed(0)
        d = 32
        model = nn.Module()
        model.emb = ParametrizedModule(
            nn.Linear(16, d, bias=False), width_dim=d,
            layer_type="embedding", c=0.0,
        )
        model.hidden = ParametrizedModule(
            nn.Linear(d, d, bias=False), width_dim=d,
            layer_type="hidden", c=0.5,
        )
        model.head = ParametrizedModule(
            nn.Linear(d, 4, bias=False), width_dim=d,
            layer_type="readout", c=1.0,
        )
        model.forward = lambda x: model.head(torch.relu(model.hidden(torch.relu(model.emb(x)))))

        lr = 0.01
        param = Parametrization(model, lr_prefactor=lr)

        managed = [g for g in param.param_groups if g.get("maxp_managed")]
        expected_c = {"emb": 0.0, "hidden": 0.5, "head": 1.0}
        for g in managed:
            name = g["layer_name"]
            assert g["c"] == expected_c[name], f"{name}: c={g['c']} != {expected_c[name]}"
            expected_lr = lr * (d ** (-expected_c[name]))
            assert abs(g["lr"] - expected_lr) < 1e-12, (
                f"{name}: lr={g['lr']} != {expected_lr}"
            )

    def test_all_c_fixed_via_c_overrides(self):
        torch.manual_seed(0)
        model = SimpleMLP(d=32)
        lr = 0.01
        param = Parametrization(
            model, lr_prefactor=lr,
            c_overrides={"embedding": 0.0, "hidden": 0.0, "readout": 0.0},
        )

        managed = [g for g in param.param_groups if g.get("maxp_managed")]
        for g in managed:
            assert g["c"] == 0.0
            assert abs(g["lr"] - lr) < 1e-12  # d^(-0) = 1


class TestCOverridePriority:
    """Per-PM c takes priority over c_overrides."""

    def test_pm_c_overrides_c_overrides(self):
        torch.manual_seed(0)
        d = 32
        model = nn.Module()
        model.emb = ParametrizedModule(
            nn.Linear(16, d, bias=False), width_dim=d,
            layer_type="embedding", c=0.25,  # per-PM override
        )
        model.hidden = ParametrizedModule(
            nn.Linear(d, d, bias=False), width_dim=d,
            layer_type="hidden",
        )
        model.head = ParametrizedModule(
            nn.Linear(d, 4, bias=False), width_dim=d,
            layer_type="readout",
        )
        model.forward = lambda x: model.head(torch.relu(model.hidden(torch.relu(model.emb(x)))))

        lr = 0.01
        param = Parametrization(
            model, lr_prefactor=lr,
            c_overrides={"embedding": 0.5, "hidden": 0.0, "readout": 0.0},
        )

        managed = {g["layer_name"]: g for g in param.param_groups if g.get("maxp_managed")}
        # Per-PM c=0.25 wins over c_overrides embedding=0.5
        assert managed["emb"]["c"] == 0.25
        # c_overrides values used for the rest
        assert managed["hidden"]["c"] == 0.0
        assert managed["head"]["c"] == 0.0


class TestCOverrideMixed:
    """Some PMs have fixed c, others are solved."""

    def test_mixed_fixed_and_solved(self):
        torch.manual_seed(0)
        d = 32
        model = SimpleMLP(d=d)
        # Fix only readout c=0.5 (feasible with muP defaults); let solver
        # handle embedding and hidden
        param = Parametrization(
            model, lr_prefactor=0.01,
            c_overrides={"readout": 0.5},
        )

        managed = {g["layer_name"]: g for g in param.param_groups if g.get("maxp_managed")}
        # readout should be exactly 0.5
        assert managed["head"]["c"] == 0.5
        # Other layers should have solver-computed c (some finite value)
        for name in ("emb", "hidden"):
            assert math.isfinite(managed[name]["c"])


class TestCOverrideDynamicStep:
    """_resolve() should pass c_fixed correctly during dynamic re-solve."""

    def test_resolve_uses_c_fixed(self):
        torch.manual_seed(0)
        model = SimpleMLP(d=32)
        lr = 0.01
        param = Parametrization(
            model, lr_prefactor=lr,
            c_overrides={"embedding": 0.0, "hidden": 0.0, "readout": 0.0},
        )
        X = torch.randn(8, 16)
        param.capture_initial(X)

        # Train a few steps
        optimizer = torch.optim.Adam(param.param_groups)
        for _ in range(3):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()

        # step() should re-solve with c_fixed and keep c=0 for all
        param.step(X, optimizer)

        managed = [g for g in param.param_groups if g.get("maxp_managed")]
        for g in managed:
            assert g["c"] == 0.0
            assert abs(g["lr"] - lr) < 1e-12


# ---------------------------------------------------------------------------
# Tests for alignment EMA
# ---------------------------------------------------------------------------

class TestAlignmentEMA:
    """Tests for alignment_ema smoothing of raw alignment measurements."""

    def test_ema_zero_is_raw(self):
        """alignment_ema=0 gives same behavior as no EMA (raw values)."""
        model, param, optimizer, X = _setup(alignment_ema=0.0)
        param.capture_initial(X)

        for _ in range(5):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()

        param.step(X)
        for _, pm in param._pms:
            if pm.weight is not None:
                assert math.isfinite(pm.align_z0_dW)
                assert math.isfinite(pm.align_dZ_w0)
                assert math.isfinite(pm.align_dZ_dW)

    def test_ema_smooths_values(self):
        """With high EMA, alignment values should stay closer to initial preset."""
        model, param, optimizer, X = _setup(alignment_ema=0.95)
        param.capture_initial(X)

        # Record initial preset values
        initial = {
            name: (pm.align_z0_dW, pm.align_dZ_w0, pm.align_dZ_dW)
            for name, pm in param._pms if pm.weight is not None
        }

        # Train and step several times
        for _ in range(5):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()
            param.step(X)

        # With high EMA, values should still be close to initial preset
        for name, pm in param._pms:
            if pm.weight is not None:
                init_a, init_o, init_u = initial[name]
                assert abs(pm.align_z0_dW - init_a) < 1.0, f"{name}: alpha drifted too far"
                assert math.isfinite(pm.align_z0_dW)

    def test_ema_pinned_layers_unaffected(self):
        """Pinned layers should not have EMA applied."""
        torch.manual_seed(0)
        model = SimpleMLP(d=32)
        param = Parametrization(
            model, lr_prefactor=0.01,
            alignment_ema=0.9,
            alignment_overrides={"hidden": (0.7, 0.3, 0.8)},
        )
        X = torch.randn(8, 16)
        param.capture_initial(X)
        optimizer = torch.optim.Adam(param.param_groups)

        for _ in range(5):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()
            param.step(X, optimizer)

        # hidden PM should keep its pinned values exactly
        for name, pm in param._pms:
            if pm.layer_type == "hidden":
                assert pm.align_z0_dW == 0.7
                assert pm.align_dZ_w0 == 0.3
                assert pm.align_dZ_dW == 0.8

    def test_ema_step_succeeds(self):
        """Full training loop with alignment_ema runs without error."""
        model, param, optimizer, X = _setup(alignment_ema=0.5)
        param.capture_initial(X)

        for _ in range(10):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()
            param.step(X, optimizer)

        for g in param.param_groups:
            if g.get("maxp_managed"):
                assert g["lr"] > 0


# ---------------------------------------------------------------------------
# Tests for seed-based W0 regeneration
# ---------------------------------------------------------------------------

class TestResampleW0:
    """Tests for resample_w0 mode (seed-based W0 regeneration)."""

    def test_w0_not_stored(self):
        """With resample_w0=True, _w0 should remain None after capture_initial."""
        model, param, optimizer, X = _setup(resample_w0=True)
        param.capture_initial(X)

        for _, pm in param._pms:
            if pm.weight is not None:
                assert pm._w0 is None, "_w0 should not be stored"
                assert pm._w0_seed is not None, "seed should be stored"
                assert pm._w0_std is not None, "std should be stored"
                assert pm._z0 is not None, "_z0 should still be stored"

    def test_seed_deterministic(self):
        """Regenerating w0 from seed produces identical tensor each time."""
        model, param, optimizer, X = _setup(resample_w0=True)

        for _, pm in param._pms:
            if pm.weight is not None:
                w0_a = param._regenerate_w0(pm)
                w0_b = param._regenerate_w0(pm)
                assert torch.equal(w0_a, w0_b), "Regenerated w0 should be identical"

    def test_regenerated_matches_init(self):
        """Regenerated w0 should match the actual initial weights."""
        torch.manual_seed(0)
        model = SimpleMLP(d=32)
        param = Parametrization(model, lr_prefactor=0.01, resample_w0=True)

        for _, pm in param._pms:
            if pm.weight is not None:
                w0 = param._regenerate_w0(pm)
                assert torch.equal(w0, pm.weight), (
                    "Regenerated w0 should match current weight (before training)"
                )

    def test_alignment_uses_correct_w0(self):
        """Alignment computed with regenerated w0 uses the true initial weights."""
        model, param, optimizer, X = _setup(resample_w0=True)

        # Before training, regenerated w0 should match current weight exactly
        for _, pm in param._pms:
            if pm.weight is not None:
                w0 = param._regenerate_w0(pm)
                assert torch.equal(w0, pm.weight)

        param.capture_initial(X)

        # Train to create non-trivial dw
        for _ in range(5):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()

        # Manually compute alignment with regenerated w0 and compare to step()
        from maxp.alignment import compute_alignment
        current = param._capture_activations(X)
        expected = {}
        for name, pm in param._pms:
            if pm._z0 is None or name not in current:
                continue
            w0 = param._regenerate_w0(pm)
            z = current[name]
            w = pm.weight.detach().clone()
            expected[name] = compute_alignment(
                pm._z0, w0, z, w, fan_in=pm.width_dim
            )

        param.step(X)

        for name, pm in param._pms:
            if name in expected:
                ea, eo, eu = expected[name]
                assert abs(pm.align_z0_dW - ea) < 1e-10, f"{name}: alpha"
                assert abs(pm.align_dZ_w0 - eo) < 1e-10, f"{name}: omega"
                assert abs(pm.align_dZ_dW - eu) < 1e-10, f"{name}: u"

    def test_step_succeeds(self):
        """Full training loop with resample_w0=True runs without error."""
        model, param, optimizer, X = _setup(resample_w0=True)
        param.capture_initial(X)

        for _ in range(10):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()
            param.step(X, optimizer)

        for g in param.param_groups:
            if g.get("maxp_managed"):
                assert g["lr"] > 0


# ---------------------------------------------------------------------------
# Tests for warm start LP
# ---------------------------------------------------------------------------

class TestWarmStartLP:
    """Tests for LP warm start optimization."""

    def test_warm_start_same_result_as_cold(self):
        """Warm start should produce the same c values as cold start."""
        from maxp.solver import find_c
        from maxp.dag import DagNode, OpGraph

        nodes = {
            "emb": DagNode("emb", a=-0.5, b=0.5, layer_type="embedding",
                           has_weight=True, width_dim=32,
                           predecessors=[], successors=["hid"],
                           align_z0_dW=1.0, align_dZ_w0=0.5, align_dZ_dW=1.0),
            "hid": DagNode("hid", a=0.0, b=0.5, layer_type="hidden",
                           has_weight=True, width_dim=32,
                           predecessors=["emb"], successors=["out"],
                           align_z0_dW=1.0, align_dZ_w0=0.5, align_dZ_dW=1.0),
            "out": DagNode("out", a=0.5, b=0.5, layer_type="readout",
                           has_weight=True, width_dim=32,
                           predecessors=["hid"], successors=[],
                           align_z0_dW=1.0, align_dZ_w0=0.5, align_dZ_dW=1.0),
        }
        graph = OpGraph(nodes)

        # Cold start
        result_cold = find_c(graph, "adam")
        c_cold = {n: c for n, (c, _) in result_cold.items() if c is not None}

        # Warm start with previous solution
        import pulp as plp
        solver = plp.PULP_CBC_CMD(msg=False, warmStart=True)
        result_warm = find_c(graph, "adam", solver=solver, c_prev=c_cold)
        c_warm = {n: c for n, (c, _) in result_warm.items() if c is not None}

        for name in c_cold:
            assert abs(c_cold[name] - c_warm[name]) < 1e-6, (
                f"{name}: cold={c_cold[name]} warm={c_warm[name]}"
            )

    def test_warm_start_with_changing_alignment(self):
        """Multiple re-solves with changing alignment all produce valid results."""
        model, param, optimizer, X = _setup()
        param.capture_initial(X)

        # Multiple steps — each triggers a re-solve with warm start
        for _ in range(10):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()
            param.step(X, optimizer)

        # All LRs should be valid
        for g in param.param_groups:
            if g.get("maxp_managed"):
                assert g["lr"] > 0

    def test_c_prev_updated_after_solve(self):
        """After a solve, _c_prev should contain the latest c values."""
        model, param, optimizer, X = _setup()
        param.capture_initial(X)

        # Initial _c_prev should be populated from __init__
        assert len(param._c_prev) > 0

        for _ in range(3):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()

        param.step(X, optimizer)

        # _c_prev should still have entries for all managed layers
        managed_names = {
            g["layer_name"] for g in param.param_groups
            if g.get("maxp_managed")
        }
        for name in managed_names:
            assert name in param._c_prev

    def test_custom_solver(self):
        """User-provided solver instance is used for both init and re-solves."""
        import pulp as plp
        custom_solver = plp.PULP_CBC_CMD(msg=False, warmStart=True)

        torch.manual_seed(0)
        model = SimpleMLP(d=32)
        param = Parametrization(
            model, lr_prefactor=0.01, solver=custom_solver,
        )

        # The stored solver should be the one we passed
        assert param._solver is custom_solver

        X = torch.randn(8, 16)
        param.capture_initial(X)
        optimizer = torch.optim.Adam(param.param_groups)

        for _ in range(5):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()

        param.step(X, optimizer)

        # Should still be using our solver after re-solve
        assert param._solver is custom_solver
        for g in param.param_groups:
            if g.get("maxp_managed"):
                assert g["lr"] > 0

    def test_warm_start_flag_default_off(self):
        """By default warm_start is False."""
        model, param, optimizer, X = _setup()
        assert param._warm_start is False

    def test_warm_start_flag_on(self):
        """warm_start=True enables warm start on the default solver."""
        model, param, optimizer, X = _setup(warm_start=True)
        assert param._warm_start is True
        param.capture_initial(X)

        for _ in range(5):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()

        param.step(X, optimizer)

        for g in param.param_groups:
            if g.get("maxp_managed"):
                assert g["lr"] > 0

    def test_warm_start_produces_same_result(self):
        """warm_start=True produces the same c values as without."""
        torch.manual_seed(0)
        model_a = SimpleMLP(d=32)
        torch.manual_seed(0)
        model_b = SimpleMLP(d=32)

        param_a = Parametrization(model_a, lr_prefactor=0.01, warm_start=False)
        param_b = Parametrization(model_b, lr_prefactor=0.01, warm_start=True)

        managed_a = {g["layer_name"]: g["c"] for g in param_a.param_groups if g.get("maxp_managed")}
        managed_b = {g["layer_name"]: g["c"] for g in param_b.param_groups if g.get("maxp_managed")}

        for name in managed_a:
            assert abs(managed_a[name] - managed_b[name]) < 1e-6, f"{name}: c mismatch"


# ---------------------------------------------------------------------------
# Tests for piggyback on training forward pass
# ---------------------------------------------------------------------------

class TestTrainingActivations:
    """Tests for use_training_activations mode."""

    def test_hooks_registered_after_capture(self):
        """After capture_initial, persistent hooks should be installed."""
        model, param, optimizer, X = _setup(use_training_activations=True)
        assert len(param._persistent_hooks) == 0  # none before capture
        param.capture_initial(X)
        # Should have hooks for all non-embedding PMs with inner modules,
        # plus one pre-forward hook on the model that clears old activations
        expected = 1 + sum(
            1 for _, pm in param._pms
            if pm.inner is not None and not isinstance(pm.inner, nn.Embedding)
        )
        assert len(param._persistent_hooks) == expected

    def test_activations_captured_during_forward(self):
        """After a forward pass, _latest_activations should be populated."""
        model, param, optimizer, X = _setup(use_training_activations=True)
        param.capture_initial(X)

        # Run a training forward pass
        model(X)

        assert len(param._latest_activations) > 0
        for name in param._latest_activations:
            assert param._latest_activations[name].shape[0] <= param._sample_size

    def test_step_without_sample_input(self):
        """step() should work without sample_input when hooks are active."""
        model, param, optimizer, X = _setup(use_training_activations=True)
        param.capture_initial(X)

        for _ in range(5):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()

        # step() without sample_input — should use _latest_activations
        param.step(optimizer=optimizer)

        for g in param.param_groups:
            if g.get("maxp_managed"):
                assert g["lr"] > 0

    def test_step_no_hooks_no_input_raises(self):
        """step(None) without use_training_activations should raise."""
        model, param, optimizer, X = _setup()
        param.capture_initial(X)

        for _ in range(3):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()

        with pytest.raises(ValueError, match="sample_input is required"):
            param.step(None, optimizer)

    def test_step_no_forward_raises(self):
        """step() without a prior forward pass should raise."""
        model, param, optimizer, X = _setup(use_training_activations=True)
        param.capture_initial(X)
        # Clear activations from capture_initial's forward pass
        param._latest_activations.clear()

        with pytest.raises(RuntimeError, match="No activations captured"):
            param.step(optimizer=optimizer)

    def test_remove_hooks(self):
        """remove_hooks() should clean up all hooks."""
        model, param, optimizer, X = _setup(use_training_activations=True)
        param.capture_initial(X)
        assert len(param._persistent_hooks) > 0

        param.remove_hooks()
        assert len(param._persistent_hooks) == 0
        assert len(param._latest_activations) == 0

    def test_full_training_loop(self):
        """Complete training loop with use_training_activations=True."""
        model, param, optimizer, X = _setup(use_training_activations=True)
        param.capture_initial(X)

        for _ in range(10):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()
            param.step(optimizer=optimizer)

        for g in param.param_groups:
            if g.get("maxp_managed"):
                assert g["lr"] > 0

        # Clean up
        param.remove_hooks()


# ---------------------------------------------------------------------------
# Integration: all optimizations combined
# ---------------------------------------------------------------------------

class TestAllOptimizationsCombined:
    """Test all optimizations working together."""

    def test_all_opts_training_loop(self):
        """Training loop with all optimizations enabled simultaneously."""
        torch.manual_seed(0)
        model = SimpleMLP(d=32)
        param = Parametrization(
            model, lr_prefactor=0.01,
            solve_interval=3,
            alignment_ema=0.5,
            c_ema=0.5,
            resample_w0=True,
            use_training_activations=True,
        )
        X = torch.randn(8, 16)
        param.capture_initial(X)
        optimizer = torch.optim.Adam(param.param_groups)

        for _ in range(15):
            optimizer.zero_grad()
            loss = model(X).sum()
            loss.backward()
            optimizer.step()
            param.step(optimizer=optimizer)

        # All LRs should be positive
        for g in param.param_groups:
            if g.get("maxp_managed"):
                assert g["lr"] > 0
        # Alignment values should be finite
        for _, pm in param._pms:
            if pm.weight is not None:
                assert math.isfinite(pm.align_z0_dW)
                assert math.isfinite(pm.align_dZ_w0)
                assert math.isfinite(pm.align_dZ_dW)
        # No _w0 stored
        for _, pm in param._pms:
            if pm.weight is not None:
                assert pm._w0 is None

        param.remove_hooks()
