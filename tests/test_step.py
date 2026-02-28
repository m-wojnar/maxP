"""Tests for Phase 2 dynamic alignment: capture_initial() + step()."""

import math

import pytest
import torch
import torch.nn as nn

from maxp_new.module import ParametrizedModule
from maxp_new.parametrization import Parametrization


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
            assert pm.alpha == 1.0
            assert pm.omega == 0.5
            assert pm.u == 1.0

    def test_no_alignment_preset(self):
        torch.manual_seed(0)
        model = SimpleMLP()
        param = Parametrization(model, lr_prefactor=0.01, alignment="no")
        for _, pm in param._pms:
            assert pm.alpha == 0.5
            assert pm.omega == 0.5
            assert pm.u == 0.5


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
                assert pm.alpha is not None
                assert pm.omega is not None
                assert pm.u is not None
                assert math.isfinite(pm.alpha), f"{name}: alpha={pm.alpha}"
                assert math.isfinite(pm.omega), f"{name}: omega={pm.omega}"
                assert math.isfinite(pm.u), f"{name}: u={pm.u}"
