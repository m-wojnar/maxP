#!/usr/bin/env python3
"""Integration test: train a small MLP with all scheduler optimizations.

Runs short training loops on synthetic data to verify that the new
optimization features (alignment_ema, resample_w0, warm start,
use_training_activations, custom solver) work correctly in realistic
training, not just in unit-test isolation.

Also benchmarks time and memory: baseline vs fully optimized.

Usage:
    python -m pytest tests/test_optimizations_integration.py -v -s
"""

import math
import time
import tracemalloc

import torch
import torch.nn as nn
import torch.nn.functional as F

from maxp.module import ParametrizedModule
from maxp.parametrization import Parametrization


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SmallMLP(nn.Module):
    """4-layer MLP for integration testing."""

    def __init__(self, d_in=32, d=64, d_out=10, n_hidden=2):
        super().__init__()
        self.emb = ParametrizedModule(
            nn.Linear(d_in, d, bias=False), width_dim=d, layer_type="embedding",
        )
        self.hiddens = nn.ModuleList([
            ParametrizedModule(
                nn.Linear(d, d, bias=False), width_dim=d, layer_type="hidden",
            )
            for _ in range(n_hidden)
        ])
        self.head = ParametrizedModule(
            nn.Linear(d, d_out, bias=False), width_dim=d, layer_type="readout",
        )

    def forward(self, x):
        x = torch.relu(self.emb(x))
        for h in self.hiddens:
            x = torch.relu(h(x))
        return self.head(x)


class WiderMLP(nn.Module):
    """Wider MLP to make memory/time differences more visible."""

    def __init__(self, d_in=128, d=512, d_out=10, n_hidden=4):
        super().__init__()
        self.emb = ParametrizedModule(
            nn.Linear(d_in, d, bias=False), width_dim=d, layer_type="embedding",
        )
        self.hiddens = nn.ModuleList([
            ParametrizedModule(
                nn.Linear(d, d, bias=False), width_dim=d, layer_type="hidden",
            )
            for _ in range(n_hidden)
        ])
        self.head = ParametrizedModule(
            nn.Linear(d, d_out, bias=False), width_dim=d, layer_type="readout",
        )

    def forward(self, x):
        x = torch.relu(self.emb(x))
        for h in self.hiddens:
            x = torch.relu(h(x))
        return self.head(x)


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_loop(model, param, optimizer, X, Y, n_steps, use_hooks=False):
    """Run n_steps of training, return list of losses."""
    losses = []
    for step in range(n_steps):
        idx = torch.randint(0, X.shape[0], (32,))
        optimizer.zero_grad()
        logits = model(X[idx])
        loss = F.cross_entropy(logits, Y[idx])
        loss.backward()
        optimizer.step()

        if use_hooks:
            param.step(optimizer=optimizer)
        else:
            sample = X[:32]
            param.step(sample, optimizer)

        losses.append(loss.item())
    return losses


# ---------------------------------------------------------------------------
# Individual feature tests
# ---------------------------------------------------------------------------

class TestBaselineDynamicTraining:
    """Baseline: dynamic maxP without any optimizations."""

    def test_baseline_trains(self):
        torch.manual_seed(0)
        X = torch.randn(200, 32)
        Y = torch.randint(0, 10, (200,))

        model = SmallMLP()
        param = Parametrization(model, lr_prefactor=0.01)
        optimizer = torch.optim.Adam(param.param_groups)
        param.capture_initial(X[:32])

        losses = train_loop(model, param, optimizer, X, Y, n_steps=50)

        assert all(math.isfinite(l) for l in losses), "Training diverged"
        assert losses[-1] < losses[0], "Loss should decrease"
        print(f"  Baseline: loss {losses[0]:.4f} -> {losses[-1]:.4f}")


class TestAlignmentEMATraining:
    """Training with alignment_ema enabled."""

    def test_alignment_ema_trains(self):
        torch.manual_seed(0)
        X = torch.randn(200, 32)
        Y = torch.randint(0, 10, (200,))

        model = SmallMLP()
        param = Parametrization(model, lr_prefactor=0.01, alignment_ema=0.8)
        optimizer = torch.optim.Adam(param.param_groups)
        param.capture_initial(X[:32])

        losses = train_loop(model, param, optimizer, X, Y, n_steps=50)

        assert all(math.isfinite(l) for l in losses), "Training diverged"
        assert losses[-1] < losses[0], "Loss should decrease"
        print(f"  Alignment EMA: loss {losses[0]:.4f} -> {losses[-1]:.4f}")


class TestResampleW0Training:
    """Training with resample_w0 enabled."""

    def test_resample_w0_trains(self):
        torch.manual_seed(0)
        X = torch.randn(200, 32)
        Y = torch.randint(0, 10, (200,))

        model = SmallMLP()
        param = Parametrization(model, lr_prefactor=0.01, resample_w0=True)
        optimizer = torch.optim.Adam(param.param_groups)
        param.capture_initial(X[:32])

        for _, pm in param._pms:
            if pm.weight is not None:
                assert pm._w0 is None, "W0 should not be stored"

        losses = train_loop(model, param, optimizer, X, Y, n_steps=50)

        assert all(math.isfinite(l) for l in losses), "Training diverged"
        assert losses[-1] < losses[0], "Loss should decrease"
        print(f"  Resample W0: loss {losses[0]:.4f} -> {losses[-1]:.4f}")


class TestTrainingActivationsTraining:
    """Training with use_training_activations (piggyback hooks)."""

    def test_training_activations_trains(self):
        torch.manual_seed(0)
        X = torch.randn(200, 32)
        Y = torch.randint(0, 10, (200,))

        model = SmallMLP()
        param = Parametrization(
            model, lr_prefactor=0.01, use_training_activations=True,
        )
        optimizer = torch.optim.Adam(param.param_groups)
        param.capture_initial(X[:32])

        losses = train_loop(model, param, optimizer, X, Y, n_steps=50, use_hooks=True)

        assert all(math.isfinite(l) for l in losses), "Training diverged"
        assert losses[-1] < losses[0], "Loss should decrease"
        print(f"  Training activations: loss {losses[0]:.4f} -> {losses[-1]:.4f}")

        param.remove_hooks()


class TestSolveIntervalTraining:
    """Training with solve_interval > 1."""

    def test_solve_interval_trains(self):
        torch.manual_seed(0)
        X = torch.randn(200, 32)
        Y = torch.randint(0, 10, (200,))

        model = SmallMLP()
        param = Parametrization(
            model, lr_prefactor=0.01, solve_interval=10, c_ema=0.5,
        )
        optimizer = torch.optim.Adam(param.param_groups)
        param.capture_initial(X[:32])

        losses = train_loop(model, param, optimizer, X, Y, n_steps=50)

        assert all(math.isfinite(l) for l in losses), "Training diverged"
        assert losses[-1] < losses[0], "Loss should decrease"
        print(f"  Solve interval=10: loss {losses[0]:.4f} -> {losses[-1]:.4f}")


class TestCustomSolverTraining:
    """Training with a user-provided PuLP solver."""

    def test_custom_solver_trains(self):
        import pulp as plp

        torch.manual_seed(0)
        X = torch.randn(200, 32)
        Y = torch.randint(0, 10, (200,))

        solver = plp.PULP_CBC_CMD(msg=False, warmStart=True)
        model = SmallMLP()
        param = Parametrization(
            model, lr_prefactor=0.01, solver=solver,
        )
        optimizer = torch.optim.Adam(param.param_groups)
        param.capture_initial(X[:32])

        losses = train_loop(model, param, optimizer, X, Y, n_steps=50)

        assert all(math.isfinite(l) for l in losses), "Training diverged"
        assert losses[-1] < losses[0], "Loss should decrease"
        assert param._solver is solver, "Should still use our solver"
        print(f"  Custom solver: loss {losses[0]:.4f} -> {losses[-1]:.4f}")


class TestAllOptimizationsTraining:
    """Training with ALL optimizations enabled simultaneously."""

    def test_all_opts_train(self):
        torch.manual_seed(0)
        X = torch.randn(200, 32)
        Y = torch.randint(0, 10, (200,))

        model = SmallMLP()
        param = Parametrization(
            model, lr_prefactor=0.01,
            solve_interval=5,
            alignment_ema=0.7,
            c_ema=0.3,
            resample_w0=True,
            use_training_activations=True,
            warmup_steps=5,
        )
        optimizer = torch.optim.Adam(param.param_groups)
        param.capture_initial(X[:32])

        for _, pm in param._pms:
            if pm.weight is not None:
                assert pm._w0 is None

        assert len(param._persistent_hooks) > 0

        losses = train_loop(model, param, optimizer, X, Y, n_steps=100, use_hooks=True)

        assert all(math.isfinite(l) for l in losses), "Training diverged"
        assert losses[-1] < losses[0], "Loss should decrease"

        for name, pm in param._pms:
            if pm.weight is not None:
                assert math.isfinite(pm.alpha), f"{name}: alpha not finite"
                assert math.isfinite(pm.omega), f"{name}: omega not finite"
                assert math.isfinite(pm.u), f"{name}: u not finite"

        for g in param.param_groups:
            if g.get("maxp_managed"):
                assert g["lr"] > 0, f"LR <= 0 for {g['layer_name']}"

        print(f"  All opts combined: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
        managed = [g for g in param.param_groups if g.get('maxp_managed')]
        lr_strs = [f"{g['layer_name']}={g['lr']:.6f}" for g in managed]
        print(f"  Final LRs: {lr_strs}")

        param.remove_hooks()

    def test_all_opts_sgd(self):
        """Same but with SGD optimizer type."""
        torch.manual_seed(0)
        X = torch.randn(200, 32)
        Y = torch.randint(0, 10, (200,))

        model = SmallMLP()
        param = Parametrization(
            model, lr_prefactor=0.1,
            optimizer_type="sgd",
            solve_interval=5,
            alignment_ema=0.7,
            c_ema=0.3,
            resample_w0=True,
            use_training_activations=True,
            warmup_steps=5,
        )
        optimizer = torch.optim.SGD(param.param_groups, lr=0.1)
        param.capture_initial(X[:32])

        losses = train_loop(model, param, optimizer, X, Y, n_steps=100, use_hooks=True)

        assert all(math.isfinite(l) for l in losses), "Training diverged"
        print(f"  All opts SGD: loss {losses[0]:.4f} -> {losses[-1]:.4f}")

        param.remove_hooks()


# ---------------------------------------------------------------------------
# Benchmark: baseline vs optimized (time + memory)
# ---------------------------------------------------------------------------

class TestBenchmarkComparison:
    """Compare baseline vs fully optimized: time and memory."""

    def _run_benchmark(self, label, model_cls, param_kwargs,
                       use_hooks, X, Y, n_steps):
        """Train and measure time + peak memory. Returns (losses, time, peak_mb)."""
        torch.manual_seed(0)
        model = model_cls()
        param = Parametrization(model, lr_prefactor=0.01, **param_kwargs)
        optimizer = torch.optim.Adam(param.param_groups)
        param.capture_initial(X[:32])

        # Measure memory
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        t0 = time.perf_counter()
        losses = train_loop(model, param, optimizer, X, Y, n_steps, use_hooks)
        elapsed = time.perf_counter() - t0

        snapshot_after = tracemalloc.take_snapshot()
        # Get peak traced memory
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024

        if hasattr(param, 'remove_hooks'):
            param.remove_hooks()

        return losses, elapsed, peak_mb

    def test_benchmark_small(self):
        """Benchmark baseline vs optimized on SmallMLP (d=64)."""
        X = torch.randn(200, 32)
        Y = torch.randint(0, 10, (200,))
        n_steps = 50

        # Baseline: every-step solve, stored W0, separate forward pass
        losses_base, t_base, mem_base = self._run_benchmark(
            "baseline", SmallMLP, {}, False, X, Y, n_steps,
        )

        # Optimized: all features
        losses_opt, t_opt, mem_opt = self._run_benchmark(
            "optimized", SmallMLP,
            dict(solve_interval=10, alignment_ema=0.7, c_ema=0.3,
                 resample_w0=True, use_training_activations=True,
                 warmup_steps=5),
            True, X, Y, n_steps,
        )

        assert all(math.isfinite(l) for l in losses_base), "Baseline diverged"
        assert all(math.isfinite(l) for l in losses_opt), "Optimized diverged"
        assert losses_base[-1] < losses_base[0], "Baseline loss should decrease"
        assert losses_opt[-1] < losses_opt[0], "Optimized loss should decrease"

        speedup = t_base / t_opt if t_opt > 0 else float("inf")

        print(f"\n  === SmallMLP (d=64, {n_steps} steps) ===")
        print(f"  Baseline:  loss {losses_base[0]:.4f}->{losses_base[-1]:.4f}  "
              f"time={t_base:.3f}s  peak_mem={mem_base:.2f}MB")
        print(f"  Optimized: loss {losses_opt[0]:.4f}->{losses_opt[-1]:.4f}  "
              f"time={t_opt:.3f}s  peak_mem={mem_opt:.2f}MB")
        print(f"  Speedup: {speedup:.2f}x")

    def test_benchmark_wider(self):
        """Benchmark on WiderMLP (d=512) — memory savings more visible."""
        X = torch.randn(500, 128)
        Y = torch.randint(0, 10, (500,))
        n_steps = 30

        losses_base, t_base, mem_base = self._run_benchmark(
            "baseline", WiderMLP, {}, False, X, Y, n_steps,
        )

        losses_opt, t_opt, mem_opt = self._run_benchmark(
            "optimized", WiderMLP,
            dict(solve_interval=10, alignment_ema=0.7, c_ema=0.3,
                 resample_w0=True, use_training_activations=True,
                 warmup_steps=5),
            True, X, Y, n_steps,
        )

        assert all(math.isfinite(l) for l in losses_base), "Baseline diverged"
        assert all(math.isfinite(l) for l in losses_opt), "Optimized diverged"

        speedup = t_base / t_opt if t_opt > 0 else float("inf")

        print(f"\n  === WiderMLP (d=512, 6 layers, {n_steps} steps) ===")
        print(f"  Baseline:  loss {losses_base[0]:.4f}->{losses_base[-1]:.4f}  "
              f"time={t_base:.3f}s  peak_mem={mem_base:.2f}MB")
        print(f"  Optimized: loss {losses_opt[0]:.4f}->{losses_opt[-1]:.4f}  "
              f"time={t_opt:.3f}s  peak_mem={mem_opt:.2f}MB")
        print(f"  Speedup: {speedup:.2f}x")
