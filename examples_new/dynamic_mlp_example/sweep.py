#!/usr/bin/env python3
"""LR sweep: Conservative vs muP vs maxP (dynamic) on CIFAR-10 MLP.

Sweeps lr_prefactor for each method, picks the best LR per method
(lowest final smoothed loss), then produces a comparison plot.

Also records per-layer alignment and LR history for the best maxP run.

Usage (CPU, quick test):
    python sweep.py --steps 500 --lrs 0.001 0.003 0.01

GPU cluster (full sweep):
    python sweep.py --width 512 --steps 5000 --seed 42
    python sweep.py --width 1024 --steps 5000 --seed 42

Custom LR grid:
    python sweep.py --lrs 0.001 0.003 0.01 0.03 0.1
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

from parametrized_mlp import ParametrizedMLP
from maxp_new import Parametrization


# ── Data ────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_cifar10(device: torch.device, root: str = "./data"):
    ds = datasets.CIFAR10(root=root, train=True, download=True)
    X = torch.tensor(ds.data, dtype=torch.float32) / 255.0
    X = X.permute(0, 3, 1, 2)
    X = transforms.functional.normalize(
        X, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616),
    )
    X = X.reshape(X.shape[0], -1).to(device)
    Y = torch.tensor(ds.targets, dtype=torch.long).to(device)
    return X, Y


def batch_iter(X, Y, batch_size):
    n = X.shape[0]
    while True:
        idx = torch.randint(0, n, (batch_size,), device=X.device)
        yield X[idx], Y[idx]


# ── Run result ──────────────────────────────────────────────────────────

@dataclass
class RunResult:
    method: str
    lr_prefactor: float
    losses: list[float]
    layer_history: dict[str, list[dict]] = field(default_factory=dict)

    @property
    def final_loss(self) -> float:
        """Smoothed final loss (last 50-step average, robust to noise)."""
        tail = [v for v in self.losses[-50:] if math.isfinite(v)]
        return sum(tail) / len(tail) if tail else float("inf")

    @property
    def diverged(self) -> bool:
        return any(not math.isfinite(v) for v in self.losses)


# ── Training ────────────────────────────────────────────────────────────

def train_static(
    width, X, Y, *, lr_prefactor, n_steps, n_layers, batch_size, seed,
    alignment, desc="",
) -> RunResult:
    torch.manual_seed(seed)
    model = ParametrizedMLP(hidden_dim=width, n_layers=n_layers).to(X.device)
    param = Parametrization(model, lr_prefactor=lr_prefactor, alignment=alignment)
    optimizer = torch.optim.Adam(param.param_groups)

    method = "muP" if alignment == "full" else "Conservative"
    losses = []
    it = batch_iter(X, Y, batch_size)
    pbar = tqdm(range(n_steps), desc=desc, leave=False, ncols=90)
    for _ in pbar:
        xb, yb = next(it)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        if not math.isfinite(loss.item()):
            losses.append(float("nan"))
            break
        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pbar.close()
    return RunResult(method=method, lr_prefactor=lr_prefactor, losses=losses)


def train_maxp(
    width, X, Y, *, lr_prefactor, n_steps, n_layers, batch_size, seed,
    warmup_steps, solve_interval, sample_size, desc="",
) -> RunResult:
    torch.manual_seed(seed)
    model = ParametrizedMLP(hidden_dim=width, n_layers=n_layers).to(X.device)
    param = Parametrization(
        model,
        lr_prefactor=lr_prefactor,
        alignment="full",
        warmup_steps=warmup_steps,
        solve_interval=solve_interval,
        sample_size=sample_size,
    )
    optimizer = torch.optim.Adam(param.param_groups)

    sample_X = X[:sample_size]
    param.capture_initial(sample_X)

    # Track per-layer alignment and LR
    layer_history: dict[str, list[dict]] = {}
    for name, pm in param._pms:
        if pm.weight is not None:
            layer_history[name] = []

    losses = []
    it = batch_iter(X, Y, batch_size)
    pbar = tqdm(range(n_steps), desc=desc, leave=False, ncols=90)
    for _ in pbar:
        xb, yb = next(it)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        if not math.isfinite(loss.item()):
            losses.append(float("nan"))
            break
        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        param.step(sample_X, optimizer)

        for name, pm in param._pms:
            if pm.weight is not None:
                layer_history[name].append({
                    "alpha": pm.alpha,
                    "omega": pm.omega,
                    "u": pm.u,
                    "lr": next(
                        g["lr"] for g in param.param_groups
                        if g.get("layer_name") == name
                    ),
                })
    pbar.close()
    return RunResult(
        method="maxP", lr_prefactor=lr_prefactor,
        losses=losses, layer_history=layer_history,
    )


# ── Sweep ───────────────────────────────────────────────────────────────

def sweep(
    lrs: list[float],
    width: int,
    X, Y,
    *,
    n_steps: int,
    n_layers: int,
    batch_size: int,
    seed: int,
    warmup_steps: int,
    solve_interval: int,
    sample_size: int,
) -> dict[str, list[RunResult]]:
    """Run all methods at each LR. Returns {method: [RunResult, ...]}."""
    results: dict[str, list[RunResult]] = {
        "Conservative": [],
        "muP": [],
        "maxP": [],
    }

    total = len(lrs) * 3
    run_idx = 0

    for lr in lrs:
        run_idx += 1
        print(f"\n[{run_idx}/{total}] Conservative  lr={lr}")
        results["Conservative"].append(train_static(
            width, X, Y, lr_prefactor=lr, n_steps=n_steps,
            n_layers=n_layers, batch_size=batch_size, seed=seed,
            alignment="no", desc=f"Cons lr={lr}",
        ))

        run_idx += 1
        print(f"[{run_idx}/{total}] muP           lr={lr}")
        results["muP"].append(train_static(
            width, X, Y, lr_prefactor=lr, n_steps=n_steps,
            n_layers=n_layers, batch_size=batch_size, seed=seed,
            alignment="full", desc=f"muP lr={lr}",
        ))

        run_idx += 1
        print(f"[{run_idx}/{total}] maxP          lr={lr}")
        results["maxP"].append(train_maxp(
            width, X, Y, lr_prefactor=lr, n_steps=n_steps,
            n_layers=n_layers, batch_size=batch_size, seed=seed,
            warmup_steps=warmup_steps, solve_interval=solve_interval,
            sample_size=sample_size, desc=f"maxP lr={lr}",
        ))

    return results


def pick_best(runs: list[RunResult]) -> RunResult:
    """Pick the run with the lowest final smoothed loss (non-diverged)."""
    valid = [r for r in runs if not r.diverged]
    if not valid:
        return min(runs, key=lambda r: r.final_loss)
    return min(valid, key=lambda r: r.final_loss)


# ── Plotting ────────────────────────────────────────────────────────────

def smooth(values, window=20):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid").tolist()


def _short_name(name: str) -> str:
    return name.replace(".inner", "").replace("hidden_layers.", "hidden.")


_LAYER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def plot_sweep(
    results: dict[str, list[RunResult]],
    best: dict[str, RunResult],
    filename: str = "sweep.png",
    window: int = 50,
):
    import matplotlib.pyplot as plt

    maxp_best = best["maxP"]
    has_hist = bool(maxp_best.layer_history)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.35)

    ax_loss = fig.add_subplot(gs[0, 0:2])
    ax_log = fig.add_subplot(gs[0, 2])
    ax_lr = fig.add_subplot(gs[0, 3])
    ax_alpha = fig.add_subplot(gs[1, 0])
    ax_omega = fig.add_subplot(gs[1, 1])
    ax_u = fig.add_subplot(gs[1, 2])
    ax_table = fig.add_subplot(gs[1, 3])

    method_style = {
        "Conservative": {"color": "#7f7f7f", "ls": "-"},
        "muP": {"color": "#1f77b4", "ls": "-"},
        "maxP": {"color": "#d62728", "ls": "--"},
    }

    # ── Best-of loss curves ──
    offset = window // 2
    for method, run in best.items():
        st = method_style[method]
        sm = smooth(run.losses, window)
        ax_loss.plot(run.losses, alpha=0.10, color=st["color"], linewidth=0.5)
        ax_loss.plot(
            range(offset, offset + len(sm)), sm,
            color=st["color"], linewidth=2.2, linestyle=st["ls"],
            label=f'{method} (lr={run.lr_prefactor})',
        )
        ax_log.plot(run.losses, alpha=0.10, color=st["color"], linewidth=0.5)
        ax_log.plot(
            range(offset, offset + len(sm)), sm,
            color=st["color"], linewidth=2.2, linestyle=st["ls"],
            label=f'{method} (lr={run.lr_prefactor})',
        )

    # Zoom linear loss
    all_losses = sum((r.losses for r in best.values()), [])
    finite = [v for v in all_losses if math.isfinite(v)]
    if finite:
        lo, hi = np.percentile(finite, 1), np.percentile(finite, 95)
        pad = 0.10 * (hi - lo)
        ax_loss.set_ylim(max(0, lo - pad), hi + pad)

    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Train Loss")
    ax_loss.set_title("Best-of-sweep Loss Curves")
    ax_loss.legend(fontsize="small")
    ax_loss.grid(True, alpha=0.3)

    ax_log.set_xlabel("Step")
    ax_log.set_ylabel("Train Loss (log)")
    ax_log.set_yscale("log")
    ax_log.set_title("Log scale")
    ax_log.legend(fontsize="x-small")
    ax_log.grid(True, alpha=0.3)

    # ── Per-layer alignment + LR for best maxP ──
    if has_hist:
        hist = maxp_best.layer_history
        names = list(hist.keys())
        colors = {n: _LAYER_COLORS[i % len(_LAYER_COLORS)] for i, n in enumerate(names)}

        for name, history in hist.items():
            steps = range(len(history))
            short = _short_name(name)
            c = colors[name]
            ax_alpha.plot(steps, [h["alpha"] for h in history], color=c, lw=1.2, label=short)
            ax_omega.plot(steps, [h["omega"] for h in history], color=c, lw=1.2, label=short)
            ax_u.plot(steps, [h["u"] for h in history], color=c, lw=1.2, label=short)
            ax_lr.plot(steps, [h["lr"] for h in history], color=c, lw=1.2, label=short)

        for ax, ref, lbl in [
            (ax_alpha, 1.0, "full=1.0"), (ax_omega, 0.5, "full=0.5"), (ax_u, 1.0, "full=1.0"),
        ]:
            ax.axhline(ref, color="k", ls=":", lw=0.8, alpha=0.5, label=lbl)

        for ax, key in [(ax_alpha, "alpha"), (ax_omega, "omega"), (ax_u, "u"), (ax_lr, "lr")]:
            all_vals = []
            for history in hist.values():
                all_vals.extend(h[key] for h in history if math.isfinite(h[key]))
            if all_vals:
                lo, hi = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
                pad = 0.15 * max(hi - lo, 1e-8)
                ax.set_ylim(lo - pad, hi + pad)

    for ax, title, ylabel in [
        (ax_alpha, r"$\alpha$ (z₀ @ $\Delta$w)", r"$\alpha$"),
        (ax_omega, r"$\omega$ ($\Delta$z @ w₀)", r"$\omega$"),
        (ax_u, r"$u$ ($\Delta$z @ $\Delta$w)", r"$u$"),
        (ax_lr, "Per-layer LR (maxP best)", "LR"),
    ]:
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize="x-small", ncol=2)
        ax.grid(True, alpha=0.3)

    # ── Summary table ──
    ax_table.axis("off")
    rows = []
    for method in ["Conservative", "muP", "maxP"]:
        for run in results[method]:
            rows.append([
                method,
                f"{run.lr_prefactor:.4f}",
                f"{run.final_loss:.4f}" if not run.diverged else "DIV",
                "*" if run is best[method] else "",
            ])
    table = ax_table.table(
        cellText=rows,
        colLabels=["Method", "LR", "Final Loss", "Best"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    ax_table.set_title("All runs", fontsize=10)

    fig.suptitle("LR Sweep: Conservative vs muP vs maxP — CIFAR-10 MLP", fontsize=14)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {filename}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────

DEFAULT_LRS = [0.001, 0.003, 0.01, 0.03, 0.1]


def main():
    parser = argparse.ArgumentParser(
        description="LR sweep: Conservative vs muP vs maxP on CIFAR-10 MLP"
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS,
                        help="LR prefactors to sweep")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--solve-interval", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output", type=str, default="sweep.png")
    parser.add_argument("--save-json", type=str, default=None,
                        help="Save sweep results to JSON (losses + best LRs)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Width: {args.width}, Layers: {args.n_layers}, Steps: {args.steps}")
    print(f"LR sweep: {args.lrs}")
    print(f"maxP: warmup={args.warmup_steps}, interval={args.solve_interval}, "
          f"samples={args.sample_size}")
    print(f"Total runs: {len(args.lrs) * 3}")
    print()

    print("Loading CIFAR-10...")
    X, Y = load_cifar10(device)

    results = sweep(
        lrs=args.lrs,
        width=args.width,
        X=X, Y=Y,
        n_steps=args.steps,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        solve_interval=args.solve_interval,
        sample_size=args.sample_size,
    )

    best = {method: pick_best(runs) for method, runs in results.items()}

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Best of sweep (width={args.width}, steps={args.steps}):")
    for method, run in best.items():
        tag = "DIV" if run.diverged else f"{run.final_loss:.4f}"
        print(f"    {method:15s}  lr={run.lr_prefactor:<8.4f}  loss={tag}")
    print(f"{'='*60}")

    if not args.no_plot:
        plot_sweep(results, best, filename=args.output)

    if args.save_json:
        data = {}
        for method, runs in results.items():
            data[method] = [
                {"lr": r.lr_prefactor, "final_loss": r.final_loss, "diverged": r.diverged}
                for r in runs
            ]
        data["best"] = {m: r.lr_prefactor for m, r in best.items()}
        with open(args.save_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {args.save_json}")


if __name__ == "__main__":
    main()
