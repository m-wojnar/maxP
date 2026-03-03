#!/usr/bin/env python3
"""LR sweep: SP vs muP vs maxP dynamic (AdamW) on Shakespeare GPT.

Sweeps lr for each method, picks the best LR per method
(lowest final smoothed loss), then produces a comparison plot.

Also records per-layer alignment and LR history for the best maxP run.

Usage (CPU, quick test):
    python sweep.py --steps 200 --lrs 0.001 0.01

Full run:
    python sweep.py --d-model 128 --n-layers 4 --steps 2000 --seed 42
"""

import argparse
import json
import math
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.nanogpt_example.parametrized_gpt import ParametrizedGPT
from maxp import Parametrization


# ── Data ────────────────────────────────────────────────────────────────

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_shakespeare(device: torch.device, data_dir: str = "./data"):
    """Load Shakespeare text, build char vocab, encode to tensor."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.exists(path):
        print(f"Downloading Shakespeare to {path}...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)

    with open(path, "r") as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}

    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long, device=device)
    return data, vocab_size, chars


def batch_iter(data, seq_len, batch_size):
    """Yield random subsequences from the encoded text."""
    n = data.shape[0] - seq_len - 1
    while True:
        idx = torch.randint(0, n, (batch_size,), device=data.device)
        x = torch.stack([data[i : i + seq_len] for i in idx])
        y = torch.stack([data[i + 1 : i + seq_len + 1] for i in idx])
        yield x, y


# ── SP (a,b) overrides ─────────────────────────────────────────────────
# SP: a=0 for all layers, b=0 for embedding, b=0.5 for hidden/readout.
SP_AB = {
    "embedding": (0.0, 0.0),
    "hidden":    (0.0, 0.5),
    "readout":   (0.0, 0.5),
}


# ── Run result ──────────────────────────────────────────────────────────

@dataclass
class RunResult:
    method: str
    lr: float
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

def _make_model(d_model, n_heads, n_layers, vocab_size, seq_len, device):
    model = ParametrizedGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=4 * d_model,
        n_layers=n_layers,
        max_seq_len=seq_len,
    ).to(device)
    sample_input = torch.randint(0, vocab_size, (1, seq_len), device=device)
    return model, sample_input


def train_sp(
    d_model, n_heads, data, vocab_size, *,
    lr, n_steps, n_layers, seq_len, batch_size, seed,
    alignment="full", desc="",
) -> RunResult:
    """SP baseline: ParametrizedGPT with SP (a,b) + AdamW + static alignment."""
    torch.manual_seed(seed)
    model, sample_input = _make_model(d_model, n_heads, n_layers, vocab_size, seq_len, data.device)
    param = Parametrization(
        model,
        lr_prefactor=lr,
        optimizer_type="adam",
        alignment=alignment,
        ab_overrides=SP_AB,
        sample_input=sample_input,
    )
    optimizer = torch.optim.AdamW(param.param_groups, lr=lr)

    losses = []
    it = batch_iter(data, seq_len, batch_size)
    pbar = tqdm(range(n_steps), desc=desc, leave=False, ncols=90)
    for _ in pbar:
        xb, yb = next(it)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))
        if not math.isfinite(loss.item()):
            losses.append(float("nan"))
            break
        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pbar.close()
    return RunResult(method="SP", lr=lr, losses=losses)


def train_mup(
    d_model, n_heads, data, vocab_size, *,
    lr, n_steps, n_layers, seq_len, batch_size, seed,
    alignment="full", desc="",
) -> RunResult:
    """muP baseline: ParametrizedGPT with default muP (a,b) + AdamW + static alignment."""
    torch.manual_seed(seed)
    model, sample_input = _make_model(d_model, n_heads, n_layers, vocab_size, seq_len, data.device)
    param = Parametrization(
        model,
        lr_prefactor=lr,
        optimizer_type="adam",
        alignment=alignment,
        sample_input=sample_input,
    )
    optimizer = torch.optim.AdamW(param.param_groups, lr=lr)

    losses = []
    it = batch_iter(data, seq_len, batch_size)
    pbar = tqdm(range(n_steps), desc=desc, leave=False, ncols=90)
    for _ in pbar:
        xb, yb = next(it)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))
        if not math.isfinite(loss.item()):
            losses.append(float("nan"))
            break
        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pbar.close()
    return RunResult(method="muP", lr=lr, losses=losses)


def train_maxp(
    d_model, n_heads, data, vocab_size, *,
    lr, n_steps, n_layers, seq_len, batch_size, seed,
    warmup_steps, solve_interval, sample_size, desc="",
) -> RunResult:
    """maxP dynamic: ParametrizedGPT + AdamW + dynamic alignment solving."""
    torch.manual_seed(seed)
    model, sample_input = _make_model(d_model, n_heads, n_layers, vocab_size, seq_len, data.device)
    param = Parametrization(
        model,
        lr_prefactor=lr,
        optimizer_type="adam",
        alignment="full",
        warmup_steps=warmup_steps,
        solve_interval=solve_interval,
        sample_size=sample_size,
        sample_input=sample_input,
    )
    optimizer = torch.optim.AdamW(param.param_groups, lr=lr)

    # Build sample input for alignment measurement
    sample_x = torch.stack([
        data[i : i + seq_len]
        for i in torch.randint(0, data.shape[0] - seq_len - 1, (sample_size,))
    ])
    param.capture_initial(sample_x)

    # Track per-layer alignment and LR
    layer_history: dict[str, list[dict]] = {}
    for name, pm in param._pms:
        if pm.weight is not None:
            layer_history[name] = []

    losses = []
    it = batch_iter(data, seq_len, batch_size)
    pbar = tqdm(range(n_steps), desc=desc, leave=False, ncols=90)
    for _ in pbar:
        xb, yb = next(it)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))
        if not math.isfinite(loss.item()):
            losses.append(float("nan"))
            break
        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        param.step(sample_x, optimizer)

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
        method="maxP", lr=lr,
        losses=losses, layer_history=layer_history,
    )


# ── Sweep ───────────────────────────────────────────────────────────────

def sweep(
    lrs: list[float],
    d_model: int,
    n_heads: int,
    data,
    vocab_size: int,
    *,
    n_steps: int,
    n_layers: int,
    seq_len: int,
    batch_size: int,
    seed: int,
    warmup_steps: int,
    solve_interval: int,
    sample_size: int,
) -> dict[str, list[RunResult]]:
    """Run all methods at each LR. Returns {method: [RunResult, ...]}."""
    results: dict[str, list[RunResult]] = {
        "SP": [],
        "muP": [],
        "maxP": [],
    }

    common = dict(
        d_model=d_model, n_heads=n_heads, data=data, vocab_size=vocab_size,
        n_steps=n_steps, n_layers=n_layers, seq_len=seq_len,
        batch_size=batch_size, seed=seed,
    )

    total = len(lrs) * 3
    run_idx = 0

    def _report(run: RunResult):
        tag = "DIV" if run.diverged else f"{run.final_loss:.4f}"
        print(f"  → final_loss={tag}")

    for lr_val in lrs:
        run_idx += 1
        print(f"\n[{run_idx}/{total}] SP    lr={lr_val}")
        results["SP"].append(train_sp(
            **common, lr=lr_val, desc=f"SP lr={lr_val}",
        ))
        _report(results["SP"][-1])

        run_idx += 1
        print(f"[{run_idx}/{total}] muP   lr={lr_val}")
        results["muP"].append(train_mup(
            **common, lr=lr_val, desc=f"muP lr={lr_val}",
        ))
        _report(results["muP"][-1])

        run_idx += 1
        print(f"[{run_idx}/{total}] maxP  lr={lr_val}")
        results["maxP"].append(train_maxp(
            **common, lr=lr_val,
            warmup_steps=warmup_steps, solve_interval=solve_interval,
            sample_size=sample_size, desc=f"maxP lr={lr_val}",
        ))
        _report(results["maxP"][-1])

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
    return name.replace(".inner", "").replace("blocks.", "b.")


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
        "SP": {"color": "#7f7f7f", "ls": "-"},
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
            label=f'{method} (lr={run.lr})',
        )
        ax_log.plot(run.losses, alpha=0.10, color=st["color"], linewidth=0.5)
        ax_log.plot(
            range(offset, offset + len(sm)), sm,
            color=st["color"], linewidth=2.2, linestyle=st["ls"],
            label=f'{method} (lr={run.lr})',
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
    for method in ["SP", "muP", "maxP"]:
        for run in results[method]:
            rows.append([
                method,
                f"{run.lr:.4f}",
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

    fig.suptitle("LR Sweep: SP vs muP vs maxP (AdamW) — Shakespeare GPT", fontsize=14)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {filename}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────

DEFAULT_LRS = [0.03, 0.1, 0.3]


def main():
    parser = argparse.ArgumentParser(
        description="LR sweep: SP vs muP vs maxP (AdamW) on Shakespeare GPT"
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS,
                        help="LR values to sweep")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--solve-interval", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output", type=str, default="sweep.png")
    parser.add_argument("--save-json", type=str, default=None,
                        help="Save sweep results to JSON (losses + best LRs)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"d_model: {args.d_model}, Heads: {args.n_heads}, "
          f"Layers: {args.n_layers}, Seq len: {args.seq_len}")
    print(f"Steps: {args.steps}, Batch size: {args.batch_size}")
    print(f"LR sweep: {args.lrs}")
    print(f"maxP: warmup={args.warmup_steps}, interval={args.solve_interval}, "
          f"samples={args.sample_size}")
    print(f"Total runs: {len(args.lrs) * 3}")
    print()

    print("Loading Shakespeare...")
    data, vocab_size, chars = load_shakespeare(device)
    print(f"  {len(data):,} chars, vocab size: {vocab_size}")

    results = sweep(
        lrs=args.lrs,
        d_model=args.d_model,
        n_heads=args.n_heads,
        data=data,
        vocab_size=vocab_size,
        n_steps=args.steps,
        n_layers=args.n_layers,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        solve_interval=args.solve_interval,
        sample_size=args.sample_size,
    )

    best = {method: pick_best(runs) for method, runs in results.items()}

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Best of sweep (d_model={args.d_model}, layers={args.n_layers}, "
          f"steps={args.steps}):")
    for method, run in best.items():
        tag = "DIV" if run.diverged else f"{run.final_loss:.4f}"
        print(f"    {method:8s}  lr={run.lr:<8.4f}  loss={tag}")
    print(f"{'='*60}")

    if not args.no_plot:
        plot_sweep(results, best, filename=args.output)

    if args.save_json:
        json_data = {}
        for method, runs in results.items():
            json_data[method] = [
                {"lr": r.lr, "final_loss": r.final_loss, "diverged": r.diverged}
                for r in runs
            ]
        json_data["best"] = {m: r.lr for m, r in best.items()}
        with open(args.save_json, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"Results saved to {args.save_json}")


if __name__ == "__main__":
    main()
