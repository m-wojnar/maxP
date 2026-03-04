#!/usr/bin/env python3
"""LR transfer check: SP vs muP across widths on Shakespeare GPT.

For each method (SP, muP), trains at multiple (width, LR) combinations
and plots final loss vs LR with one curve per width.

Expected result:
  - muP: optimal LR is stable across widths (curves align horizontally).
  - SP:  optimal LR shifts left as width grows (no LR transfer).

Usage (quick test, CPU):
    python lr_transfer.py --steps 200 --lrs 0.01 0.1 --widths 64 128

Full run:
    python lr_transfer.py --steps 2000
"""

import argparse
import math
import os
import sys
import urllib.request
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
    return data, vocab_size


def batch_iter(data, seq_len, batch_size):
    n = data.shape[0] - seq_len - 1
    while True:
        idx = torch.randint(0, n, (batch_size,), device=data.device)
        x = torch.stack([data[i : i + seq_len] for i in idx])
        y = torch.stack([data[i + 1 : i + seq_len + 1] for i in idx])
        yield x, y


# ── SP (a,b) overrides ─────────────────────────────────────────────────

SP_AB = {
    "embedding": (0.0, 0.0),
    "hidden":    (0.0, 0.5),
    "readout":   (0.0, 0.5),
}


# ── Training ────────────────────────────────────────────────────────────

def train_run(
    d_model, n_heads, data, vocab_size, *,
    lr, n_steps, n_layers, seq_len, batch_size, seed,
    method, desc="",
) -> float:
    """Train one run, return final smoothed loss."""
    torch.manual_seed(seed)
    model = ParametrizedGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=4 * d_model,
        n_layers=n_layers,
        max_seq_len=seq_len,
    ).to(data.device)
    sample_input = torch.randint(0, vocab_size, (1, seq_len), device=data.device)

    param_kw = dict(
        lr_prefactor=lr,
        optimizer_type="adam",
        alignment="full",
        sample_input=sample_input,
    )
    if method == "SP":
        param_kw["ab_overrides"] = SP_AB

    param = Parametrization(model, **param_kw)
    optimizer = torch.optim.AdamW(param.param_groups, lr=lr)

    losses = []
    it = batch_iter(data, seq_len, batch_size)
    pbar = tqdm(range(n_steps), desc=desc, leave=False, ncols=90)
    for _ in pbar:
        xb, yb = next(it)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))
        if not math.isfinite(loss.item()):
            pbar.close()
            return float("inf")
        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pbar.close()

    # Smoothed final loss (last 50 steps)
    tail = losses[-50:]
    return sum(tail) / len(tail)


# ── Plotting ────────────────────────────────────────────────────────────

WIDTH_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def plot_transfer(
    sp_results: dict[int, dict[float, float]],
    mup_results: dict[int, dict[float, float]],
    widths: list[int],
    filename: str = "lr_transfer.png",
):
    import matplotlib.pyplot as plt

    fig, (ax_sp, ax_mup) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for i, w in enumerate(widths):
        color = WIDTH_COLORS[i % len(WIDTH_COLORS)]

        # SP
        lrs = sorted(sp_results[w].keys())
        losses = [sp_results[w][lr] for lr in lrs]
        ax_sp.plot(lrs, losses, "o-", color=color, label=f"w={w}", linewidth=1.8, markersize=5)

        # muP
        lrs = sorted(mup_results[w].keys())
        losses = [mup_results[w][lr] for lr in lrs]
        ax_mup.plot(lrs, losses, "o-", color=color, label=f"w={w}", linewidth=1.8, markersize=5)

    for ax, title in [(ax_sp, "SP"), (ax_mup, "muP")]:
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax_sp.set_ylabel("Final Loss (smoothed)")

    # Clip y-axis to useful range (ignore diverged runs)
    all_finite = []
    for results in [sp_results, mup_results]:
        for w_data in results.values():
            all_finite.extend(v for v in w_data.values() if math.isfinite(v) and v < 100)
    if all_finite:
        lo = min(all_finite) * 0.95
        hi = np.percentile(all_finite, 95) * 1.05
        for ax in (ax_sp, ax_mup):
            ax.set_ylim(lo, hi)

    fig.suptitle("LR Transfer: SP vs muP across widths — Shakespeare GPT", fontsize=13)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {filename}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────

DEFAULT_LRS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
DEFAULT_WIDTHS = [64, 128, 256, 512]


def main():
    parser = argparse.ArgumentParser(
        description="LR transfer check: SP vs muP across widths"
    )
    parser.add_argument("--widths", type=int, nargs="+", default=DEFAULT_WIDTHS)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output", type=str, default="lr_transfer.png")
    args = parser.parse_args()

    # Validate widths are divisible by n_heads
    for w in args.widths:
        if w % args.n_heads != 0:
            parser.error(f"Width {w} not divisible by n_heads={args.n_heads}")

    device = get_device()
    total_runs = len(args.widths) * len(args.lrs) * 2
    print(f"Device: {device}")
    print(f"Widths: {args.widths}")
    print(f"LRs: {args.lrs}")
    print(f"Steps: {args.steps}, Layers: {args.n_layers}, Seq len: {args.seq_len}")
    print(f"Total runs: {total_runs}")
    print()

    print("Loading Shakespeare...")
    data, vocab_size = load_shakespeare(device)
    print(f"  {len(data):,} chars, vocab size: {vocab_size}")

    common = dict(
        data=data, vocab_size=vocab_size,
        n_steps=args.steps, n_layers=args.n_layers,
        seq_len=args.seq_len, batch_size=args.batch_size, seed=args.seed,
    )

    sp_results: dict[int, dict[float, float]] = {}
    mup_results: dict[int, dict[float, float]] = {}
    run_idx = 0

    for w in args.widths:
        n_heads = args.n_heads
        sp_results[w] = {}
        mup_results[w] = {}

        for lr_val in args.lrs:
            for method, results in [("SP", sp_results), ("muP", mup_results)]:
                run_idx += 1
                desc = f"[{run_idx}/{total_runs}] {method} w={w} lr={lr_val}"
                print(f"\n{desc}")
                final_loss = train_run(
                    d_model=w, n_heads=n_heads, **common,
                    lr=lr_val, method=method, desc=desc,
                )
                results[w][lr_val] = final_loss
                tag = "DIV" if not math.isfinite(final_loss) else f"{final_loss:.4f}"
                print(f"  → final_loss={tag}")

    # Summary
    print(f"\n{'='*65}")
    print("  SP — optimal LR per width:")
    for w in args.widths:
        best_lr = min(sp_results[w], key=lambda lr: sp_results[w][lr])
        print(f"    w={w:<4d}  best_lr={best_lr:<8.4f}  loss={sp_results[w][best_lr]:.4f}")
    print()
    print("  muP — optimal LR per width:")
    for w in args.widths:
        best_lr = min(mup_results[w], key=lambda lr: mup_results[w][lr])
        print(f"    w={w:<4d}  best_lr={best_lr:<8.4f}  loss={mup_results[w][best_lr]:.4f}")
    print(f"{'='*65}")

    if not args.no_plot:
        plot_transfer(sp_results, mup_results, args.widths, filename=args.output)


if __name__ == "__main__":
    main()
