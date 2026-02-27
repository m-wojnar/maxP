#!/usr/bin/env python3
"""LR transfer demo: SP (Standard Parametrization) vs muP across widths.

For each width, sweeps over a range of learning rates, trains to completion,
and records the final loss. The resulting plot has LR on the x-axis and
final loss on the y-axis, with one curve per width.

muP curves should overlap (the optimal LR transfers across widths).
SP curves should shift apart (each width needs a different LR).

Usage:
    python examples_new/mlp_example/train.py
    python examples_new/mlp_example/train.py --widths 128 256 512
    python examples_new/mlp_example/train.py --steps 2000 --n-lrs 15
    python examples_new/mlp_example/train.py --no-plot
"""

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from mlp import MLP
from parametrized_mlp import make_parametrized_mlp


# ── Data ────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_cifar10(device: torch.device, root: str = "./data") -> tuple[torch.Tensor, torch.Tensor]:
    """Load CIFAR-10 train split as flattened, normalised tensors on device."""
    ds = datasets.CIFAR10(root=root, train=True, download=True)
    X = torch.tensor(ds.data, dtype=torch.float32) / 255.0
    X = X.permute(0, 3, 1, 2)
    X = transforms.functional.normalize(
        X, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616),
    )
    X = X.reshape(X.shape[0], -1).to(device)
    Y = torch.tensor(ds.targets, dtype=torch.long).to(device)
    return X, Y


def batch_iter(X: torch.Tensor, Y: torch.Tensor, batch_size: int):
    """Infinite random-batch iterator."""
    n = X.shape[0]
    while True:
        idx = torch.randint(0, n, (batch_size,), device=X.device)
        yield X[idx], Y[idx]


# ── Training ────────────────────────────────────────────────────────────

def train_sp(
    width: int,
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    lr: float,
    n_steps: int,
    n_layers: int,
    batch_size: int,
    seed: int,
) -> float:
    """Train a vanilla MLP (SP) and return the final train loss."""
    torch.manual_seed(seed)
    model = MLP(hidden_dim=width, n_layers=n_layers).to(X.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    it = batch_iter(X, Y, batch_size)
    last_loss = float("nan")
    for _ in range(n_steps):
        xb, yb = next(it)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        if not math.isfinite(loss.item()):
            return float("nan")
        last_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return last_loss


def train_mup(
    width: int,
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    lr_prefactor: float,
    n_steps: int,
    n_layers: int,
    batch_size: int,
    seed: int,
) -> float:
    """Train a parametrized MLP (muP) and return the final train loss."""
    torch.manual_seed(seed)
    model, param = make_parametrized_mlp(
        hidden_dim=width,
        n_layers=n_layers,
        lr_prefactor=lr_prefactor,
    )
    model = model.to(X.device)
    optimizer = torch.optim.Adam(param.param_groups)
    it = batch_iter(X, Y, batch_size)
    last_loss = float("nan")
    for _ in range(n_steps):
        xb, yb = next(it)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        if not math.isfinite(loss.item()):
            return float("nan")
        last_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return last_loss


# ── Plotting ────────────────────────────────────────────────────────────

COLORS = {
    128: "#1f77b4",
    256: "#ff7f0e",
    512: "#2ca02c",
    1024: "#d62728",
    2048: "#9467bd",
}


def plot_results(
    sp_results: dict[int, list[tuple[float, float]]],
    mup_results: dict[int, list[tuple[float, float]]],
    filename: str = "lr_transfer.png",
):
    """Two-panel plot: LR (x) vs final loss (y), one curve per width.

    Each results dict maps width -> list of (lr, final_loss) pairs.
    """
    import matplotlib.pyplot as plt

    fig, (ax_sp, ax_mup) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for width, pairs in sorted(sp_results.items()):
        lrs = [lr for lr, _ in pairs]
        losses = [loss for _, loss in pairs]
        color = COLORS.get(width, None)
        ax_sp.plot(lrs, losses, "o-", label=f"w={width}", color=color, markersize=4)
    ax_sp.set_title("SP (Standard Parametrization)")
    ax_sp.set_xlabel("Learning Rate")
    ax_sp.set_ylabel("Final Train Loss")
    ax_sp.set_xscale("log")
    ax_sp.set_yscale("log")
    ax_sp.legend()
    ax_sp.grid(True, alpha=0.3)

    for width, pairs in sorted(mup_results.items()):
        lrs = [lr for lr, _ in pairs]
        losses = [loss for _, loss in pairs]
        color = COLORS.get(width, None)
        ax_mup.plot(lrs, losses, "o-", label=f"w={width}", color=color, markersize=4)
    ax_mup.set_title("muP (Maximal Update Parametrization)")
    ax_mup.set_xlabel("Learning Rate (prefactor)")
    ax_mup.set_xscale("log")
    ax_mup.set_yscale("log")
    ax_mup.legend()
    ax_mup.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Plot saved to {filename}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LR transfer: SP vs muP")
    parser.add_argument("--widths", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sp-lr-min", type=float, default=1e-4)
    parser.add_argument("--sp-lr-max", type=float, default=1e-2)
    parser.add_argument("--mup-lr-min", type=float, default=1e-3)
    parser.add_argument("--mup-lr-max", type=float, default=1e-1)
    parser.add_argument("--n-lrs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output", type=str, default="lr_transfer.png")
    args = parser.parse_args()

    sp_lrs = np.logspace(np.log10(args.sp_lr_min), np.log10(args.sp_lr_max), args.n_lrs).tolist()
    mup_lrs = np.logspace(np.log10(args.mup_lr_min), np.log10(args.mup_lr_max), args.n_lrs).tolist()

    device = get_device()
    print(f"Using device: {device}")
    print("Loading CIFAR-10...")
    X, Y = load_cifar10(device)

    sp_results: dict[int, list[tuple[float, float]]] = {}
    mup_results: dict[int, list[tuple[float, float]]] = {}

    total_runs = len(args.widths) * (len(sp_lrs) + len(mup_lrs))
    run = 0

    for width in args.widths:
        sp_results[width] = []
        mup_results[width] = []

        for lr in sp_lrs:
            run += 1
            print(f"[{run}/{total_runs}] SP  w={width}  lr={lr:.2e} ...", end=" ", flush=True)
            sp_loss = train_sp(
                width, X, Y,
                lr=lr,
                n_steps=args.steps,
                n_layers=args.n_layers,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            sp_results[width].append((lr, sp_loss))
            print(f"loss={sp_loss:.4f}" if math.isfinite(sp_loss) else "diverged")

        for lr in mup_lrs:
            run += 1
            print(f"[{run}/{total_runs}] muP w={width}  lr={lr:.2e} ...", end=" ", flush=True)
            mup_loss = train_mup(
                width, X, Y,
                lr_prefactor=lr,
                n_steps=args.steps,
                n_layers=args.n_layers,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            mup_results[width].append((lr, mup_loss))
            print(f"loss={mup_loss:.4f}" if math.isfinite(mup_loss) else "diverged")

    if not args.no_plot:
        plot_results(sp_results, mup_results, filename=args.output)


if __name__ == "__main__":
    main()
