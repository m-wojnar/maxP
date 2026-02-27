#!/usr/bin/env python3
"""LR transfer demo for ViT: SP vs muP across widths.

Sweeps learning rates at several embed_dim widths. The plot has LR on the
x-axis and final train loss on the y-axis, with one curve per width.

muP curves should overlap (LR transfers). SP curves should shift.

Usage:
    python examples_new/vit_example/train.py
    python examples_new/vit_example/train.py --widths 64 128 256
    python examples_new/vit_example/train.py --steps 1000 --n-lrs 7
    python examples_new/vit_example/train.py --no-plot
"""

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from vit import ViT
from parametrized_vit import make_parametrized_vit


# ── Data ────────────────────────────────────────────────────────────────

def load_cifar10(root: str = "./data") -> tuple[torch.Tensor, torch.Tensor]:
    """Load CIFAR-10 train split as normalised image tensors."""
    ds = datasets.CIFAR10(root=root, train=True, download=True)
    X = torch.tensor(ds.data, dtype=torch.float32) / 255.0
    X = X.permute(0, 3, 1, 2)  # (50000, 3, 32, 32)
    X = transforms.functional.normalize(
        X, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616),
    )
    Y = torch.tensor(ds.targets, dtype=torch.long)
    return X, Y


def batch_iter(X: torch.Tensor, Y: torch.Tensor, batch_size: int):
    n = X.shape[0]
    while True:
        idx = torch.randint(0, n, (batch_size,))
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
    mlp_ratio: float,
    batch_size: int,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    model = ViT(embed_dim=width, n_layers=n_layers, mlp_ratio=mlp_ratio)
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
    mlp_ratio: float,
    batch_size: int,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    model, param = make_parametrized_vit(
        embed_dim=width,
        n_layers=n_layers,
        mlp_ratio=mlp_ratio,
        lr_prefactor=lr_prefactor,
    )
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
    64: "#1f77b4",
    128: "#ff7f0e",
    256: "#2ca02c",
    512: "#d62728",
    1024: "#9467bd",
}


def plot_results(
    sp_results: dict[int, list[tuple[float, float]]],
    mup_results: dict[int, list[tuple[float, float]]],
    filename: str = "lr_transfer_vit.png",
):
    import matplotlib.pyplot as plt

    fig, (ax_sp, ax_mup) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for width, pairs in sorted(sp_results.items()):
        lrs = [lr for lr, _ in pairs]
        losses = [loss for _, loss in pairs]
        color = COLORS.get(width, None)
        ax_sp.plot(lrs, losses, "o-", label=f"d={width}", color=color, markersize=4)
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
        ax_mup.plot(lrs, losses, "o-", label=f"d={width}", color=color, markersize=4)
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
    parser = argparse.ArgumentParser(description="LR transfer: SP vs muP (ViT)")
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sp-lr-min", type=float, default=1e-4)
    parser.add_argument("--sp-lr-max", type=float, default=1e-2)
    parser.add_argument("--mup-lr-min", type=float, default=1e-3)
    parser.add_argument("--mup-lr-max", type=float, default=1e-1)
    parser.add_argument("--n-lrs", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output", type=str, default="lr_transfer_vit.png")
    args = parser.parse_args()

    sp_lrs = np.logspace(np.log10(args.sp_lr_min), np.log10(args.sp_lr_max), args.n_lrs).tolist()
    mup_lrs = np.logspace(np.log10(args.mup_lr_min), np.log10(args.mup_lr_max), args.n_lrs).tolist()

    print("Loading CIFAR-10...")
    X, Y = load_cifar10()

    sp_results: dict[int, list[tuple[float, float]]] = {}
    mup_results: dict[int, list[tuple[float, float]]] = {}

    total_runs = len(args.widths) * (len(sp_lrs) + len(mup_lrs))
    run = 0

    for width in args.widths:
        sp_results[width] = []
        mup_results[width] = []

        for lr in sp_lrs:
            run += 1
            print(f"[{run}/{total_runs}] SP  d={width}  lr={lr:.2e} ...", end=" ", flush=True)
            loss = train_sp(
                width, X, Y,
                lr=lr,
                n_steps=args.steps,
                n_layers=args.n_layers,
                mlp_ratio=args.mlp_ratio,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            sp_results[width].append((lr, loss))
            print(f"loss={loss:.4f}" if math.isfinite(loss) else "diverged")

        for lr in mup_lrs:
            run += 1
            print(f"[{run}/{total_runs}] muP d={width}  lr={lr:.2e} ...", end=" ", flush=True)
            loss = train_mup(
                width, X, Y,
                lr_prefactor=lr,
                n_steps=args.steps,
                n_layers=args.n_layers,
                mlp_ratio=args.mlp_ratio,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            mup_results[width].append((lr, loss))
            print(f"loss={loss:.4f}" if math.isfinite(loss) else "diverged")

    if not args.no_plot:
        plot_results(sp_results, mup_results, filename=args.output)


if __name__ == "__main__":
    main()
