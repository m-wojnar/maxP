#!/usr/bin/env python3
"""muP vs Conservative on CIFAR-10 MLP — static alignment comparison.

Trains two variants of the same ParametrizedMLP architecture:

  1. Conservative — static "no" alignment (alpha=0, omega=0, u=0).
  2. muP — static "full" alignment (alpha=1, omega=0.5, u=1).

Usage:
    python examples_new/mlp_example/mup_vs_maxp.py
    python examples_new/mlp_example/mup_vs_maxp.py --width 256 --steps 1000
"""

import argparse
import math

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


# ── Training ────────────────────────────────────────────────────────────

def train(
    width, X, Y, *, lr_prefactor, n_steps, n_layers, batch_size, seed,
    alignment, desc="",
) -> list[float]:
    """Train with a static alignment preset. Returns per-step losses."""
    torch.manual_seed(seed)
    model = ParametrizedMLP(hidden_dim=width, n_layers=n_layers).to(X.device)
    param = Parametrization(model, lr_prefactor=lr_prefactor, alignment=alignment)
    optimizer = torch.optim.Adam(param.param_groups)

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
    return losses


# ── Plotting ────────────────────────────────────────────────────────────

def smooth(values, window=20):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid").tolist()


def plot_results(cons_losses, mup_losses, filename="mup_vs_maxp.png", window=20):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    cons_sm = smooth(cons_losses, window)
    mup_sm = smooth(mup_losses, window)
    offset = window // 2

    for ax in (ax1, ax2):
        ax.plot(cons_losses, alpha=0.15, color="#7f7f7f", linewidth=0.5)
        ax.plot(mup_losses, alpha=0.15, color="#1f77b4", linewidth=0.5)
        ax.plot(range(offset, offset + len(cons_sm)), cons_sm,
                color="#7f7f7f", linewidth=2, label='Conservative ("no")')
        ax.plot(range(offset, offset + len(mup_sm)), mup_sm,
                color="#1f77b4", linewidth=2, label='muP ("full")')
        ax.set_xlabel("Step")
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel("Train Loss")
    ax1.set_title("Loss Curves")
    ax2.set_ylabel("Train Loss (log)")
    ax2.set_yscale("log")
    ax2.set_title("Loss Curves (log scale)")

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Plot saved to {filename}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Conservative vs muP (static alignment) on CIFAR-10 MLP"
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-prefactor", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output", type=str, default="mup_vs_maxp.png")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Width: {args.width}, Layers: {args.n_layers}, Steps: {args.steps}")
    print(f"LR prefactor: {args.lr_prefactor}")
    print()

    print("Loading CIFAR-10...")
    X, Y = load_cifar10(device)

    kw = dict(
        width=args.width, X=X, Y=Y, lr_prefactor=args.lr_prefactor,
        n_steps=args.steps, n_layers=args.n_layers,
        batch_size=args.batch_size, seed=args.seed,
    )

    print("\n--- Conservative (static, no alignment) ---")
    cons_losses = train(**kw, alignment="no", desc="Conservative")
    print(f"Final loss: {cons_losses[-1]:.4f}")

    print("\n--- muP (static, full alignment) ---")
    mup_losses = train(**kw, alignment="full", desc="muP")
    print(f"Final loss: {mup_losses[-1]:.4f}")

    print(f"\n{'='*45}")
    print(f"  Conservative:  {cons_losses[-1]:.4f}")
    print(f"  muP (full):    {mup_losses[-1]:.4f}")
    print(f"{'='*45}")

    if not args.no_plot:
        plot_results(cons_losses, mup_losses, filename=args.output)


if __name__ == "__main__":
    main()
