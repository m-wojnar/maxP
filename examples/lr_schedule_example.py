"""Dynamic maxP LR visualization.

Shows how maxP dynamically adjusts per-layer learning rates based on
measured alignment during training.  Each layer's LR evolves independently
as the LP solver re-computes c exponents from (alpha, omega, u) metrics.

A standard scheduler envelope (cosine, wsd) can be composed on top by
scaling lr_prefactor — the per-layer *ratios* still evolve dynamically.

Usage:
    python examples/lr_schedule_example.py [cosine|constant|wsd]
"""

import math
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from maxp import ParametrizedModule, Parametrization

# -- Config ------------------------------------------------------------------

TOTAL_STEPS = 1000
BATCH_SIZE = 64
SAMPLE_SIZE = 64
INPUT_DIM = 128
HIDDEN_DIM = 64
N_CLASSES = 50
BASE_LR = 0.05
WARMUP_STEPS = 30


# -- Data --------------------------------------------------------------------


def make_data(n=10000, seed=0):
    """Synthetic clustered classification data."""
    torch.manual_seed(seed)
    centers = torch.randn(N_CLASSES, INPUT_DIM)
    labels = torch.randint(0, N_CLASSES, (n,))
    X = centers[labels] + 3.0 * torch.randn(n, INPUT_DIM)
    return X, labels


# -- Model -------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = ParametrizedModule(
            nn.Linear(INPUT_DIM, HIDDEN_DIM, bias=False),
            width_dim=HIDDEN_DIM,
            layer_type="embedding",
        )
        self.hidden = ParametrizedModule(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False),
            width_dim=HIDDEN_DIM,
            layer_type="hidden",
        )
        self.readout = ParametrizedModule(
            nn.Linear(HIDDEN_DIM, N_CLASSES, bias=False),
            width_dim=HIDDEN_DIM,
            layer_type="readout",
        )

    def forward(self, x):
        x = self.embed(x).relu()
        x = self.hidden(x).relu()
        return self.readout(x)


# -- Schedule envelope -------------------------------------------------------


def schedule_factor(step: int, total: int, name: str) -> float:
    """Multiplicative factor for lr_prefactor at a given step."""
    if name == "constant":
        return 1.0
    if name == "cosine":
        return 0.5 * (1 + math.cos(math.pi * step / total))
    if name == "wsd":
        warmup = int(0.1 * total)
        decay_start = int(0.8 * total)
        if step < warmup:
            return (step + 1) / warmup
        if step < decay_start:
            return 1.0
        t = (step - decay_start) / (total - decay_start)
        return 0.5 * (1 + math.cos(math.pi * t))
    if name == "early-stop":
        cutoff = int(0.4 * total)
        if step >= cutoff:
            return 0.0
        return 0.5 * (1 + math.cos(math.pi * step / cutoff))
    raise ValueError(f"Unknown schedule: {name!r}")


# -- Main --------------------------------------------------------------------


def main(schedule_name: str = "cosine"):
    torch.manual_seed(42)
    X, Y = make_data()

    model = MLP()
    param = Parametrization(
        model,
        optimizer_type="sgd",
        lr_prefactor=BASE_LR,
        warmup_steps=WARMUP_STEPS,
        solve_interval=1,
        sample_size=SAMPLE_SIZE,
    )

    optimizer = torch.optim.SGD(param.param_groups)
    sample_X = X[:SAMPLE_SIZE]
    param.capture_initial(sample_X)

    # Identify managed layers
    managed = [g for g in param.param_groups if g.get("maxp_managed")]
    layer_names = [g["layer_name"] for g in managed]

    lr_hist = {n: [] for n in layer_names}
    loss_hist = []

    for step in range(TOTAL_STEPS):
        # Apply schedule envelope to lr_prefactor
        param.lr_prefactor = BASE_LR * schedule_factor(step, TOTAL_STEPS, schedule_name)

        # Forward / backward
        idx = torch.randint(0, len(X), (BATCH_SIZE,))
        logits = model(X[idx])
        loss = F.cross_entropy(logits, Y[idx])
        loss_hist.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Dynamic alignment re-solve → updates per-layer LRs independently
        param.step(sample_X, optimizer)

        # Record
        for name, pm in param._pms:
            if pm.weight is None:
                continue
            g = next(g for g in param.param_groups if g.get("layer_name") == name)
            lr_hist[name].append(g["lr"])

    # -- Plot -----------------------------------------------------------------

    master_lr = [
        BASE_LR * schedule_factor(s, TOTAL_STEPS, schedule_name)
        for s in range(TOTAL_STEPS)
    ]

    steps = range(TOTAL_STEPS)
    fig, (ax_master, ax_layer, ax_loss) = plt.subplots(1, 3, figsize=(15, 4.5))

    # Left: master LR schedule (lr_prefactor envelope)
    ax_master.plot(steps, master_lr, color="black", linewidth=2)
    ax_master.set_xlabel("Step")
    ax_master.set_ylabel("lr_prefactor")
    ax_master.set_title(f"Master LR schedule  ({schedule_name})")
    ax_master.grid(True, alpha=0.3)

    # Middle: per-layer effective LRs (master × fan_in^(-c), c dynamic)
    for name in layer_names:
        short = name.replace(".inner", "")
        ax_layer.plot(steps, lr_hist[name], label=short, linewidth=1.5)
    ax_layer.set_xlabel("Step")
    ax_layer.set_ylabel("Effective LR")
    ax_layer.set_title("Per-layer LRs  (maxP dynamic)")
    ax_layer.legend()
    ax_layer.grid(True, alpha=0.3)

    # Right: training loss (raw + smoothed)
    import numpy as np
    ax_loss.plot(steps, loss_hist, color="black", linewidth=0.5, alpha=0.3)
    win = 20
    if len(loss_hist) > win:
        kernel = np.ones(win) / win
        sm = np.convolve(loss_hist, kernel, mode="valid")
        offset = win // 2
        ax_loss.plot(range(offset, offset + len(sm)), sm,
                     color="black", linewidth=1.8)
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Train loss")
    ax_loss.set_title("Training loss")
    ax_loss.grid(True, alpha=0.3)

    fig.tight_layout()
    out = f"examples/lr_schedule_{schedule_name}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    schedule = sys.argv[1] if len(sys.argv) > 1 else "cosine"
    main(schedule)
