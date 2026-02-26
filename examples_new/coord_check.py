"""Coordinate check: verify parametrization is width-invariant.

For each width, init model, train for a few steps, and measure
abs(activation).mean() at each layer via forward hooks.

If correctly parametrized, activations stay O(1) across widths at every step.

Matches the nanoGPT-mup coord check format:
  - Rows = parametrizations, Cols = layer types
  - X-axis = width (log), Y-axis = abs(activation).mean() (log)
  - Lines colored by training step

Usage:
    python examples_new/coord_check.py              # prints table
    python examples_new/coord_check.py --plot        # saves plot to coord_check.png
"""

import argparse
from functools import partial

import numpy as np
import torch
import torch.nn as nn

import sys; sys.path.insert(0, ".")
from maxp_new.parametrization import Parametrization


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.embed = nn.Linear(d_in, d_hidden)
        self.act1 = nn.ReLU()
        self.hidden = nn.Linear(d_hidden, d_hidden)
        self.act2 = nn.ReLU()
        self.readout = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = self.act1(self.embed(x))
        x = self.act2(self.hidden(x))
        return self.readout(x)


class MLPLayerNorm(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(d_in, d_hidden), nn.LayerNorm(d_hidden))
        self.act1 = nn.ReLU()
        self.hidden = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.LayerNorm(d_hidden))
        self.act2 = nn.ReLU()
        self.readout = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = self.act1(self.embed(x))
        x = self.act2(self.hidden(x))
        return self.readout(x)


# Layer names to hook and their display labels
LAYER_NAMES = ["embed", "hidden", "readout"]
LAYER_LABELS = ["Embedding", "Hidden", "Readout"]


def coord_check(make_model, widths, d_in=8, d_out=2, n_steps=10, n_seeds=5):
    """Run coord check: per-layer abs(activation).mean() across widths and steps.

    Returns: dict[layer_name] -> np.array of shape (n_steps, n_widths, n_seeds)
    """
    results = {name: np.zeros((n_steps, len(widths), n_seeds)) for name in LAYER_NAMES}

    for seed_idx in range(n_seeds):
        for width_idx, w in enumerate(widths):
            torch.manual_seed(seed_idx * 1000 + w)
            model, opt = make_model(d_in, w, d_out)

            # Register hooks (on outer module, so ScaledModule scaling is included)
            activations = {}
            handles = []
            for name in LAYER_NAMES:
                mod = getattr(model, name)
                def hook(module, input, output, key=name):
                    activations[key] = output.detach().abs().mean().item()
                handles.append(mod.register_forward_hook(hook))

            for step in range(n_steps):
                torch.manual_seed(seed_idx * 100000 + step)
                x = torch.randn(64, d_in)
                y = torch.randn(64, d_out)

                # Forward (hooks fire here)
                out = model(x)
                for name in LAYER_NAMES:
                    results[name][step, width_idx, seed_idx] = activations[name]

                # Train step
                loss = nn.functional.mse_loss(out, y)
                loss.backward()
                opt.step()
                opt.zero_grad()

            for h in handles:
                h.remove()

    return results


# --- Model factories ---

def make_sp(d_in, d_hidden, d_out):
    """SP: standard PyTorch init, global LR. Should be unstable."""
    model = MLP(d_in, d_hidden, d_out)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    return model, opt


def make_sp_maxp(d_in, d_hidden, d_out):
    """SP init + solver-optimized per-layer LR."""
    model = MLP(d_in, d_hidden, d_out)
    param = Parametrization(
        model,
        layers={
            "embed": {"a": 0.0, "b": 0.0},
            "hidden": {"a": 0.0, "b": 0.5},
            "readout": {"a": 0.0, "b": 0.5},
        },
        optimizer_type="adam",
        alignment="full",
        lr_prefactor=1e-2,
    )
    opt = torch.optim.AdamW(param.param_groups)
    return model, opt


def make_mup_maxp(d_in, d_hidden, d_out):
    """muP init + solver-optimized per-layer LR."""
    model = MLP(d_in, d_hidden, d_out)
    param = Parametrization(
        model,
        layers={
            "embed": {"a": -0.5, "b": 0.5},
            "hidden": {"a": 0.0, "b": 0.5},
            "readout": {"a": 0.5, "b": 0.0},
        },
        optimizer_type="adam",
        alignment="full",
        lr_prefactor=1e-2,
    )
    opt = torch.optim.AdamW(param.param_groups)
    return model, opt


def make_sp_ln(d_in, d_hidden, d_out):
    """SP with LayerNorm, global LR."""
    model = MLPLayerNorm(d_in, d_hidden, d_out)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    return model, opt


def make_sp_maxp_ln(d_in, d_hidden, d_out):
    """SP + LayerNorm + solver-optimized per-layer LR."""
    model = MLPLayerNorm(d_in, d_hidden, d_out)
    param = Parametrization(
        model,
        layers={
            "embed.0": {"a": 0.0, "b": 0.0},
            "hidden.0": {"a": 0.0, "b": 0.5},
            "readout": {"a": 0.0, "b": 0.5},
        },
        optimizer_type="adam",
        alignment="full",
        lr_prefactor=1e-2,
    )
    opt = torch.optim.AdamW(param.param_groups)
    return model, opt


# --- Visualization ---

PARAMETRIZATIONS = [
    ("SP (default PyTorch)", make_sp),
    ("SP + maxP LR", make_sp_maxp),
    ("SP + LayerNorm", make_sp_ln),
    ("SP + LayerNorm + maxP LR", make_sp_maxp_ln),
]


def plot_coord_check(all_results: dict[str, dict], widths, filename="coord_check.png"):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    n_rows = len(all_results)
    n_cols = len(LAYER_NAMES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (param_name, layer_results) in enumerate(all_results.items()):
        n_steps = layer_results[LAYER_NAMES[0]].shape[0]
        cmap = cm.coolwarm
        norm = plt.Normalize(0, n_steps - 1)

        for col_idx, (layer_name, layer_label) in enumerate(zip(LAYER_NAMES, LAYER_LABELS)):
            ax = axes[row_idx, col_idx]
            data = layer_results[layer_name]  # (n_steps, n_widths, n_seeds)

            for step in range(n_steps):
                means = data[step].mean(axis=1)  # avg over seeds
                stderrs = data[step].std(axis=1) / np.sqrt(data.shape[2])
                color = cmap(norm(step))
                ax.plot(widths, means, marker=".", color=color,
                        label=f"{step}" if col_idx == 0 else None)
                ax.fill_between(widths, means - stderrs, means + stderrs,
                                color=color, alpha=0.3)

            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            if row_idx == 0:
                ax.set_title(layer_label)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Width")
            if col_idx == 0:
                ax.set_ylabel(f"{param_name}\nabs(act).mean()")
                ax.legend(loc="upper left", fontsize=6, title="Step", ncol=2)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Saved plot to {filename}")


def print_results(all_results: dict[str, dict], widths):
    for param_name, layer_results in all_results.items():
        print(f"\n  {param_name}")
        n_steps = layer_results[LAYER_NAMES[0]].shape[0]
        # Print step 0 (init) and last step
        for step in [0, n_steps - 1]:
            label = "init" if step == 0 else f"step {step}"
            header = "  " + f"{'width':>6}"
            for ln in LAYER_NAMES:
                header += f"  {ln:>10}"
            if step == 0:
                print(header)
                print("  " + "-" * (6 + 12 * len(LAYER_NAMES)))
            print(f"  [{label}]")
            for w_idx, w in enumerate(widths):
                row = f"  {w:>6}"
                for ln in LAYER_NAMES:
                    val = layer_results[ln][step, w_idx].mean()
                    row += f"  {val:>10.4f}"
                print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    widths = [64, 128, 256, 512, 1024, 2048, 4096]

    all_results = {}
    for name, factory in PARAMETRIZATIONS:
        print(f"Running {name}...")
        all_results[name] = coord_check(factory, widths,
                                        n_steps=args.steps, n_seeds=args.seeds)

    print_results(all_results, widths)

    if args.plot:
        plot_coord_check(all_results, widths)
