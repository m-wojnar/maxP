"""Coordinate check for ParametrizedModule-based models.

Compares activation stability across widths and training steps for:
  - vanilla: ParametrizedModule wrapping, but no Parametrization applied
             (PyTorch default init, no scaling, flat LR)
  - muP:     Parametrization applied (ABC init + scaling + per-layer LR)

Produces one figure per width-scaling axis (d_model, head_dim), matching
the layout of diagnose.py: one subplot per ParametrizedModule, with
vanilla (dashed) and muP (solid) overlaid.

Usage:
    python examples_new/coord_check_pm.py              # text table
    python examples_new/coord_check_pm.py --plot        # + PNG figures
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

import sys; sys.path.insert(0, ".")
from maxp_new.utils import ParametrizedModule
from maxp_new.parametrization import Parametrization
from examples_new.parameterized_transformer import Transformer


# ---------------------------------------------------------------------------
# Core coord-check routine
# ---------------------------------------------------------------------------

def coord_check(
    make_model_fn,
    make_input_fn,
    widths,
    n_steps=10,
    n_seeds=3,
):
    """Measure ParametrizedModule outputs across widths and training steps.

    Returns:
        pm_names: list of ParametrizedModule names.
        stats: dict  name → np.array of shape (n_steps, n_widths, n_seeds)
    """
    # Discover PM names from a reference model
    ref_model, _ = make_model_fn(widths[0])
    pm_names = []
    for name, mod in ref_model.named_modules():
        if isinstance(mod, ParametrizedModule):
            pm_names.append(name)
    del ref_model

    stats = {n: np.zeros((n_steps, len(widths), n_seeds)) for n in pm_names}

    for seed_idx in range(n_seeds):
        for w_idx, w in enumerate(widths):
            torch.manual_seed(seed_idx * 1000 + w)
            model, param_groups = make_model_fn(w)

            if param_groups is not None:
                opt = torch.optim.AdamW(param_groups)
            else:
                opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

            for step in range(n_steps):
                captured = {}
                hooks = []
                for name, mod in model.named_modules():
                    if isinstance(mod, ParametrizedModule):
                        def make_hook(n):
                            def hook_fn(module, inp, out):
                                captured[n] = out.detach().float().abs().mean().item()
                            return hook_fn
                        hooks.append(mod.register_forward_hook(make_hook(name)))

                torch.manual_seed(seed_idx * 100000 + step)
                input_ids, targets = make_input_fn(w)
                logits = model(input_ids)

                for h in hooks:
                    h.remove()

                for name in pm_names:
                    if name in captured:
                        stats[name][step, w_idx, seed_idx] = captured[name]

                vocab_size = logits.shape[-1]
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size), targets.reshape(-1))
                loss.backward()
                opt.step()
                opt.zero_grad()

    return pm_names, stats


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def short_name(name):
    """Shorten 'blocks.0.attn.qkv' to 'b0.attn.qkv' etc."""
    return name.replace("blocks.", "b")


def print_table(label, pm_names, stats, widths, steps_to_show=None):
    if steps_to_show is None:
        n_steps = stats[pm_names[0]].shape[0]
        steps_to_show = [0, n_steps - 1]

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    for step in steps_to_show:
        print(f"\n  Step {step}:")
        print(f"  {'module':<30s}", end="")
        for w in widths:
            print(f"{'w=' + str(w):>10s}", end="")
        print(f"{'ratio':>10s}")
        print(f"  {'-' * (30 + 10 * len(widths) + 10)}")

        for name in pm_names:
            data = stats[name]
            means = data[step].mean(axis=1)
            ratio = means[-1] / means[0] if means[0] > 1e-12 else float('inf')
            print(f"  {short_name(name):<30s}", end="")
            for val in means:
                print(f"{val:>10.4f}", end="")
            print(f"{ratio:>10.2f}x")


def plot_axis(axis_name, pm_names, stats_vanilla, stats_mup, widths, filename):
    """One figure per axis: subplot per module, vanilla vs muP overlaid."""
    import matplotlib.pyplot as plt
    from matplotlib import cm

    n_modules = len(pm_names)
    n_steps = stats_vanilla[pm_names[0]].shape[0]
    n_cols = min(4, n_modules)
    n_rows = (n_modules + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    cmap = cm.coolwarm
    norm = plt.Normalize(0, n_steps - 1)

    for plot_idx, name in enumerate(pm_names):
        row, col = divmod(plot_idx, n_cols)
        ax = axes[row, col]

        data_v = stats_vanilla[name]
        data_m = stats_mup[name]

        for step in range(n_steps):
            color = cmap(norm(step))

            # Vanilla: dashed
            means_v = data_v[step].mean(axis=1)
            ax.plot(widths, means_v, marker="x", color=color, markersize=3,
                    linestyle="--", linewidth=1, alpha=0.6)

            # muP: solid
            means_m = data_m[step].mean(axis=1)
            stds_m = data_m[step].std(axis=1) / max(1, np.sqrt(data_m.shape[2]))
            ax.plot(widths, means_m, marker=".", color=color, markersize=4,
                    linestyle="-", linewidth=1.5)
            ax.fill_between(widths, means_m - stds_m, means_m + stds_m,
                            color=color, alpha=0.15)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_title(short_name(name), fontsize=8, family="monospace")

        if row == n_rows - 1:
            ax.set_xlabel(axis_name)
        if col == 0:
            ax.set_ylabel("abs(act).mean()")

    # Hide unused subplots
    for i in range(n_modules, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    # Legend: one dummy entry for vanilla and muP
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="gray", linestyle="--", marker="x",
               markersize=4, label="vanilla"),
        Line2D([0], [0], color="gray", linestyle="-", marker=".",
               markersize=6, linewidth=1.5, label="muP"),
    ]
    fig.legend(handles=legend_elements, loc="upper right",
               fontsize=9, framealpha=0.9)

    fig.suptitle(f"Coord check — {axis_name} axis  (dashed=vanilla, solid=muP)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  Saved {filename}")


# ---------------------------------------------------------------------------
# Transformer model factories — per axis
# ---------------------------------------------------------------------------

def _make_vanilla(d_model, n_heads, d_ff):
    model = Transformer(vocab_size=256, d_model=d_model, n_heads=n_heads,
                        d_ff=d_ff, n_layers=2)
    return model, None

def _make_mup(d_model, n_heads, d_ff):
    model = Transformer(vocab_size=256, d_model=d_model, n_heads=n_heads,
                        d_ff=d_ff, n_layers=2)
    param = Parametrization(model, lr_prefactor=0.1)
    return model, param.param_groups


# d_model axis: scale d_model and n_heads, fixed head_dim=16
AXES = {
    "d_model": {
        "make_vanilla": lambda w: _make_vanilla(d_model=w, n_heads=w // 16, d_ff=2 * w),
        "make_mup":     lambda w: _make_mup(d_model=w, n_heads=w // 16, d_ff=2 * w),
        "widths": [64, 128, 256, 512, 1024],
    },
    # head_dim axis: fixed n_heads=4, scale head_dim (d_model = 4 * head_dim)
    "head_dim": {
        "make_vanilla": lambda w: _make_vanilla(d_model=4 * w, n_heads=4, d_ff=8 * w),
        "make_mup":     lambda w: _make_mup(d_model=4 * w, n_heads=4, d_ff=8 * w),
        "widths": [16, 32, 64, 128, 256],
    },
}


def make_input(width):
    input_ids = torch.randint(0, 256, (16, 32))
    targets = torch.randint(0, 256, (16, 32))
    return input_ids, targets


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    for axis_name, cfg in AXES.items():
        widths = cfg["widths"]

        print(f"\n--- Axis: {axis_name} ---")

        print(f"  Running vanilla...")
        names_v, stats_v = coord_check(
            cfg["make_vanilla"], make_input, widths,
            n_steps=args.steps, n_seeds=args.seeds)

        print(f"  Running muP...")
        names_m, stats_m = coord_check(
            cfg["make_mup"], make_input, widths,
            n_steps=args.steps, n_seeds=args.seeds)

        print_table(f"Vanilla — {axis_name} axis", names_v, stats_v, widths)
        print_table(f"muP — {axis_name} axis", names_m, stats_m, widths)

        if args.plot:
            plot_axis(axis_name, names_v, stats_v, stats_m, widths,
                      f"coord_check_{axis_name}.png")
