"""Diagnostic: trace, classify, and coord-check every op in an architecture.

For each width-scaling dimension, independently:
  1. Trace at two widths → classify every op
  2. Filter to ops affected by this dimension (hidden/readout)
  3. Sweep widths, measure abs(activation).mean() per op across training steps
  4. Produce a figure with one subplot per affected op

Usage:
    python examples_new/diagnose.py --plot
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

import sys; sys.path.insert(0, ".")
from maxp_new.trace import classify, measure_activations, ClassifiedOp
from examples_new.transformer import Transformer


def op_label(op: ClassifiedOp) -> str:
    """Short human-readable label for an op."""
    if op.param_name:
        name = op.param_name.replace(".weight", "")
    elif op.module_path:
        name = op.module_path
    else:
        name = f"op{op.index}"

    loc = ""
    if op.source_loc and not op.parametrized:
        loc = f" @ {op.source_loc}"

    return f"#{op.index} {name}{loc}"


def diagnose_axis(
    make_model_fn,
    make_input_fn,
    widths,
    n_steps=10,
    n_seeds=3,
):
    """Classify ops and measure activations for a single width axis.

    Returns:
        all_ops: list of ClassifiedOp (traced ops, no elementwise)
        affected_indices: list of ints — indices into all_ops for non-embedding ops
        act_stats: np.array of shape (n_all_ops, n_steps, n_widths, n_seeds)
    """
    # Classify using the two smallest widths
    small_model = make_model_fn(widths[0])
    large_model = make_model_fn(widths[1])
    ops = classify(small_model, large_model, make_input_fn(widths[0]), make_input_fn(widths[1]))

    traced_ops = [op for op in ops if op.op != "elementwise"]
    affected = list(range(len(traced_ops)))
    n_ops = len(traced_ops)

    # Sweep widths
    act_stats = np.zeros((n_ops, n_steps, len(widths), n_seeds))

    for seed_idx in range(n_seeds):
        for w_idx, w in enumerate(widths):
            torch.manual_seed(seed_idx * 1000 + w)
            model = make_model_fn(w)
            vocab_size = model.tok_emb.num_embeddings
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

            for step in range(n_steps):
                torch.manual_seed(seed_idx * 100000 + step)
                x = torch.randint(0, vocab_size, (16, 32))
                targets = torch.randint(0, vocab_size, (16, 32))

                stats = measure_activations(model, x)
                for op_idx in range(n_ops):
                    act_stats[op_idx, step, w_idx, seed_idx] = stats[op_idx]

                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
                loss.backward()
                opt.step()
                opt.zero_grad()

    return traced_ops, affected, act_stats


def print_axis(axis_name, ops, affected, act_stats, widths):
    """Print table for one axis."""
    print(f"\n=== Axis: {axis_name} ===")
    print(f"  Affected ops ({len(affected)} of {len(ops)} traced):\n")
    print(f"  {'idx':<4} {'op':<10} {'type':<10} {'p?':<4} {'label':<45}", end="")
    for w in widths:
        print(f"{axis_name + '=' + str(w):>14}", end="")
    print()
    print("  " + "-" * (73 + 14 * len(widths)))

    for i in affected:
        op = ops[i]
        label = op_label(op)
        p = "Y" if op.parametrized else "N"
        print(f"  {op.index:<4} {op.op:<10} {op.layer_type:<10} {p:<4} {label:<45}", end="")
        for w_idx in range(len(widths)):
            val = act_stats[i, 0, w_idx].mean()
            print(f"{val:>14.4f}", end="")
        print()


def plot_axis(axis_name, ops, affected, act_stats, widths, filename):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    n_affected = len(affected)
    if n_affected == 0:
        print(f"  No affected ops for axis '{axis_name}', skipping plot.")
        return

    n_steps = act_stats.shape[1]
    n_cols = min(4, n_affected)
    n_rows = (n_affected + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False)

    cmap = cm.coolwarm
    norm = plt.Normalize(0, n_steps - 1)

    for plot_idx, op_idx in enumerate(affected):
        row, col = divmod(plot_idx, n_cols)
        ax = axes[row, col]
        op = ops[op_idx]
        data = act_stats[op_idx]

        for step in range(n_steps):
            means = data[step].mean(axis=1)
            stderrs = data[step].std(axis=1) / np.sqrt(data.shape[2])
            color = cmap(norm(step))
            ax.plot(widths, means, marker=".", color=color, markersize=4,
                    label=f"{step}" if plot_idx == 0 else None)
            ax.fill_between(widths, means - stderrs, means + stderrs,
                            color=color, alpha=0.3)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        label = op_label(op)
        kind = f"{op.layer_type}, {'param' if op.parametrized else 'unparam'}"
        ax.set_title(f"{label}\n[{kind}]", fontsize=7, family="monospace")

        if row == n_rows - 1:
            ax.set_xlabel(axis_name)
        if col == 0:
            ax.set_ylabel("abs(act).mean()")

    for i in range(n_affected, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    axes[0, 0].legend(loc="upper left", fontsize=5, title="Step", ncol=2)

    fig.suptitle(f"Coord check — {axis_name} axis", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  Saved {filename}")


# --- Transformer-specific setup ---
#
# Each axis sweeps one width dimension. d_model and head_dim are coupled
# (d_model = n_heads * head_dim), so sweeping head_dim also sweeps d_model.
# Axes are processed in order — ops already claimed by an earlier axis are
# deduplicated and shown as warnings.

def make_transformer(**kwargs):
    defaults = dict(vocab_size=256, d_model=128, n_heads=4, d_ff=256, n_layers=2)
    defaults.update(kwargs)
    return Transformer(**defaults)


AXES = {
    # Scale d_model (and d_ff). Fixed head_dim=16, n_heads scales.
    "d_model": {
        "make_model": lambda w: make_transformer(d_model=w, n_heads=w // 16, d_ff=2 * w),
        "widths": [64, 128, 256, 512, 1024],
    },
    # Scale head_dim (n_heads=4 fixed, so d_model = 4*head_dim co-scales).
    # Ops already in d_model axis will be deduplicated.
    "head_dim": {
        "make_model": lambda w: make_transformer(d_model=4 * w, n_heads=4, d_ff=8 * w),
        "widths": [16, 32, 64, 128, 256],
    },
}


def make_input(d_model):
    return torch.randint(0, 256, (1, 8))


def _op_key(op: ClassifiedOp) -> str:
    """Stable identity for an op across axes (param_name or module_path:source_loc)."""
    if op.param_name:
        return op.param_name
    return f"{op.module_path}:{op.source_loc}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    # An op classified as "embedding" in axis A is unaffected by that width dim,
    # but may be "hidden"/"readout" in axis B.  We only dedup non-embedding ops
    # across axes; embedding-type ops don't block later axes from claiming them.
    claimed: dict[str, str] = {}  # op_key -> axis_name (non-embedding only)
    shown_embeddings: set[str] = set()  # embedding ops already displayed

    for axis_name, axis_cfg in AXES.items():
        print(f"\nDiagnosing axis: {axis_name}...")
        ops, affected, act_stats = diagnose_axis(
            make_model_fn=axis_cfg["make_model"],
            make_input_fn=make_input,
            widths=axis_cfg["widths"],
            n_steps=args.steps,
            n_seeds=args.seeds,
        )

        deduped = []
        for i in affected:
            op = ops[i]
            key = _op_key(op)

            if op.layer_type == "embedding" and op.op != "embedding":
                # Unaffected matmul (e.g. QK^T in d_model axis) — skip here,
                # it will appear in the axis where it's actually affected.
                continue
            elif op.layer_type == "embedding" and op.op == "embedding":
                # Actual nn.Embedding lookup — show in first axis only.
                if key not in shown_embeddings:
                    shown_embeddings.add(key)
                    deduped.append(i)
            else:
                # Non-embedding (hidden/readout) — dedup across axes.
                if key in claimed:
                    print(f"  WARNING: {op_label(op)} already in '{claimed[key]}' axis, skipping")
                else:
                    claimed[key] = axis_name
                    deduped.append(i)

        print_axis(axis_name, ops, deduped, act_stats, axis_cfg["widths"])

        if args.plot:
            plot_axis(axis_name, ops, deduped, act_stats,
                      axis_cfg["widths"], f"diagnose_{axis_name}.png")
