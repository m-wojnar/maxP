"""Coord-check diagnostics: sweep widths, measure activations, plot."""

from __future__ import annotations

import numpy as np

from maxp_new.trace import ClassifiedOp, classify, measure_activations


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
    train_step_fn=None,
):
    """Classify ops and measure activations for a single width axis.

    Args:
        make_model_fn: ``fn(width) -> (model, param_groups_or_None)``
        make_input_fn: ``fn(width) -> input_tensor``  (used for tracing and
            activation measurement; *not* for the training loop)
        widths: list of widths to sweep
        n_steps: training steps per width
        n_seeds: random seeds per width
        train_step_fn: ``fn(model, step_idx) -> None``  Runs one optimiser
            step (forward, backward, step, zero_grad).  If ``None`` a
            default loop using ``AdamW`` with cross-entropy is used — this
            requires the model to have a ``.tok_emb`` attribute (works for
            the example transformers).

    Returns:
        all_ops: list of ClassifiedOp (traced ops, no elementwise)
        affected_indices: list of ints — indices into all_ops
        act_stats: np.array of shape (n_all_ops, n_steps, n_widths, n_seeds)
    """
    import torch

    # Classify using the two smallest widths
    small_model, _ = make_model_fn(widths[0])
    large_model, _ = make_model_fn(widths[1])
    ops = classify(
        small_model, large_model,
        make_input_fn(widths[0]), make_input_fn(widths[1]),
    )

    traced_ops = [op for op in ops if op.op != "elementwise"]
    affected = list(range(len(traced_ops)))
    n_ops = len(traced_ops)

    act_stats = np.zeros((n_ops, n_steps, len(widths), n_seeds))

    for seed_idx in range(n_seeds):
        for w_idx, w in enumerate(widths):
            torch.manual_seed(seed_idx * 1000 + w)
            model, param_groups = make_model_fn(w)

            # Build default train_step if none supplied
            if train_step_fn is not None:
                _step = train_step_fn
            else:
                _step = _default_train_step(model, param_groups)

            for step in range(n_steps):
                torch.manual_seed(seed_idx * 100000 + step)
                x = make_input_fn(w)
                stats = measure_activations(model, x)
                for op_idx in range(n_ops):
                    act_stats[op_idx, step, w_idx, seed_idx] = stats[op_idx]

                _step(model, step)

    return traced_ops, affected, act_stats


def _default_train_step(model, param_groups):
    """Return a closure that does one AdamW step with cross-entropy loss.

    Assumes model has a ``.tok_emb`` attribute (plain ``nn.Embedding`` or
    ``ParametrizedModule`` wrapping one).
    """
    import torch
    import torch.nn.functional as F
    from maxp_new.module import ParametrizedModule

    tok_emb = model.tok_emb
    if isinstance(tok_emb, ParametrizedModule):
        tok_emb = tok_emb.inner
    vocab_size = tok_emb.num_embeddings

    if param_groups is not None:
        opt = torch.optim.AdamW(param_groups)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def step(model, step_idx):
        x = torch.randint(0, vocab_size, (16, 32))
        targets = torch.randint(0, vocab_size, (16, 32))
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()

    return step


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


def plot_axis(axis_name, ops, affected, act_stats, widths, filename,
              plot_every=1):
    """Plot coord-check figure for one axis."""
    import matplotlib.pyplot as plt
    from matplotlib import cm

    n_affected = len(affected)
    if n_affected == 0:
        print(f"  No affected ops for axis '{axis_name}', skipping plot.")
        return

    n_steps = act_stats.shape[1]
    steps_to_plot = list(range(0, n_steps, plot_every))
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

        for step in steps_to_plot:
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
