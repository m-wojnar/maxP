#!/usr/bin/env python3
"""Regenerate comparison plots from cached results.

Uses the same hyperparameter args to find the cache files, then re-plots.

Usage:
    python replot.py                    # full run defaults
    python replot.py --steps 500 ...   # match your run's args
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.nanogpt_example.sweep import RunResult, _short_name, smooth


# ── Cache loading (duplicated keys from compare_mup_maxp.py) ─────────────

def _cache_key(method: str, **kwargs) -> str:
    d = {"method": method, **{k: v for k, v in sorted(kwargs.items())}}
    return hashlib.sha256(json.dumps(d).encode()).hexdigest()[:12]


def _cache_path(cache_dir: Path, method: str, key: str) -> Path:
    return cache_dir / f"{method}_{key}.json"


def _load_result(path: Path) -> RunResult | None:
    if not path.exists():
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return RunResult(
        method=data["method"],
        lr=data["lr"],
        losses=data["losses"],
        layer_history=data.get("layer_history", {}),
    )


# ── WSD schedule (for overlay) ───────────────────────────────────────────

def wsd_factor(step, total, warmup=500, decay=1000):
    if step < warmup:
        return (step + 1) / warmup
    decay_start = total - decay
    if step < decay_start:
        return 1.0
    t = (step - decay_start) / decay
    return 0.5 * (1 + math.cos(math.pi * t))


# ── Layer type detection + color scheme ──────────────────────────────────

# Op type → base color (RGB)
_OP_COLORS = {
    "qkv":        (0.12, 0.47, 0.71),   # blue
    "attn_score": (0.68, 0.78, 0.91),   # light blue
    "proj":       (0.17, 0.63, 0.17),   # green
    "fc1":        (1.00, 0.50, 0.05),   # orange
    "fc2":        (0.84, 0.15, 0.16),   # red
    "tok_emb":    (0.58, 0.40, 0.74),   # purple
    "pos_emb":    (0.77, 0.69, 0.83),   # light purple
    "head":       (0.55, 0.34, 0.29),   # brown
}


def _detect_op_type(name: str) -> str:
    """Extract op type from fully-qualified module name (e.g. 'blocks.3.attn.qkv' → 'qkv')."""
    leaf = name.split(".")[-1]
    if leaf in _OP_COLORS:
        return leaf
    return name  # fallback to full name


def _build_layer_colors(names: list[str]) -> dict[str, tuple[float, float, float]]:
    """Assign colors: hue by op type, brightness by block index."""
    by_op: dict[str, list[str]] = {}
    for name in names:
        op = _detect_op_type(name)
        by_op.setdefault(op, []).append(name)

    colors = {}
    for op, layer_names in by_op.items():
        base = _OP_COLORS.get(op, (0.5, 0.5, 0.5))
        n = len(layer_names)
        for i, name in enumerate(layer_names):
            if n == 1:
                factor = 1.0
            else:
                # Range from 0.4 (dark/early block) to 1.0 (bright/late block)
                factor = 0.4 + 0.6 * (i / (n - 1))
            colors[name] = tuple(c * factor for c in base)
    return colors


def _op_legend_handles(names: list[str]):
    """Build legend handles — one entry per op type present in the data."""
    import matplotlib.patches as mpatches
    seen = {}
    for name in names:
        op = _detect_op_type(name)
        if op not in seen:
            seen[op] = _OP_COLORS.get(op, (0.5, 0.5, 0.5))
    return [mpatches.Patch(color=rgb, label=op) for op, rgb in seen.items()]


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_comparison(
    *results: RunResult,
    n_steps: int,
    warmup: int,
    decay: int,
    filename: str = "compare_mup_maxp.png",
    window: int = 50,
):
    import matplotlib.pyplot as plt

    # Find the first result with layer history (maxP)
    hist_result = next((r for r in results if r.layer_history), None)
    has_hist = hist_result is not None

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
        "muP (no-align)": {"color": "#7f7f7f", "ls": "-"},
        "muP":            {"color": "#1f77b4", "ls": "-"},
        "maxP":           {"color": "#d62728", "ls": "--"},
    }

    # ── Loss curves ──
    offset = window // 2
    for run in results:
        st = method_style.get(run.method, {"color": "#333333", "ls": "-"})
        sm = smooth(run.losses, window)
        ax_loss.plot(run.losses, alpha=0.10, color=st["color"], linewidth=0.5)
        ax_loss.plot(
            range(offset, offset + len(sm)), sm,
            color=st["color"], linewidth=2.2, linestyle=st["ls"],
            label=run.method,
        )
        ax_log.plot(run.losses, alpha=0.10, color=st["color"], linewidth=0.5)
        ax_log.plot(
            range(offset, offset + len(sm)), sm,
            color=st["color"], linewidth=2.2, linestyle=st["ls"],
            label=run.method,
        )

    all_losses = sum((r.losses for r in results), [])
    finite = [v for v in all_losses if math.isfinite(v)]
    if finite:
        lo, hi = np.percentile(finite, 1), np.percentile(finite, 95)
        pad = 0.10 * (hi - lo)
        ax_loss.set_ylim(max(0, lo - pad), hi + pad)

    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Train Loss")
    ax_loss.set_title("Loss: muP vs maxP (WSD schedule)")
    ax_loss.legend(fontsize="small")
    ax_loss.grid(True, alpha=0.3)

    ax_log.set_xlabel("Step")
    ax_log.set_ylabel("Train Loss (log)")
    ax_log.set_yscale("log")
    ax_log.set_title("Log scale")
    ax_log.legend(fontsize="x-small")
    ax_log.grid(True, alpha=0.3)

    # ── WSD schedule overlay ──
    sched_steps = list(range(n_steps))
    sched_vals = [wsd_factor(s, n_steps, warmup=warmup, decay=decay) for s in sched_steps]
    ax_sched = ax_loss.twinx()
    ax_sched.plot(sched_steps, sched_vals, color="gray", ls=":", lw=1.0, alpha=0.5)
    ax_sched.set_ylabel("Schedule factor", color="gray", fontsize=8)
    ax_sched.set_ylim(-0.05, 1.15)
    ax_sched.tick_params(axis="y", labelcolor="gray", labelsize=7)

    # ── Per-layer alignment + LR for maxP ──
    if has_hist:
        hist = hist_result.layer_history
        names = list(hist.keys())
        colors = _build_layer_colors(names)

        for name, history in hist.items():
            steps = range(len(history))
            short = _short_name(name)
            c = colors[name]
            ax_alpha.plot(steps, [h["alpha"] for h in history], color=c, lw=1.2)
            ax_omega.plot(steps, [h["omega"] for h in history], color=c, lw=1.2)
            ax_u.plot(steps, [h["u"] for h in history], color=c, lw=1.2)
            ax_lr.plot(steps, [h["lr"] for h in history], color=c, lw=1.2)

        for ax, ref, lbl in [
            (ax_alpha, 1.0, "full=1.0"), (ax_omega, 0.5, "full=0.5"), (ax_u, 1.0, "full=1.0"),
        ]:
            ax.axhline(ref, color="k", ls=":", lw=0.8, alpha=0.5)

        for ax, key in [(ax_alpha, "alpha"), (ax_omega, "omega"), (ax_u, "u"), (ax_lr, "lr")]:
            all_vals = []
            for history in hist.values():
                all_vals.extend(h[key] for h in history if math.isfinite(h[key]))
            if all_vals:
                lo, hi = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
                pad = 0.15 * max(hi - lo, 1e-8)
                ax.set_ylim(lo - pad, hi + pad)

        # Shared op-type legend on LR panel
        legend_handles = _op_legend_handles(names)
        ax_lr.legend(handles=legend_handles, fontsize="x-small", loc="upper right")

    for ax, title, ylabel in [
        (ax_alpha, r"$\alpha$ (z₀ @ $\Delta$w)", r"$\alpha$"),
        (ax_omega, r"$\omega$ ($\Delta$z @ w₀)", r"$\omega$"),
        (ax_u, r"$u$ ($\Delta$z @ $\Delta$w)", r"$u$"),
        (ax_lr, "Per-layer LR (maxP)", "LR"),
    ]:
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # ── Summary table ──
    ax_table.axis("off")
    rows = []
    for run in results:
        tag = f"{run.final_loss:.4f}" if not run.diverged else "DIV"
        rows.append([run.method, f"{run.lr}", tag])
    table = ax_table.table(
        cellText=rows,
        colLabels=["Method", "LR", "Final Loss"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax_table.set_title("Results", fontsize=10)

    fig.suptitle("muP vs maxP (WSD schedule) — Shakespeare GPT", fontsize=14)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {filename}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate comparison plots from cached results"
    )
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--decay", type=int, default=1000)
    parser.add_argument("--alignment-warmup", type=int, default=10)
    parser.add_argument("--solve-interval", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--c-ema", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="compare_mup_maxp.png")
    parser.add_argument("--window", type=int, default=50,
                        help="Smoothing window for loss curves")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Cache directory (default: .result_cache/ next to --output)")
    args = parser.parse_args()

    d_ff = args.d_ff or 4 * args.d_model
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.output).parent / ".result_cache"

    cache_hparams = dict(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=d_ff, seq_len=args.seq_len, steps=args.steps,
        batch_size=args.batch_size, lr=args.lr, warmup=args.warmup,
        decay=args.decay, seed=args.seed,
    )

    # Load all results
    loaded: list[tuple[str, RunResult | None]] = []

    # muP (no-align)
    noalign_hparams = {**cache_hparams, "alignment": "no"}
    noalign_key = _cache_key("muP (no-align)", **noalign_hparams)
    loaded.append(("muP (no-align)", _load_result(_cache_path(cache_dir, "muP_no-align", noalign_key))))

    # muP (full alignment)
    mup_key = _cache_key("muP", **cache_hparams)
    loaded.append(("muP", _load_result(_cache_path(cache_dir, "muP", mup_key))))

    # maxP
    maxp_hparams = {**cache_hparams,
                    "alignment_warmup": args.alignment_warmup,
                    "solve_interval": args.solve_interval,
                    "sample_size": args.sample_size,
                    "c_ema": args.c_ema}
    maxp_key = _cache_key("maxP", **maxp_hparams)
    loaded.append(("maxP", _load_result(_cache_path(cache_dir, "maxP", maxp_key))))

    results = []
    for name, result in loaded:
        if result is None:
            print(f"Warning: {name} result not cached, skipping")
        else:
            tag = "DIV" if result.diverged else f"{result.final_loss:.4f}"
            print(f"{name}: loss={tag}")
            results.append(result)

    if len(results) < 2:
        print("Need at least 2 cached results to plot. Run compare_mup_maxp.py first.")
        sys.exit(1)

    plot_comparison(
        *results,
        n_steps=args.steps,
        warmup=args.warmup,
        decay=args.decay,
        filename=args.output,
        window=args.window,
    )


if __name__ == "__main__":
    main()
