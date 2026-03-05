#!/usr/bin/env python3
"""Diagnostic: per-layer LR comparison between maxP (dynamic) and muP (no-align, static).

Loads maxP layer_history from cache and computes muP (no-align) static LRs
by instantiating the parametrization (no training needed).

Usage:
    python plot_layer_lrs.py --dataset openwebtext --preset gpt2-small \
        --maxp-lr 0.03 --noalign-lr 0.01 --steps 5000 \
        --batch-size 8 --sample-size 8 --solve-interval 100
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.nanogpt_example.compare_mup_maxp import (
    PRESETS,
    _cache_key,
    _cache_path,
    _load_result,
    _make_model,
)
from maxp import Parametrization


DATASET_VOCAB = {"openwebtext": 50257, "shakespeare": 65}

# ── Op type ordering for consistent row layout ──
OP_ORDER = ["tok_emb", "pos_emb", "qkv", "proj", "fc1", "fc2", "head"]


def _detect_op_type(name: str) -> str:
    leaf = name.split(".")[-1]
    if leaf in OP_ORDER:
        return leaf
    return name


def _block_index(name: str) -> int | None:
    """Extract block index from name like 'blocks.3.attn.qkv'."""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "blocks" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Per-layer LR comparison: maxP vs muP (no-align)"
    )
    parser.add_argument("--dataset", type=str, default="openwebtext",
                        choices=["shakespeare", "openwebtext"])
    parser.add_argument("--preset", type=str, default="none",
                        choices=["none", "gpt2-small", "gpt2-debug"])
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--decay", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    # maxP-specific
    parser.add_argument("--maxp-lr", type=float, required=True,
                        help="LR used for the maxP run")
    parser.add_argument("--alignment-warmup", type=int, default=10)
    parser.add_argument("--solve-interval", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--c-ema", type=float, default=0.0)
    # muP (no-align)
    parser.add_argument("--noalign-lr", type=float, required=True,
                        help="LR used for the muP (no-align) run")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="layer_lrs.png")
    args = parser.parse_args()

    # Resolve model dims
    defaults = {"d_model": 512, "n_heads": 8, "n_layers": 8, "d_ff": None, "seq_len": 64}
    if args.preset != "none":
        defaults.update(PRESETS[args.preset])

    d_model = args.d_model or defaults["d_model"]
    n_heads = args.n_heads or defaults["n_heads"]
    n_layers = args.n_layers or defaults["n_layers"]
    seq_len = args.seq_len or defaults["seq_len"]
    d_ff = args.d_ff or defaults["d_ff"] or 4 * d_model
    vocab_size = DATASET_VOCAB[args.dataset]

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.output).parent / ".result_cache"

    # ── Load maxP cached result ──
    cache_hparams_base = dict(
        dataset=args.dataset,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ff=d_ff, seq_len=seq_len, steps=args.steps,
        batch_size=args.batch_size, warmup=args.warmup,
        decay=args.decay, seed=args.seed,
    )
    maxp_hp = {
        **cache_hparams_base, "lr": args.maxp_lr,
        "alignment_warmup": args.alignment_warmup,
        "solve_interval": args.solve_interval,
        "sample_size": args.sample_size,
        "c_ema": args.c_ema,
    }
    maxp_key = _cache_key("maxP", **maxp_hp)
    maxp_path = _cache_path(cache_dir, "maxP", maxp_key)
    maxp_result = _load_result(maxp_path)
    if maxp_result is None:
        print(f"maxP result not found at {maxp_path}")
        print("Run compare_mup_maxp.py first.")
        sys.exit(1)

    hist = maxp_result.layer_history
    if not hist:
        print("maxP result has no layer_history")
        sys.exit(1)
    print(f"Loaded maxP result: lr={maxp_result.lr}, {len(hist)} layers, "
          f"{len(next(iter(hist.values())))} steps")

    # ── Compute muP (no-align) static per-layer LRs ──
    device = "cpu"
    torch.manual_seed(args.seed)
    model, sample_input = _make_model(d_model, n_heads, n_layers, d_ff, vocab_size, seq_len, device)
    param = Parametrization(
        model,
        lr_prefactor=args.noalign_lr,
        optimizer_type="adam",
        alignment="no",
        sample_input=sample_input,
    )
    noalign_lrs = {}
    for group in param.param_groups:
        name = group.get("layer_name", "")
        if group.get("maxp_managed", False):
            noalign_lrs[name] = group["lr"]
    print(f"Computed muP (no-align) LRs for {len(noalign_lrs)} layers at lr_prefactor={args.noalign_lr}")

    # ── Organize layers by op type and block index ──
    layer_names = list(hist.keys())
    by_op: dict[str, list[str]] = {}
    for name in layer_names:
        op = _detect_op_type(name)
        by_op.setdefault(op, []).append(name)

    # Sort ops by OP_ORDER, then unknown ops
    op_types = sorted(by_op.keys(), key=lambda o: OP_ORDER.index(o) if o in OP_ORDER else 999)

    # Sort layers within each op by block index
    for op in op_types:
        by_op[op].sort(key=lambda n: (_block_index(n) or -1, n))

    max_cols = max(len(layers) for layers in by_op.values())
    n_rows = len(op_types)

    # ── Plot ──
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        n_rows, max_cols,
        figsize=(3 * max_cols, 2.5 * n_rows),
        squeeze=False,
        sharex=True,
    )

    for row, op in enumerate(op_types):
        layers = by_op[op]
        for col in range(max_cols):
            ax = axes[row, col]
            if col >= len(layers):
                ax.set_visible(False)
                continue

            name = layers[col]
            history = hist[name]
            steps = range(len(history))
            maxp_lr_series = [h["lr"] for h in history]

            # maxP dynamic LR
            ax.plot(steps, maxp_lr_series, color="#d62728", lw=1.2, alpha=0.9,
                    label=f"maxP (lr={args.maxp_lr})")

            # muP (no-align) static LR as dashed line
            if name in noalign_lrs:
                ax.axhline(noalign_lrs[name], color="#7f7f7f", ls="--", lw=1.5,
                           alpha=0.8, label=f"muP no-align (lr={args.noalign_lr})")

            block_idx = _block_index(name)
            title = f"{op}" + (f" [block {block_idx}]" if block_idx is not None else "")
            ax.set_title(title, fontsize=8)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=6)

            if col == 0:
                ax.set_ylabel("LR", fontsize=7)
            if row == 0 and col == 0:
                ax.legend(fontsize=6, loc="upper right")

    for col in range(max_cols):
        axes[-1, col].set_xlabel("Step", fontsize=7)

    fig.suptitle(
        f"Per-layer LR: maxP (lr={args.maxp_lr}) vs muP no-align (lr={args.noalign_lr})\n"
        f"{args.dataset}, d={d_model}, {n_layers}L",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
