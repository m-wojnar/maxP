#!/usr/bin/env python3
"""Configurable experiment runner.

Edit RUNS below to specify exactly which methods/LRs to train.
Results are cached, so re-running skips completed experiments.
Comment out runs you don't need.

Usage:
    python run_experiment.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.nanogpt_example.compare_mup_maxp import (
    _cache_key,
    _cache_path,
    _load_result,
    _make_model,
    _save_result,
    train_maxp,
    train_mup,
    train_mup_noalign,
)
from examples.nanogpt_example.replot import plot_comparison
from examples.nanogpt_example.sweep import (
    RunResult,
    get_device,
    load_openwebtext,
    load_shakespeare,
    pick_best,
)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURE YOUR EXPERIMENT HERE
# ═══════════════════════════════════════════════════════════════════════════

DATASET = "openwebtext"
DATA_DIR = "./data/openwebtext"

# Model config (or use a preset below)
MODEL = dict(d_model=768, n_heads=12, n_layers=12, d_ff=3072, seq_len=1024)  # gpt2-small
# MODEL = dict(d_model=256, n_heads=8, n_layers=6, d_ff=1024, seq_len=256)  # gpt2-debug

# Shared training config
BATCH_SIZE = 8
SEED = 42
WARMUP = 500
DECAY = None  # set to 10% of each run's steps (computed per run)

# maxP-specific defaults
MAXP_DEFAULTS = dict(
    alignment_warmup=10,
    solve_interval=100,
    sample_size=8,
    c_ema=0.0,
)

# ── Define your runs ──────────────────────────────────────────────────────
# Each entry: (method, lr, steps, extra_kwargs)
#   or:       (method, lr, steps, extra_kwargs, display_name)
# method: "maxP", "muP", "muP (no-align)"
# extra_kwargs: only needed for maxP (overrides MAXP_DEFAULTS)
# display_name: optional label for plots/summary (defaults to method)

NO_ALIGN = (0.5, 0.5, 0.5)

RUNS = [
    ("maxP",           0.03, 40000, {}),
    ("maxP",           0.03, 40000, {"alignment_overrides": {"fc2": NO_ALIGN}}, "maxP (fc2-noalign)"),
    ("muP (no-align)", 0.01, 40000, {}),
    # ("muP",          0.01, 5000, {}),
]

OUTPUT = "experiment.png"

# ═══════════════════════════════════════════════════════════════════════════
# END CONFIG
# ═══════════════════════════════════════════════════════════════════════════

TRAIN_FN = {
    "maxP": train_maxp,
    "muP": train_mup,
    "muP (no-align)": train_mup_noalign,
}

FILE_PREFIX = {
    "maxP": "maxP",
    "muP": "muP",
    "muP (no-align)": "muP_no-align",
}


def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"Dataset: {DATASET}")
    print(f"Model: {MODEL}")
    print(f"Runs: {len(RUNS)}")
    print()

    cache_dir = Path(OUTPUT).parent / ".result_cache"

    # Load data
    if DATASET == "openwebtext":
        print(f"Loading OpenWebText from {DATA_DIR}...")
        data, vocab_size = load_openwebtext(DATA_DIR)
        print(f"  {len(data):,} tokens, vocab size: {vocab_size}")
    else:
        print("Loading Shakespeare...")
        data, vocab_size, _ = load_shakespeare(device)
        print(f"  {len(data):,} chars, vocab size: {vocab_size}")

    # Run experiments
    results: list[RunResult] = []

    for i, run_spec in enumerate(RUNS):
        method, lr, steps, extra = run_spec[:4]
        display_name = run_spec[4] if len(run_spec) > 4 else method
        decay = DECAY if DECAY is not None else steps // 10
        print(f"\n[{i+1}/{len(RUNS)}] {display_name}  lr={lr}  steps={steps}  decay={decay}")

        cache_hp = dict(
            dataset=DATASET, **MODEL,
            steps=steps, batch_size=BATCH_SIZE,
            lr=lr, warmup=WARMUP, decay=decay, seed=SEED,
        )

        train_kwargs = dict(
            **MODEL, data=data, vocab_size=vocab_size,
            lr=lr, n_steps=steps,
            batch_size=BATCH_SIZE, seed=SEED,
            warmup=WARMUP, decay=decay, device=device,
        )

        if method == "maxP":
            maxp_cfg = {**MAXP_DEFAULTS, **extra}
            # alignment_overrides needs special handling (not JSON-serializable as-is)
            align_ov = maxp_cfg.pop("alignment_overrides", None)
            cache_hp.update(maxp_cfg)
            if align_ov:
                cache_hp["alignment_overrides"] = {k: list(v) for k, v in align_ov.items()}
            train_kwargs.update(maxp_cfg)
            if align_ov:
                train_kwargs["alignment_overrides"] = align_ov
            train_kwargs["method_name"] = display_name
        elif method == "muP (no-align)":
            cache_hp["alignment"] = "no"

        file_prefix = FILE_PREFIX[method]
        if display_name != method:
            file_prefix = display_name.replace(" ", "_").replace("(", "").replace(")", "")
        key = _cache_key(display_name, **cache_hp)
        path = _cache_path(cache_dir, file_prefix, key)
        result = _load_result(path)

        if result is not None:
            print("  → loaded from cache")
        else:
            print("  → training...")
            result = TRAIN_FN[method](**train_kwargs)
            _save_result(path, result)

        tag = "DIV" if result.diverged else f"{result.final_loss:.4f}"
        print(f"  → final_loss={tag}")
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    for r in results:
        tag = "DIV" if r.diverged else f"{r.final_loss:.4f}"
        print(f"  {r.method:20s}  lr={r.lr}  loss={tag}")
    print(f"{'='*60}")

    # Plot (uses all results — best of each method if multiple)
    by_method: dict[str, list[RunResult]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)

    best_results = [pick_best(runs) for runs in by_method.values()]
    all_runs = by_method

    if len(best_results) >= 2:
        # Find max steps for WSD overlay; use decay from longest run
        max_steps = max(len(r.losses) for r in best_results)
        plot_decay = DECAY if DECAY is not None else max_steps // 10
        plot_comparison(
            *best_results,
            n_steps=max_steps,
            warmup=WARMUP,
            decay=plot_decay,
            filename=OUTPUT,
            all_runs=all_runs,
        )
    else:
        print("Need at least 2 methods to plot comparison.")


if __name__ == "__main__":
    main()
