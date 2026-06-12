#!/usr/bin/env python3
"""Export a measured alignment table from a run's tensorboard logs.

Reads align/{z0_dW,dZ_w0,dZ_dW}/<layer> scalars from a source run (typically
``--method mup-no --measure-only``), averages each layer over the last
``--last-frac`` of training, aggregates per op-type suffix (median over
layers), and writes a JSON table consumable by ``--alignment-table``:

    {"wq": [a, w, u], "wk": [...], ..., "output": [...]}

Keys are leaf suffixes matched by Parametrization.alignment_overrides.
tok_embeddings is excluded: its alignment is never measured (fixed fan-in),
so it keeps the "no" preset.

Usage:

    python experiments/lm/export_alignment.py runs/<run_dir> -o s1_align.json
    python experiments/lm/export_alignment.py runs/<a> runs/<b> -o avg.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re

import numpy as np

METRICS = ("z0_dW", "dZ_w0", "dZ_dW")
SUFFIXES = ("wq", "wk", "wv", "wqkv", "wo", "w1", "w2", "w3", "output")


def _suffix(layer: str) -> str | None:
    if layer == "output":
        return "output"
    m = re.search(r"(wq|wk|wv|wqkv|wo|w1|w2|w3)$", layer)
    return m.group(1) if m else None


def collect(run_dir: str, last_frac: float) -> dict[tuple[str, str], list[float]]:
    """Return (metric, suffix) -> per-layer steady-state means for one run."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = glob.glob(f"{run_dir}/tb/*/events.*")
    if not events:
        raise FileNotFoundError(f"No tensorboard events under {run_dir}/tb/")

    ea = EventAccumulator(events[0], size_guidance={"scalars": 0})
    ea.Reload()
    out: dict[tuple[str, str], list[float]] = {}
    for tag in ea.Tags()["scalars"]:
        if not tag.startswith("align/"):
            continue
        _, metric, layer = tag.split("/", 2)
        suffix = _suffix(layer)
        if suffix is None:
            continue
        vals = np.array([e.value for e in ea.Scalars(tag)])
        steady = vals[int(len(vals) * (1 - last_frac)):]
        out.setdefault((metric, suffix), []).append(float(steady.mean()))
    if not out:
        raise ValueError(f"No align/* scalars in {run_dir} — was it run with --measure-only or --method maxP?")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Export alignment table from run logs")
    p.add_argument("run_dirs", nargs="+", help="Source run director(ies); multiple are pooled")
    p.add_argument("-o", "--output", required=True, help="Output JSON path")
    p.add_argument("--last-frac", type=float, default=0.25,
                   help="Fraction of training (from the end) to average over")
    args = p.parse_args()

    pooled: dict[tuple[str, str], list[float]] = {}
    for run_dir in args.run_dirs:
        for k, v in collect(run_dir, args.last_frac).items():
            pooled.setdefault(k, []).extend(v)

    table: dict[str, list[float]] = {}
    for suffix in SUFFIXES:
        if (METRICS[0], suffix) not in pooled:
            continue
        table[suffix] = [round(float(np.median(pooled[(m, suffix)])), 4) for m in METRICS]

    with open(args.output, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Wrote {args.output}:")
    for k, v in table.items():
        print(f"  {k:8s} a={v[0]:.4f}  w={v[1]:.4f}  u={v[2]:.4f}")


if __name__ == "__main__":
    main()
