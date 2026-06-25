#!/usr/bin/env python3
"""Export a measured alignment table from a vision run's metrics.json.

Reads ``align/{z0_dW,dZ_w0,dZ_dW}/<layer>`` scalars logged per row by
``train.py --measure-only`` (a ``mup-no`` source run), averages each layer over
the last ``--last-frac`` of training, and writes a JSON table consumable by
``train.py --method maxP-meas --alignment-table``:

    {"blocks.0.attn.qkv": [a, w, u], ..., "head": [...]}

Keys are EXACT PM names. The width ladder holds depth = 12 constant, so the
block names are identical at every scale and the table transfers verbatim
(down to s1, up to s3/s4). Exact-name keys also avoid the ``proj`` leaf
collision (``patch_embed.proj`` vs ``blocks.N.attn.proj``).

``patch_embed.*`` (input embedding) is excluded: its alignment is never
measured (fixed fan-in), so it keeps the "no" preset.

Usage:
    python experiments/vision/export_alignment.py runs/<run_dir> -o s2_align.json
    python experiments/vision/export_alignment.py runs/<a> runs/<b> -o avg.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("z0_dW", "dZ_w0", "dZ_dW")


def collect(run_dir: str, last_frac: float) -> dict[tuple[str, str], float]:
    """Return (metric, layer) -> steady-state mean over the last `last_frac`."""
    path = Path(run_dir) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"No metrics.json under {run_dir}")

    series: dict[tuple[str, str], list[float]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key, val in row.items():
                if not key.startswith("align/"):
                    continue
                _, metric, layer = key.split("/", 2)
                if layer.startswith("patch_embed"):
                    continue
                series[(metric, layer)].append(float(val))

    if not series:
        raise ValueError(
            f"No align/* scalars in {run_dir}/metrics.json — "
            "was it run with --measure-only?"
        )

    out: dict[tuple[str, str], float] = {}
    for key, vals in series.items():
        arr = np.asarray(vals)
        steady = arr[int(len(arr) * (1 - last_frac)):]
        out[key] = float(steady.mean())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Export alignment table from vision run logs")
    p.add_argument("run_dirs", nargs="+", help="Source run director(ies); multiple are pooled (e.g. seeds)")
    p.add_argument("-o", "--output", required=True, help="Output JSON path")
    p.add_argument("--last-frac", type=float, default=0.25,
                   help="Fraction of training (from the end) to average over")
    args = p.parse_args()

    pooled: dict[tuple[str, str], list[float]] = defaultdict(list)
    for run_dir in args.run_dirs:
        for key, val in collect(run_dir, args.last_frac).items():
            pooled[key].append(val)

    layers = sorted({layer for (_, layer) in pooled})
    table: dict[str, list[float]] = {}
    for layer in layers:
        if any((m, layer) not in pooled for m in METRICS):
            continue
        table[layer] = [round(float(np.median(pooled[(m, layer)])), 4) for m in METRICS]

    with open(args.output, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Wrote {args.output} ({len(table)} layers):")
    for k, v in table.items():
        print(f"  {k:28s} a={v[0]:.4f}  w={v[1]:.4f}  u={v[2]:.4f}")


if __name__ == "__main__":
    main()
