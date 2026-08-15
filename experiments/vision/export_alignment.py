#!/usr/bin/env python3
"""Export a measured alignment table from a vision run's metrics.json.

Reads ``align/{z0_dW,dZ_w0,dZ_dW}/<layer>`` scalars logged per row by
``train.py --measure-only`` (a ``mup-no`` source run), averages each layer over
the last ``--last-frac`` of training, then **groups by op-type across depth**
(median over the 12 blocks) and writes a JSON table consumable by
``train.py --method maxP-meas --alignment-table``:

    {"wq": [a, w, u], "wk": [...], ..., "w3": [...], "head": [...]}

Keys are leaf-type suffixes (matched by ``Parametrization.alignment_overrides``
via its leaf fallback), exactly mirroring the LM experiment's grouping. Pooling
per type kills the per-layer measurement noise that an exact-name table carries.

``patch_embed.*`` (input embedding) is excluded: its alignment is never
measured (fixed fan-in), so it keeps the "no" preset. This is the analog of the
LM exporter excluding ``tok_embeddings``; the readout ``head`` IS included.

Usage:
    python experiments/vision/export_alignment.py runs/<run_dir> -o s2_align.json
    python experiments/vision/export_alignment.py runs/<a> runs/<b> -o avg.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import re

import numpy as np

METRICS = ("z0_dW", "dZ_w0", "dZ_dW")
SUFFIXES = ("wq", "wk", "wv", "wo", "w1", "w2", "w3", "head")


def _suffix(layer: str) -> str | None:
    """Map an exact PM name to its LM-style op-type group, or None to drop."""
    if layer == "head" or layer.endswith(".head"):
        return "head"
    m = re.search(r"(wq|wk|wv|wo|w1|w2|w3)$", layer)
    return m.group(1) if m else None


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

    # Pool by (metric, op-type) across BOTH depth (12 blocks) and run dirs/seeds.
    pooled: dict[tuple[str, str], list[float]] = defaultdict(list)
    for run_dir in args.run_dirs:
        try:
            collected = collect(run_dir, args.last_frac)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [warn] skipping {run_dir}: {e}")
            continue
        for (metric, layer), val in collected.items():
            suffix = _suffix(layer)
            if suffix is None:
                continue
            pooled[(metric, suffix)].append(val)
    if not pooled:
        raise SystemExit("No usable align/* metrics in any run dir")

    table: dict[str, list[float]] = {}
    for suffix in SUFFIXES:
        if any((m, suffix) not in pooled for m in METRICS):
            continue
        table[suffix] = [round(float(np.median(pooled[(m, suffix)])), 4) for m in METRICS]

    with open(args.output, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Wrote {args.output} ({len(table)} op-types):")
    for k, v in table.items():
        print(f"  {k:6s} a={v[0]:.4f}  w={v[1]:.4f}  u={v[2]:.4f}")


if __name__ == "__main__":
    main()
