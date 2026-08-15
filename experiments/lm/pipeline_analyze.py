#!/usr/bin/env python3
"""Decision helpers for the unattended pipeline (run on the login node).

Subcommands:
    best-lr  --glob 'runs/*_s2_mupno_lr*_meas'
        Picks the LR with lowest final loss (median over seeds).
        Prints ONLY the lr tag (e.g. "1e-02") on stdout; details on stderr.

    e1  --baseline-glob G1 --transfer-glob G2 [--tol 0.02]
        Compares best (over LRs, median over seeds) final loss.
        Exit 0 if transfer <= baseline + tol (continue to s4/s5), else 1.

    gates  --runs-glob 'runs/*_s2_mupno_lr{BEST}_s*_meas' --adjacent-glob '...'
        Seed-stability and source-LR-stability of exported alignment tables.
        Report on stdout, always exit 0 (warnings only).
"""

from __future__ import annotations

import argparse
import glob as globlib
import re
import sys

import numpy as np


def _final_loss(run_dir: str, last_k: int = 5) -> float | None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = globlib.glob(f"{run_dir}/tb/*/events.*")
    if not events:
        return None
    ea = EventAccumulator(events[0], size_guidance={"scalars": 0})
    ea.Reload()
    tags = [t for t in ea.Tags()["scalars"] if "loss" in t and "validation" not in t]
    if not tags:
        return None
    tag = next((t for t in tags if "global_avg" in t), tags[0])
    vals = [e.value for e in ea.Scalars(tag)]
    return float(np.mean(vals[-last_k:])) if vals else None


def _by_lr(pattern: str) -> dict[str, list[float]]:
    """lr tag -> final losses (one per seed/run)."""
    out: dict[str, list[float]] = {}
    for run_dir in sorted(globlib.glob(pattern)):
        m = re.search(r"_lr([^_]+)_", run_dir)
        if not m:
            continue
        loss = _final_loss(run_dir)
        if loss is None or not np.isfinite(loss):
            print(f"  [warn] no usable loss in {run_dir}", file=sys.stderr)
            continue
        out.setdefault(m.group(1), []).append(loss)
    if not out:
        raise SystemExit(f"No usable runs match {pattern}")
    return out


def cmd_best_lr(args) -> None:
    by_lr = _by_lr(args.glob)
    med = {lr: float(np.median(v)) for lr, v in by_lr.items()}
    best = min(med, key=med.get)
    for lr in sorted(med, key=lambda t: float(t)):
        mark = " <-- best" if lr == best else ""
        print(f"  lr {lr}: median {med[lr]:.4f} over {len(by_lr[lr])} run(s){mark}",
              file=sys.stderr)
    tags = sorted(med, key=lambda t: float(t))
    if best in (tags[0], tags[-1]):
        print(f"  [warn] best LR {best} is at the grid edge — optimum may lie outside",
              file=sys.stderr)
    print(best)


def cmd_e1(args) -> None:
    base = {lr: float(np.median(v)) for lr, v in _by_lr(args.baseline_glob).items()}
    tran = {lr: float(np.median(v)) for lr, v in _by_lr(args.transfer_glob).items()}
    b_lr, b = min(base.items(), key=lambda kv: kv[1])
    t_lr, t = min(tran.items(), key=lambda kv: kv[1])
    print(f"E1 verdict: baseline best {b:.4f} @ lr {b_lr} | transfer best {t:.4f} @ lr {t_lr}")
    if t <= b + args.tol:
        print(f"PASS (transfer within tol {args.tol} of baseline or better) — continue to s4/s5")
    else:
        print(f"FAIL (transfer worse by {t - b:.4f} > tol {args.tol}) — stopping before s4/s5")
        raise SystemExit(1)


def _table_from_runs(pattern: str) -> dict[str, list[float]]:
    from export_alignment import collect, METRICS, SUFFIXES

    pooled: dict[tuple[str, str], list[float]] = {}
    for run_dir in sorted(globlib.glob(pattern)):
        for k, v in collect(run_dir, last_frac=0.25).items():
            pooled.setdefault(k, []).extend(v)
    return {s: [float(np.median(pooled[(m, s)])) for m in METRICS]
            for s in SUFFIXES if (METRICS[0], s) in pooled}


def cmd_gates(args) -> None:
    runs = sorted(globlib.glob(args.runs_glob))
    print(f"Gate report — source runs: {[r.split('/')[-1] for r in runs]}")
    pooled = _table_from_runs(args.runs_glob)

    # G1: per-seed tables vs pooled
    worst = 0.0
    for r in runs:
        t = _table_from_runs(r)
        for k in t:
            for i in range(3):
                worst = max(worst, abs(t[k][i] - pooled[k][i]))
    status = "OK" if worst < args.warn else "WARN — noisy estimator, check per-seed tables"
    print(f"G1 seed stability: max |alignment delta| seed-vs-pooled = {worst:.4f}  [{status}]")

    # G2: adjacent source LR vs best
    if args.adjacent_glob and globlib.glob(args.adjacent_glob):
        adj = _table_from_runs(args.adjacent_glob)
        worst = max(abs(adj[k][i] - pooled[k][i])
                    for k in set(adj) & set(pooled) for i in range(3))
        status = "OK" if worst < args.warn else "WARN — source-LR sensitive"
        print(f"G2 source-LR stability: max |alignment delta| = {worst:.4f}  [{status}]")
    else:
        print("G2 source-LR stability: skipped (no adjacent-LR runs)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("best-lr"); b.add_argument("--glob", required=True)
    b.set_defaults(fn=cmd_best_lr)

    e = sub.add_parser("e1")
    e.add_argument("--baseline-glob", required=True)
    e.add_argument("--transfer-glob", required=True)
    e.add_argument("--tol", type=float, default=0.02)
    e.set_defaults(fn=cmd_e1)

    g = sub.add_parser("gates")
    g.add_argument("--runs-glob", required=True)
    g.add_argument("--adjacent-glob", default=None)
    g.add_argument("--warn", type=float, default=0.05)
    g.set_defaults(fn=cmd_gates)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
