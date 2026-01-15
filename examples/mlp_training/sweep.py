#!/usr/bin/env python3
"""
Sweep runner for parallel hyperparameter search using worker-based parallelism.

Each worker processes a subset of the grid based on WORKER_ID and N_WORKERS.
Use run_sweep.sh to launch multiple workers in parallel.

Usage:
    # Single worker (runs all configurations)
    python sweep.py --config sweep_config.yaml

    # With environment variables (typically set by run_sweep.sh)
    WORKER_ID=0 N_WORKERS=4 python sweep.py --config sweep_config.yaml

    # Dry run (just print configurations)
    python sweep.py --config sweep_config.yaml --dry-run
"""

import argparse
import copy
import itertools
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from train import train


def set_nested_value(d: dict, key: str, value: Any) -> None:
    """Set a nested dictionary value using dot notation."""
    keys = key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def generate_run_name(params: dict) -> str:
    """Generate a descriptive run name from parameters."""
    parts = []
    for key, value in sorted(params.items()):
        short_key = key.split(".")[-1]  # Use last part of dotted key
        if isinstance(value, float):
            parts.append(f"{short_key}={value:.0e}")
        else:
            parts.append(f"{short_key}={value}")
    return "_".join(parts)


def generate_sweep_grid(sweep_config: dict, script_dir: Path) -> list[tuple]:
    """Generate all (run_id, run_name, config, sweep_dict) tuples from sweep specification."""
    # Load base config
    base_config_path = script_dir / sweep_config["base_config"]
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    # Get parameter grids
    sweep_params = sweep_config.get("sweep", {})
    overrides = sweep_config.get("overrides", {})

    # Generate all combinations
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[name] for name in param_names]

    grid = []
    for run_id, values in enumerate(itertools.product(*param_values)):
        # Create config copy
        config = copy.deepcopy(base_config)

        # Apply sweep parameters
        sweep_dict = {}
        for name, value in zip(param_names, values):
            set_nested_value(config, name, value)
            sweep_dict[name] = value

        # Apply fixed overrides
        for key, value in overrides.items():
            set_nested_value(config, key, value)

        # Generate run name and output directory
        run_name = generate_run_name(sweep_dict)
        output_dir = Path(sweep_config["output_base"]) / sweep_config["experiment_name"] / run_name
        set_nested_value(config, "logging.output_dir", str(output_dir))

        grid.append((run_id, run_name, config, sweep_dict))

    return grid


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="sweep_config.yaml")
    args.add_argument("--dry-run", action="store_true", default=False)
    args = args.parse_args()

    # Get worker info from environment
    worker_id = int(os.environ.get("WORKER_ID", 0))
    n_workers = int(os.environ.get("N_WORKERS", 1))

    script_dir = Path(args.config).parent.resolve()

    # Load sweep config
    with open(args.config) as f:
        sweep_config = yaml.safe_load(f)

    # Generate grid
    grid = generate_sweep_grid(sweep_config, script_dir)
    total_runs = len(grid)

    print(f"Generated {total_runs} configurations")
    print(f"Worker {worker_id}/{n_workers} will run {len([r for r in grid if r[0] % n_workers == worker_id])} runs")

    if args.dry_run:
        print("\n=== DRY RUN ===\n")
        for run_id, run_name, config, sweep_dict in grid:
            assigned_to = run_id % n_workers
            marker = " <-- this worker" if assigned_to == worker_id else ""
            print(f"[{run_id:3d}] {run_name}")
            print(f"      Params: {sweep_dict}")
            print(f"      Worker: {assigned_to}{marker}")
            print()
        return

    # Create output directory for job configs
    job_dir = script_dir / "jobs" / sweep_config["experiment_name"]
    job_dir.mkdir(parents=True, exist_ok=True)

    # Run assigned configurations
    for run_id, run_name, config, sweep_dict in grid:
        if run_id % n_workers != worker_id:
            continue

        print(f"\n{'='*60}")
        print(f"Starting {run_name} (id={run_id}) on worker {worker_id}...")
        print(f"Params: {sweep_dict}")
        print(f"{'='*60}\n")

        # Save config for reference
        config_path = job_dir / f"{run_name}_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        t0 = time.time()
        try:
            sys.path.insert(0, str(script_dir))
            train(config)
            status = "completed"
        except Exception as e:
            print(f"ERROR in {run_name}: {e}")
            status = "failed"
        tf = time.time()

        print(f"\nExperiment {run_name} (id={run_id}) {status} in {tf - t0:.2f}s by worker {worker_id}.")

    print(f"\nWorker {worker_id} finished all assigned runs.")


if __name__ == "__main__":
    main()
