#!/usr/bin/env python3
"""Generate and submit SLURM jobs for vision sweeps."""

from __future__ import annotations

import argparse
import subprocess
from datetime import date
from pathlib import Path
from string import Template


# Every scale defaults to the full LR grid. The pipeline narrows s3/s4/s5
# DYNAMICALLY at runtime (coordinators release a per-arm bracket via --require
# sentinels — see pipeline.sh); no scale hardcodes a single LR, so the optimum
# is always data-driven, never assumed.
ALL_LRS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]

SCALE_CONFIGS = {
    "debug": {
        "wall": "01:00:00",
        "methods": ["mup-no"],
        "lrs": [3e-3],
        "seeds": [1],
    },
    "s1": {
        "wall": "24:00:00",
        "methods": ["mup-no", "maxP-meas"],
        "lrs": ALL_LRS,
        "seeds": [1, 2, 3],
    },
    "s2": {
        "wall": "24:00:00",
        "methods": ["mup-no", "maxP-meas"],
        "lrs": ALL_LRS,
        "seeds": [1, 2, 3],
    },
    "s3": {
        "wall": "48:00:00",
        "methods": ["mup-no", "maxP-meas"],
        "lrs": ALL_LRS,
        "seeds": [1, 2],
    },
    "s4": {
        "wall": "48:00:00",
        "methods": ["mup-no", "maxP-meas"],
        "lrs": ALL_LRS,
        "seeds": [1, 2],
        "chain": 2,
    },
    "s5": {
        "wall": "48:00:00",
        "methods": ["mup-no", "maxP-meas"],
        "lrs": ALL_LRS,
        "seeds": [1],
        "chain": 4,
    },
}


SLURM_TEMPLATE = Template(
    """\
#!/bin/bash -l
#SBATCH --job-name maxp_${scale}_${method_tag}_lr${lr_tag}_s${seed}
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 72
#SBATCH --mem-per-gpu 118GB
#SBATCH --time ${wall_time}
#SBATCH --account plgadlers-gpu-gh200
#SBATCH --partition plgrid-gpu-gh200
#SBATCH --gres gpu:1
#SBATCH --output ${output_dir}/maxp_${scale}_${method_tag}_lr${lr_tag}_s${seed}.out
#SBATCH --error  ${output_dir}/maxp_${scale}_${method_tag}_lr${lr_tag}_s${seed}.err

module add ML-bundle/25.10
source "${venv_path}/bin/activate"
cd "${repo_path}"

if [ -f "${output_dir}/final_metrics.json" ]; then
    echo "[skip] ${output_dir} already complete (final_metrics.json present)"
    exit 0
fi
${runtime_guards}
export OMP_NUM_THREADS=32
export HF_HOME="$${HF_HOME:-/net/scratch/hscra/plgrid/plgmwojnar/hf}"
export HF_HUB_DOWNLOAD_TIMEOUT="$${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="$${HF_HUB_ETAG_TIMEOUT:-120}"
export WANDB_PROJECT="$${WANDB_PROJECT:-maxP-vision}"
export WANDB_RUN_NAME="maxp_${scale}_${method_tag}_lr${lr_tag}_s${seed}"

python experiments/vision/train.py \\
  --scale ${scale} \\
  --method ${method} \\
  --lr ${lr} \\
  --seed ${seed} \\
  --dataset ${dataset} \\
  --epochs ${epochs} \\
  --batch-size ${batch_size} \\
  --num-workers ${num_workers} \\
  --val-interval ${val_interval} \\
  --val-steps ${val_steps} \\
  --keep-latest-k ${keep_latest_k} \\
  --output-dir ${output_dir} \\
  ${resume_arg} ${method_args} ${extra_train_args}
"""
)


def _lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("+", "")


def _method_tag(method: str) -> str:
    return method.replace("-", "")


def _is_complete(out_dir: Path) -> bool:
    """A run is complete once train.py has written final_metrics.json."""
    return (out_dir / "final_metrics.json").is_file()


def main():
    parser = argparse.ArgumentParser(
        description="Launch maxP timm vision sweeps via SLURM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scale", choices=list(SCALE_CONFIGS), required=True)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--lrs", type=float, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)

    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--venv-path", required=True)
    parser.add_argument("--repo-path", required=True)

    parser.add_argument("--dataset", default="imagenet12k")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Epochs — identical at every width (fixed sample budget, muTransfer setup)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--val-interval", type=int, default=500,
                        help="Run validation every N steps (must be a multiple of --log-interval=20)")
    parser.add_argument("--val-steps", type=int, default=50,
                        help="Batches per validation pass (keep small — eval compute ≈ train compute)")

    parser.add_argument("--alignment-table", default=None,
                        help="JSON table for maxP-meas runs (from export_alignment.py)")
    parser.add_argument("--measure-only", action="store_true",
                        help="Append --measure-only (mup-no alignment source run)")
    parser.add_argument("--tag", default=None,
                        help="Suffix appended to run names (e.g. 'meas', 'transfer-s2')")
    parser.add_argument("--job-ids-file", default=None,
                        help="Append submitted SLURM job IDs (one per line) for downstream deps")
    parser.add_argument("--dependency", default=None,
                        help="sbatch --dependency spec applied to each run's first link")
    parser.add_argument("--require", action="append", default=[],
                        help="File that must exist at job runtime or the job exits cleanly; "
                             "'{lr}' is replaced by the run's lr tag. Repeatable.")
    parser.add_argument("--chain", type=int, default=None,
                        help="Number of resume-chain links per run (default: per-scale config). "
                             "Links run sequentially via --dependency=afterany; each resumes from "
                             "the latest checkpoint, and a completed run no-ops via the final_metrics guard.")
    parser.add_argument("--extra-train-args", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Force resubmit even if final_metrics.json already exists")
    args = parser.parse_args()

    sc = SCALE_CONFIGS[args.scale]
    methods = args.methods or sc["methods"]
    lrs = args.lrs or sc["lrs"]
    seeds = args.seeds or sc["seeds"]

    if "maxP-meas" in methods and not args.measure_only and not args.alignment_table:
        raise SystemExit("maxP-meas runs require --alignment-table (from export_alignment.py)")

    chain = args.chain if args.chain is not None else sc.get("chain", 1)
    if chain > 1 and args.measure_only:
        raise SystemExit("--measure-only requires chain==1: resume re-snapshots z0/w0 and "
                         "corrupts alignment measurement (use full-budget single jobs to measure)")
    
    today = date.today().strftime("%Y-%m-%d")
    submitted = skipped = 0

    for method in methods:
        method_args = ""
        if method == "maxP-meas":
            method_args = f"--alignment-table {args.alignment_table}"
        elif args.measure_only:
            method_args = "--measure-only"

        for lr in lrs:
            for seed in seeds:
                run_name = f"{today}_{args.scale}_{_method_tag(method)}_lr{_lr_tag(lr)}_s{seed}"
                if args.tag:
                    run_name += f"_{args.tag}"
                out_dir = Path(args.runs_dir) / run_name

                if not args.resume and _is_complete(out_dir):
                    print(f"[skip]   {run_name}")
                    skipped += 1
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)

                # Runtime gating: jobs can be pre-submitted before the decision
                # that enables them exists; '{lr}' resolves to this run's lr tag.
                guards = []
                for path in args.require:
                    path = path.replace("{lr}", _lr_tag(lr))
                    guards.append(
                        f'if [ ! -f "{path}" ]; then '
                        f'echo "gate: {path} missing — skipping."; exit 0; fi')
                runtime_guards = ("\n" + "\n".join(guards)) if guards else ""

                script = SLURM_TEMPLATE.substitute(
                    scale=args.scale,
                    method_tag=_method_tag(method),
                    lr_tag=_lr_tag(lr),
                    seed=seed,
                    wall_time=sc["wall"],
                    output_dir=str(out_dir),
                    venv_path=args.venv_path,
                    repo_path=args.repo_path,
                    method=method,
                    lr=lr,
                    dataset=args.dataset,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    val_interval=args.val_interval,
                    val_steps=args.val_steps,
                    keep_latest_k=1,
                    resume_arg="--resume" if chain > 1 else "",
                    method_args=method_args,
                    extra_train_args=args.extra_train_args,
                    runtime_guards=runtime_guards,
                )
                script_path = out_dir / "job.sh"
                script_path.write_text(script)

                prev = None
                for link in range(chain):
                    if prev is not None:
                        dep_arg = f"--dependency=afterany:{prev}"
                    elif args.dependency:
                        dep_arg = f"--dependency={args.dependency}"
                    else:
                        dep_arg = None
                    if args.dry_run:
                        suffix = f" ({dep_arg})" if dep_arg else ""
                        print(f"[dry]    sbatch link {link + 1}/{chain} {script_path}{suffix}")
                        prev = f"<job{link}>"
                        continue
                    cmd = ["sbatch"]
                    if dep_arg:
                        cmd.append(dep_arg)
                    cmd.append(str(script_path))
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise SystemExit(
                            f"sbatch failed for {run_name} (link {link + 1}/{chain}): "
                            f"{result.stderr.strip()}"
                        )
                    prev = result.stdout.strip().split()[-1]
                    if args.job_ids_file:
                        with open(args.job_ids_file, "a") as f:
                            f.write(f"{prev}\n")
                    print(f"[submit] {run_name}  link {link + 1}/{chain}  →  job {prev}")
                submitted += 1

    action = "would submit" if args.dry_run else "submitted"
    print(f"\nDone: {action} {submitted} jobs, skipped {skipped} completed runs.")


if __name__ == "__main__":
    main()
