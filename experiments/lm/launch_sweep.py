#!/usr/bin/env python3
"""Generate and submit SLURM jobs for the full LM sweep.

Usage:

    python experiments/lm/launch_sweep.py \\
        --scale s3 \\
        --methods mup-no \\
        --lrs 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 \\
        --seeds 1 2 \\
        --runs-dir /net/storage/pr3/plgrid/plggadlers/maxP/runs \\
        --dataset HuggingFaceFW/fineweb-edu \\
        --hf-assets-path /net/storage/pr3/plgrid/plggadlers/maxP/experiments/lm/assets/hf/Llama-3.1-8B \\
        --venv-path /net/storage/pr3/plgrid/plggadlers/maxP/.venv \\
        --repo-path /net/storage/pr3/plgrid/plggadlers/maxP \\
        [--dry-run] [--resume]

By default skips runs where any checkpoint already exists (started or completed).
With --resume, submits all runs regardless (to resume interrupted training).
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import date
from pathlib import Path
from string import Template


# ---------------------------------------------------------------------------
# Per-scale defaults
# ---------------------------------------------------------------------------

# Wall times: total train FLOPs at 30% MFU on GH200 (494 TFLOPS BF16 dense
# -> ~148 TF/GPU effective), x1.5 safety margin, rounded up:
#   s1 0.2h | s2 1.7h | s3 18.4h | s4 (4 GPU) 55h | s5 (4 GPU) ~730h
# The cluster caps wall time at 48h, so s4/s5 checkpoint periodically and are
# submitted as a chain of dependent jobs ("chain" below): each job resumes
# from the latest checkpoint and exits when the wall limit hits; the next in
# the chain continues. torchtitan auto-resumes when a checkpoint exists.
SCALE_CONFIGS = {
    "s1": {"wall": "01:00:00", "nodes": 1, "gpus": 1, "ckpt": None, "chain": 1},
    "s2": {"wall": "03:00:00", "nodes": 1, "gpus": 1, "ckpt": None, "chain": 1},
    "s3": {"wall": "24:00:00", "nodes": 1, "gpus": 1, "ckpt": None, "chain": 1},
    "s4": {"wall": "48:00:00", "nodes": 1, "gpus": 4, "ckpt": 5000, "chain": 2},
    "s5": {"wall": "48:00:00", "nodes": 1, "gpus": 4, "ckpt": 2500, "chain": 12},
}

SCALE_METHODS = {
    "s1": ["mup-no"],
    "s2": ["mup-no"],
    "s3": ["mup-no"],
    "s4": ["mup-no"],
    "s5": ["mup-no"],
}

SCALE_SEEDS = {
    "s1": [1, 2, 3],
    "s2": [1, 2, 3],
    "s3": [1, 2],
    "s4": [1, 2],
    "s5": [1],
}

# 7-point grid at half-decade spacing; architecture-agnostic default for
# every scale. Large-scale sweeps should pass an explicit --lrs subset
# (the pipeline gates per-LR via --require sentinels instead).
ALL_LRS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0]


# ---------------------------------------------------------------------------
# SLURM job script template
# ---------------------------------------------------------------------------

SLURM_TEMPLATE = Template("""\
#!/bin/bash -l
#SBATCH --job-name maxp_${scale}_${method_tag}_lr${lr_tag}_s${seed}
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 72
#SBATCH --mem-per-gpu 118GB
#SBATCH --time ${wall_time}
#SBATCH --account plgadlers-gpu-gh200
#SBATCH --partition plgrid-gpu-gh200
#SBATCH --gres gpu:${gpus_per_node}
#SBATCH --output ${output_dir}/maxp_${scale}_${method_tag}_lr${lr_tag}_s${seed}.out
#SBATCH --error  ${output_dir}/maxp_${scale}_${method_tag}_lr${lr_tag}_s${seed}.err

module add ML-bundle/25.10
source "${venv_path}/bin/activate"
cd "${repo_path}"

export OMP_NUM_THREADS=16
export HF_HOME="$${HF_HOME:-/net/scratch/hscra/plgrid/plgmwojnar/hf}"
export HF_HUB_DOWNLOAD_TIMEOUT="$${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="$${HF_HUB_ETAG_TIMEOUT:-120}"
export WANDB_PROJECT="$${WANDB_PROJECT:-maxP-lm}"
export WANDB_RUN_NAME="maxp_${scale}_${method_tag}_lr${lr_tag}_s${seed}"

# Distributed setup
find_free_port() {
    python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$$(find_free_port)
export NCCL_SOCKET_IFNAME=lo

echo "Master Port: $${MASTER_PORT}"

# Chained jobs: skip if an earlier link already finished this run.
if [ -f "${output_dir}/COMPLETED" ]; then
    echo "Run already completed — skipping."
    exit 0
fi
${runtime_guards}
torchrun --nproc_per_node=${gpus_per_node} --master_addr=$${MASTER_ADDR} --master_port=$${MASTER_PORT} experiments/lm/train.py \\
    --scale ${scale} \\
    --method ${method} \\
    --lr ${lr} \\
    --seed ${seed} \\
    --batch-size ${batch_size} \\
    --dataset ${dataset} \\
    ${dataset_path_arg} \\
    --num-workers ${num_workers} \\
    --prefetch-factor ${prefetch_factor} \\
    --hf-assets-path ${hf_assets_path} \\
    ${extra_args} \\
    --output-dir ${output_dir}
""")


def _lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("+", "")


def _method_tag(method: str) -> str:
    return method.replace("-", "")


def _has_checkpoint(out_dir: Path) -> bool:
    ckpt_dir = out_dir / "checkpoint"
    return ckpt_dir.is_dir() and any(ckpt_dir.iterdir())


def main() -> None:
    p = argparse.ArgumentParser(description="Launch maxP LLaMA-3 SLURM sweep")
    p.add_argument("--scale", required=True, choices=list(SCALE_CONFIGS))
    p.add_argument("--methods", nargs="+", default=None,
                   help="Override method list (default: per-scale defaults)")
    p.add_argument("--lrs", type=float, nargs="+", default=None,
                   help="Override LR list (default: all 7 values)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Override seed list (default: per-scale defaults)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Global batch size (divided by WORLD_SIZE to get per-GPU)")
    p.add_argument("--gpus-per-node", type=int, default=None,
                   help="GPUs per SLURM node (default: per-scale value from SCALE_CONFIGS)")
    p.add_argument("--runs-dir", required=True, help="Root directory for run outputs")
    p.add_argument("--dataset", default="fineweb-edu",
                   help="HuggingFace dataset name (default: per-scale defaults)")
    p.add_argument("--dataset-path", default=None,
                   help="Optional local path to dataset assets (otherwise streamed from HF)")
    p.add_argument("--num-workers", type=int, default=8,
                   help="DataLoader num_workers for prefetching")
    p.add_argument("--prefetch-factor", type=int, default=4,
                   help="Batches prefetched per DataLoader worker")
    p.add_argument("--hf-assets-path", required=True,
                   help="Path to local HF tokenizer assets directory")
    p.add_argument("--venv-path", required=True, help="Path to Python venv")
    p.add_argument("--repo-path", required=True, help="Path to maxP repo root")
    p.add_argument("--measure-only", action="store_true",
                   help="Pass --measure-only to train.py (alignment source runs)")
    p.add_argument("--alignment-table", default=None,
                   help="Pass --alignment-table to train.py (maxP-meas runs)")
    p.add_argument("--tag", default=None,
                   help="Suffix appended to run names (e.g. 'meas', 'transfer-s1')")
    p.add_argument("--chain", type=int, default=None,
                   help="Submit N dependent jobs per run (default: per-scale value); "
                        "later jobs resume from the latest checkpoint")
    p.add_argument("--job-ids-file", default=None,
                   help="Append submitted SLURM job IDs (one per line) for "
                        "downstream dependency wiring")
    p.add_argument("--dependency", default=None,
                   help="sbatch --dependency spec applied to each run's first "
                        "job (e.g. 'afterany:123:456')")
    p.add_argument("--require", action="append", default=[],
                   help="File that must exist at job runtime or the job exits "
                        "cleanly; '{lr}' is replaced by the run's lr tag. "
                        "Repeatable.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print sbatch commands without submitting")
    p.add_argument("--resume", action="store_true",
                   help="Submit all runs even if a checkpoint exists (resume interrupted training)")
    args = p.parse_args()

    scale = args.scale
    sc = SCALE_CONFIGS[scale]
    methods = args.methods or SCALE_METHODS[scale]
    if "maxP-meas" in methods and not args.alignment_table:
        p.error("--methods maxP-meas requires --alignment-table")
    lrs = args.lrs or ALL_LRS
    seeds = args.seeds or SCALE_SEEDS[scale]
    chain = args.chain if args.chain is not None else sc["chain"]
    if chain > 1 and (args.measure_only or "maxP" in methods):
        p.error("--measure-only and dynamic maxP require runs that finish in "
                "one job (chain=1): resuming re-snapshots z0/w0 from the "
                "checkpoint and corrupts alignment measurement")
    gpus_per_node = args.gpus_per_node if args.gpus_per_node is not None else sc["gpus"]

    today = date.today().strftime("%Y-%m-%d")
    submitted = skipped = 0

    for method in methods:
        for lr in lrs:
            for seed in seeds:
                run_name = (
                    f"{today}_{scale}_{_method_tag(method)}_lr{_lr_tag(lr)}_s{seed}"
                )
                if args.tag:
                    run_name += f"_{args.tag}"
                out_dir = Path(args.runs_dir) / run_name

                if not args.resume and _has_checkpoint(out_dir):
                    print(f"[skip]   {run_name}")
                    skipped += 1
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)

                dataset_path_arg = f"--dataset-path {args.dataset_path}" if args.dataset_path else ""

                extra = []
                if args.measure_only:
                    extra.append("--measure-only")
                if args.alignment_table:
                    extra.append(f"--alignment-table {args.alignment_table}")
                if sc["ckpt"]:
                    extra.append(f"--checkpoint-interval {sc['ckpt']}")
                extra_args = " ".join(extra)

                # Runtime gating: lets jobs be pre-submitted before the
                # decision that enables them exists (login node submits only).
                guards = []
                for path in args.require:
                    path = path.replace("{lr}", _lr_tag(lr))
                    guards.append(
                        f'if [ ! -f "{path}" ]; then '
                        f'echo "gate: {path} missing — skipping."; exit 0; fi')
                runtime_guards = ("\n" + "\n".join(guards) + "\n") if guards else ""

                script_content = SLURM_TEMPLATE.substitute(
                    scale=scale,
                    method_tag=_method_tag(method),
                    lr_tag=_lr_tag(lr),
                    seed=seed,
                    wall_time=sc["wall"],
                    output_dir=str(out_dir),
                    venv_path=args.venv_path,
                    repo_path=args.repo_path,
                    method=method,
                    lr=lr,
                    batch_size=args.batch_size,
                    gpus_per_node=gpus_per_node,
                    dataset=args.dataset,
                    dataset_path_arg=dataset_path_arg,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                    hf_assets_path=args.hf_assets_path,
                    extra_args=extra_args,
                    runtime_guards=runtime_guards,
                )
                script_path = out_dir / "job.sh"
                script_path.write_text(script_content)

                prev_job_id = None
                for link in range(chain):
                    if prev_job_id:
                        dep = ["--dependency", f"afterany:{prev_job_id}"]
                    elif args.dependency:
                        dep = ["--dependency", args.dependency]
                    else:
                        dep = []
                    if args.dry_run:
                        dep_str = f" {' '.join(dep)}" if dep else ""
                        print(f"[dry]    sbatch{dep_str} {script_path}")
                        continue
                    result = subprocess.run(
                        ["sbatch", *dep, str(script_path)],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        raise SystemExit(
                            f"sbatch failed for {run_name} "
                            f"(link {link + 1}/{chain}): {result.stderr.strip()}")
                    out = result.stdout.strip()
                    prev_job_id = out.split()[-1] if out else None
                    if args.job_ids_file and prev_job_id:
                        with open(args.job_ids_file, "a") as f:
                            f.write(f"{prev_job_id}\n")
                    chain_tag = f" [chain {link + 1}/{chain}]" if chain > 1 else ""
                    print(f"[submit] {run_name}{chain_tag}  →  {out}")
                submitted += 1

    action = "would submit" if args.dry_run else "submitted"
    print(f"\nDone: {action} {submitted} jobs, skipped {skipped} completed runs.")


if __name__ == "__main__":
    main()
