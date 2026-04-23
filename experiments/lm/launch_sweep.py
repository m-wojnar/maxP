#!/usr/bin/env python3
"""Generate and submit SLURM jobs for the full LM sweep.

Usage:

    python experiments/lm/launch_sweep.py \\
        --scale s3 \\
        --methods maxP mup-full mup-no \\
        --lrs 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 \\
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

SCALE_CONFIGS = {
    "s1": {"wall": "02:00:00", "nodes": 1},
    "s2": {"wall": "02:00:00", "nodes": 1},
    "s3": {"wall": "06:00:00", "nodes": 1},
    "s4": {"wall": "24:00:00", "nodes": 1},
    "s5": {"wall": "48:00:00", "nodes": 1},
}

# Supported methods (currently implemented)
SCALE_METHODS = {
    "s1": ["maxP", "mup-full", "mup-no"],
    "s2": ["maxP", "mup-full", "mup-no"],
    "s3": ["maxP", "mup-full", "mup-no"],
    "s4": ["maxP", "mup-full", "mup-no"],
    "s5": ["maxP", "mup-full", "mup-no"],
}

SCALE_SEEDS = {
    "s1": [1, 2, 3],
    "s2": [1, 2],
    "s3": [1, 2],
    "s4": [1],
    "s5": [1],
}

# Full LR grid from experiments.md §4.1
ALL_LRS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]

S5_SINGLE_LR = [1e-2]  # transfer test: single best LR from S4


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

export OMP_NUM_THREADS=8
export HF_HOME="$${HF_HOME:-/net/storage/pr3/plgrid/plggadlers/hf_cache}"
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

torchrun --nproc_per_node=${gpus_per_node} --master_addr=$${MASTER_ADDR} --master_port=$${MASTER_PORT} experiments/lm/train.py \\
    --scale ${scale} \\
    --method ${method} \\
    --lr ${lr} \\
    --seed ${seed} \\
    --batch-size ${batch_size} \\
    --dataset ${dataset} \\
    --hf-assets-path ${hf_assets_path} \\
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
    p.add_argument("--batch-size", type=int, default=8,
                   help="Local batch size per GPU")
    p.add_argument("--gpus-per-node", type=int, default=1,
                   help="GPUs per SLURM node (also sets --nproc_per_node)")
    p.add_argument("--runs-dir", required=True, help="Root directory for run outputs")
    p.add_argument("--dataset", required=True,
                   help="HuggingFace dataset name (e.g. HuggingFaceFW/fineweb-edu)")
    p.add_argument("--hf-assets-path", required=True,
                   help="Path to local HF tokenizer assets directory")
    p.add_argument("--venv-path", required=True, help="Path to Python venv")
    p.add_argument("--repo-path", required=True, help="Path to maxP repo root")
    p.add_argument("--dry-run", action="store_true",
                   help="Print sbatch commands without submitting")
    p.add_argument("--resume", action="store_true",
                   help="Submit all runs even if a checkpoint exists (resume interrupted training)")
    args = p.parse_args()

    scale = args.scale
    sc = SCALE_CONFIGS[scale]
    methods = args.methods or SCALE_METHODS[scale]
    lrs = args.lrs or (S5_SINGLE_LR if scale == "s5" else ALL_LRS)
    seeds = args.seeds or SCALE_SEEDS[scale]

    today = date.today().strftime("%Y-%m-%d")
    submitted = skipped = 0

    for method in methods:
        for lr in lrs:
            for seed in seeds:
                run_name = (
                    f"{today}_{scale}_{_method_tag(method)}_lr{_lr_tag(lr)}_s{seed}"
                )
                out_dir = Path(args.runs_dir) / run_name

                if not args.resume and _has_checkpoint(out_dir):
                    print(f"[skip]   {run_name}")
                    skipped += 1
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)

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
                    gpus_per_node=args.gpus_per_node,
                    dataset=args.dataset,
                    hf_assets_path=args.hf_assets_path,
                )
                script_path = out_dir / "job.sh"
                script_path.write_text(script_content)

                if args.dry_run:
                    print(f"[dry]    sbatch {script_path}")
                else:
                    result = subprocess.run(
                        ["sbatch", str(script_path)],
                        capture_output=True,
                        text=True,
                    )
                    print(f"[submit] {run_name}  →  {result.stdout.strip()}")
                submitted += 1

    action = "would submit" if args.dry_run else "submitted"
    print(f"\nDone: {action} {submitted} jobs, skipped {skipped} completed runs.")


if __name__ == "__main__":
    main()
