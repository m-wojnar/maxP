#!/usr/bin/env python3
"""Generate and submit SLURM jobs for vision sweeps."""

from __future__ import annotations

import argparse
import subprocess
from datetime import date
from pathlib import Path
from string import Template


ALL_LRS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
TRANSFER_LR = [1e-2]

SCALE_CONFIGS = {
    "debug": {
        "wall": "01:00:00",
        "gpus": 1,
        "methods": ["maxP"],
        "lrs": [3e-3],
        "seeds": [1],
        "image_size": 224,
        "drop_path_rate": 0.0,
    },
    "vit-s": {
        "wall": "24:00:00",
        "gpus": 1,
        "methods": ["maxP", "mup-full", "mup-no"],
        "lrs": ALL_LRS,
        "seeds": [1, 2, 3],
        "image_size": 224,
        "drop_path_rate": 0.1,
    },
    "vit-b": {
        "wall": "36:00:00",
        "gpus": 1,
        "methods": ["maxP", "mup-full", "mup-no"],
        "lrs": TRANSFER_LR,
        "seeds": [1],
        "image_size": 224,
        "drop_path_rate": 0.2,
    },
    "vit-l": {
        "wall": "48:00:00",
        "gpus": 1,
        "methods": ["maxP", "mup-full", "mup-no"],
        "lrs": TRANSFER_LR,
        "seeds": [1],
        "image_size": 224,
        "drop_path_rate": 0.4,
    },
    "cnx-t": {
        "wall": "24:00:00",
        "gpus": 1,
        "methods": ["maxP", "mup-full", "mup-no"],
        "lrs": ALL_LRS,
        "seeds": [1, 2, 3],
        "image_size": 224,
        "drop_path_rate": 0.1,
    },
    "cnx-s": {
        "wall": "36:00:00",
        "gpus": 1,
        "methods": ["maxP", "mup-full", "mup-no"],
        "lrs": ALL_LRS,
        "seeds": [1],
        "image_size": 224,
        "drop_path_rate": 0.2,
    },
    "cnx-b": {
        "wall": "48:00:00",
        "gpus": 1,
        "methods": ["maxP", "mup-full", "mup-no"],
        "lrs": TRANSFER_LR,
        "seeds": [1],
        "image_size": 224,
        "drop_path_rate": 0.3,
    },
    "cnx-l": {
        "wall": "72:00:00",
        "gpus": 1,
        "methods": ["maxP", "mup-full", "mup-no"],
        "lrs": TRANSFER_LR,
        "seeds": [1],
        "image_size": 224,
        "drop_path_rate": 0.4,
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
#SBATCH --gres gpu:${gpus_per_node}
#SBATCH --output ${output_dir}/maxp_${scale}_${method_tag}_lr${lr_tag}_s${seed}.out
#SBATCH --error  ${output_dir}/maxp_${scale}_${method_tag}_lr${lr_tag}_s${seed}.err

module add ML-bundle/25.10
source "${venv_path}/bin/activate"
cd "${repo_path}"

export OMP_NUM_THREADS=16
export HF_HOME="$${HF_HOME:-/net/scratch/hscra/plgrid/plgmwojnar/hf}"
export WANDB_PROJECT="$${WANDB_PROJECT:-maxP-vision}"
export WANDB_RUN_NAME="maxp_${scale}_${method_tag}_lr${lr_tag}_s${seed}"

python experiments/vision/train.py \\
  --scale ${scale} \\
  --method ${method} \\
  --lr ${lr} \\
  --seed ${seed} \\
  --dataset ${train_dataset} \\
  --epochs ${epochs} \\
  --batch-size ${batch_size} \\
  --num-workers ${num_workers} \\
  --val-steps ${val_steps} \\
  --output-dir ${output_dir} \\
  ${extra_train_args}
"""
)


def _lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("+", "")


def _method_tag(method: str) -> str:
    return method.replace("-", "")


def _has_checkpoint(out_dir: Path) -> bool:
    ckpt_dir = out_dir / "checkpoint"
    return ckpt_dir.is_dir() and any(ckpt_dir.iterdir())


def parse_args() -> argparse.Namespace:
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

    parser.add_argument("--train-dataset", default="imagenet12k")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-steps", type=int, default=500)

    parser.add_argument("--extra-train-args", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sc = SCALE_CONFIGS[args.scale]
    methods = args.methods or sc["methods"]
    lrs = args.lrs or sc["lrs"]
    seeds = args.seeds or sc["seeds"]
    gpus_per_node = args.gpus_per_node if args.gpus_per_node is not None else sc["gpus"]

    today = date.today().strftime("%Y-%m-%d")
    submitted = skipped = 0

    for method in methods:
        for lr in lrs:
            for seed in seeds:
                run_name = f"{today}_{args.scale}_{_method_tag(method)}_lr{_lr_tag(lr)}_s{seed}"
                out_dir = Path(args.runs_dir) / run_name

                if not args.resume and _has_checkpoint(out_dir):
                    print(f"[skip]   {run_name}")
                    skipped += 1
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)

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
                    train_dataset=args.train_dataset,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    val_steps=args.val_steps,
                    gpus_per_node=gpus_per_node,
                    cpus_per_gpu=args.cpus_per_gpu,
                    mem_per_gpu=args.mem_per_gpu,
                    account=args.account,
                    partition=args.partition,
                    extra_train_args=args.extra_train_args,
                )
                script_path = out_dir / "job.sh"
                script_path.write_text(script)

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
