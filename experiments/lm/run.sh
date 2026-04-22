#!/usr/bin/env bash
# Submit a full sweep for one scale.
#
# Usage:
#   bash experiments/lm/slurm/run.sh s3 [launch_sweep.py flags …]
#
# Per-scale defaults (seeds, LRs, methods) match experiments.md §4.1.
# Override any of them by passing extra flags after the scale argument.
#
# Examples:
#   bash experiments/lm/slurm/run.sh s3
#   bash experiments/lm/slurm/run.sh s5 --lrs 3e-3   # transfer LR
#   bash experiments/lm/slurm/run.sh s2 --dry-run
set -euo pipefail

SCALE="${1:?Usage: $0 <s1|s2|s3|s4|s5> [extra launch_sweep.py flags]}"
shift

# Per-scale defaults
case "$SCALE" in
    s1) SEEDS="1 2 3"; LRS="3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1" ;;
    s2) SEEDS="1 2";   LRS="3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1" ;;
    s3) SEEDS="1 2";   LRS="3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1" ;;
    s4) SEEDS="1";     LRS="3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1" ;;
    s5) SEEDS="1";     LRS="${TRANSFER_LR:-1e-2}"                 ;;
    *)  echo "Unknown scale '$SCALE'. Choose from s1 s2 s3 s4 s5."; exit 1 ;;
esac

RUNS_DIR="${RUNS_DIR:-/net/storage/pr3/plgrid/plggadlers/maxP/runs}"
DATASET="${DATASET:-fineweb-edu}"
HF_ASSETS_PATH="${HF_ASSETS_PATH:-$(pwd)/assets/hf/Llama-3.1-8B}"
VENV_PATH="${VENV_PATH:-/net/storage/pr3/plgrid/plggadlers/maxP/.venv}"
REPO_PATH="${REPO_PATH:-/net/storage/pr3/plgrid/plggadlers/maxP}"

cd "$REPO_PATH"
source "$VENV_PATH/bin/activate"

python experiments/lm/launch_sweep.py \
    --scale "$SCALE" \
    --methods maxP mup-full mup-no \
    --lrs $LRS \
    --seeds $SEEDS \
    --runs-dir "$RUNS_DIR" \
    --dataset "$DATASET" \
    --hf-assets-path "$HF_ASSETS_PATH" \
    --venv-path "$VENV_PATH" \
    --repo-path "$REPO_PATH" \
    "${@}"
