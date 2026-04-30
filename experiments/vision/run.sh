#!/usr/bin/env bash
# Submit a vision sweep for one scale.
#
# Usage:
#   bash experiments/vision/run.sh vit-s [extra launch_sweep.py flags ...]
set -euo pipefail

SCALE="${1:?Usage: $0 <debug|vit-s|vit-b|vit-l|mlp-s|mlp-m|mlp-b|mlp-l> [extra launch_sweep.py flags]}"
shift

case "$SCALE" in
    debug) METHODS="maxP"; LRS="3e-3"; SEEDS="1" ;;
    vit-s) METHODS="maxP mup-full mup-no"; LRS="3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 1.0 3.0"; SEEDS="1 2 3" ;;
    vit-b) METHODS="maxP mup-full mup-no"; LRS="3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 1.0 3.0"; SEEDS="1" ;;
    vit-l) METHODS="maxP mup-full mup-no"; LRS="${TRANSFER_LR:-1e-2}"; SEEDS="1" ;;
    mlp-s) METHODS="maxP mup-full mup-no"; LRS="3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 1.0 3.0"; SEEDS="1 2 3" ;;
    mlp-m) METHODS="maxP mup-full mup-no"; LRS="3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 1.0 3.0"; SEEDS="1" ;;
    mlp-b) METHODS="maxP mup-full mup-no"; LRS="${TRANSFER_LR:-1e-2}"; SEEDS="1" ;;
    mlp-l) METHODS="maxP mup-full mup-no"; LRS="${TRANSFER_LR:-1e-2}"; SEEDS="1" ;;
    *) echo "Unknown scale '$SCALE'. Choose from debug vit-s vit-b vit-l mlp-s mlp-m mlp-b mlp-l."; exit 1 ;;
esac

RUNS_DIR="${RUNS_DIR:-/net/storage/pr3/plgrid/plggadlers/maxP/runs}"
VENV_PATH="${VENV_PATH:-/net/storage/pr3/plgrid/plggadlers/maxP/.venv}"
REPO_PATH="${REPO_PATH:-/net/storage/pr3/plgrid/plggadlers/maxP}"

cd "$REPO_PATH"

python experiments/vision/launch_sweep.py \
    --scale "$SCALE" \
    --methods $METHODS \
    --lrs $LRS \
    --seeds $SEEDS \
    --runs-dir "$RUNS_DIR" \
    --venv-path "$VENV_PATH" \
    --repo-path "$REPO_PATH" \
    --epochs 10 \
    --batch-size 3072 \
    "${@}"
