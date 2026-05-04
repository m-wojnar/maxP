#!/usr/bin/env bash
# Quick local debug run (CPU/GPU) with a tiny model and tiny dataset slice.
#
# Usage:
#   bash experiments/vision/run_debug.sh [--steps N] [--method METHOD]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${REPO}/.venv"

SCALE="${SCALE:-debug}"
METHOD="${METHOD:-maxP}"
DATASET="${DATASET:-beans}"
STEPS="${STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/maxp_vision_debug}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scale)      SCALE="$2";      shift 2 ;;
        --method)     METHOD="$2";     shift 2 ;;
        --dataset)    DATASET="$2";    shift 2 ;;
        --steps)      STEPS="$2";      shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

source "${VENV}/bin/activate"
cd "${REPO}"

mkdir -p "${OUTPUT_DIR}"

python experiments/vision/train.py \
    --scale "${SCALE}" \
    --method "${METHOD}" \
    --dataset "${DATASET}" \
    --lr 0.03 \
    --batch-size "${BATCH_SIZE}" \
    --num-workers 0 \
    --max-steps "${STEPS}" \
    --val-steps 4 \
    --no-compile \
    --debug \
    --alignment-warmup 5 \
    --solve-interval 5 \
    --log-interval 5 \
    --val-interval 50 \
    --output-dir "${OUTPUT_DIR}"
