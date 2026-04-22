#!/usr/bin/env bash
# Quick debug run on a local GPU machine (no SLURM).
# Uses the tiny debug model (dim=256, 2 layers, vocab=2048) and
# the bundled c4_test dataset — no downloads required.
#
# Usage:
#   bash experiments/lm/run_debug.sh [--steps N] [--method METHOD] [--gpus N]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${REPO}/.venv"

# Defaults — override with env vars or flags
STEPS="${STEPS:-20}"
SCALE="${SCALE:-debug}"
METHOD="${METHOD:-mup-no}"
GPUS="${GPUS:-1}"
DATASET="${DATASET:-c4_test}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/maxp_debug}"
TOKENIZER="${TOKENIZER:-${REPO}/experiments/lm/assets/hf/Llama-3.1-8B}"
C4_TEST="${C4_TEST:-${REPO}/experiments/lm/assets/c4_test}"

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)      STEPS="$2";      shift 2 ;;
        --scale)      SCALE="$2";      shift 2 ;;
        --method)     METHOD="$2";     shift 2 ;;
        --gpus)       GPUS="$2";       shift 2 ;;
        --dataset)    DATASET="$2";    shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --tokenizer)  TOKENIZER="$2";  shift 2 ;;
        --c4-test)    C4_TEST="$2";    shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

source "${VENV}/bin/activate"
cd "${REPO}"

export OMP_NUM_THREADS=4

echo "=== maxP debug run ==="
echo "  scale:      ${SCALE}"
echo "  method:     ${METHOD}"
echo "  dataset:    ${DATASET}"
echo "  steps:      ${STEPS}"
echo "  gpus:       ${GPUS}"
echo "  output_dir: ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}"

TRAIN_ARGS=(
    --scale "${SCALE}"
    --method "${METHOD}"
    --lr 1e-3
    --steps "${STEPS}"
    --dataset "${DATASET}"
    ${C4_TEST:+--dataset-path "${C4_TEST}"}
    ${TOKENIZER:+--hf-assets-path "${TOKENIZER}"}
    --output-dir "${OUTPUT_DIR}"
)

if [[ "${GPUS}" -eq 1 ]]; then
    # Single GPU: run python directly so logs stream to terminal
    LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29500 \
        python experiments/lm/train.py "${TRAIN_ARGS[@]}"
else
    # Multi-GPU: use torchrun
    torchrun --nproc_per_node="${GPUS}" experiments/lm/train.py "${TRAIN_ARGS[@]}"
fi
