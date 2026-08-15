#!/usr/bin/env bash
# Vision muTransfer pipeline (ViT width ladder), mirroring the Llama3 LM flow.
#
# Three phases:
#   1) measure : run s2 mup-no --measure-only (1 seed) to log alignment
#   2) export  : build the alignment table from that run's metrics.json
#   3) sweep   : mup-no + maxP-meas across the ladder (maxP-meas uses the table)
#
# Usage:
#   bash experiments/vision/run.sh measure
#   bash experiments/vision/run.sh export <s2_measure_run_dir>
#   bash experiments/vision/run.sh sweep <scale> [extra launch_sweep.py flags]
set -euo pipefail

PHASE="${1:?Usage: $0 <measure|export|sweep> ...}"
shift || true

RUNS_DIR="${RUNS_DIR:-/net/storage/pr3/plgrid/plggadlers/maxP/runs}"
VENV_PATH="${VENV_PATH:-/net/storage/pr3/plgrid/plggadlers/maxP/.venv}"
REPO_PATH="${REPO_PATH:-/net/storage/pr3/plgrid/plggadlers/maxP}"
ALIGN_TABLE="${ALIGN_TABLE:-${REPO_PATH}/experiments/vision/s2_align.json}"

# Fixed epoch budget — identical #epochs (= #samples) at every width.
EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DATASET="${DATASET:-imagenet12k}"

cd "$REPO_PATH"

case "$PHASE" in
  measure)
    python experiments/vision/launch_sweep.py \
        --scale s2 --methods mup-no --lrs 3e-2 --seeds 1 \
        --measure-only \
        --runs-dir "$RUNS_DIR" --venv-path "$VENV_PATH" --repo-path "$REPO_PATH" \
        --dataset "$DATASET" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" "$@"
    ;;
  export)
    SRC="${1:?Usage: $0 export <s2_measure_run_dir>}"
    source "${VENV_PATH}/bin/activate"
    python experiments/vision/export_alignment.py "$SRC" -o "$ALIGN_TABLE"
    ;;
  sweep)
    SCALE="${1:?Usage: $0 sweep <scale>}"; shift
    python experiments/vision/launch_sweep.py \
        --scale "$SCALE" \
        --alignment-table "$ALIGN_TABLE" \
        --runs-dir "$RUNS_DIR" --venv-path "$VENV_PATH" --repo-path "$REPO_PATH" \
        --dataset "$DATASET" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" "$@"
    ;;
  *)
    echo "Unknown phase '$PHASE'. Choose measure | export | sweep."; exit 1
    ;;
esac
