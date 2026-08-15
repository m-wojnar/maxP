#!/usr/bin/env bash
# Submit a full sweep for one scale.
#
# Usage:
#   bash experiments/lm/run.sh s3 [launch_sweep.py flags …]
#
# Override any of them by passing extra flags after the scale argument.
#
# Examples:
#   # baseline mup-no sweep + passive alignment logging (source runs):
#   bash experiments/lm/run.sh s1 --measure-only --tag meas
#
#   # transfer arm: c solved from an exported alignment table
#   # (--measure-only additionally logs realized alignment for free):
#   bash experiments/lm/run.sh s2 --methods maxP-meas \
#       --alignment-table s1_align.json --measure-only --tag transfer-s1
#
#   # dynamic maxP (diagnostics only):
#   bash experiments/lm/run.sh s2 --methods maxP
#
#   bash experiments/lm/run.sh s2 --dry-run
set -euo pipefail

SCALE="${1:?Usage: $0 <s1|s2|s3|s4|s5> [extra launch_sweep.py flags]}"
shift

RUNS_DIR="${RUNS_DIR:-/net/storage/pr3/plgrid/plggadlers/maxP/runs}"
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/net/storage/pr3/plgrid/plggadlers/maxP/experiments/lm/assets/hf/Llama-3.1-8B}"
VENV_PATH="${VENV_PATH:-/net/storage/pr3/plgrid/plggadlers/maxP/.venv}"
REPO_PATH="${REPO_PATH:-/net/storage/pr3/plgrid/plggadlers/maxP}"

cd "$REPO_PATH"

python experiments/lm/launch_sweep.py \
    --scale "$SCALE" \
    --runs-dir "$RUNS_DIR" \
    --hf-assets-path "$HF_ASSETS_PATH" \
    --venv-path "$VENV_PATH" \
    --repo-path "$REPO_PATH" \
    "${@}"
