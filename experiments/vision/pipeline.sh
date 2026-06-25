#!/usr/bin/env bash
# Unattended VISION pipeline (ViT width ladder) — same muTransfer protocol as the
# LM pipeline: tune everything at small scale (s1+s2), only VERIFY at s3+.
#
# LOGIN-NODE SUBMIT-ONLY DESIGN: this script only calls sbatch (no compute, no
# polling) and exits in seconds. The whole DAG is submitted upfront:
#
#   stage A   s1 + s2 mup-no sweeps (--measure-only) ........ alignment sources
#   coord B   (afterany: A) best mup-no LR per scale, export
#             alignment tables, gate report G1-G3 .......... 1 small job
#   stage C   s1 + s2 maxP-meas sweeps — prefactor re-tune
#             under the measured table
#   coord C   (afterany: C) best maxP-meas LR per scale,
#             s1-vs-s2 consistency, writes s3 bracket sentinels
#   stage D   s3 full-grid candidates; coord C releases only
#             the 3-pt bracket around each arm's s2 optimum
#   coord D   (afterany: D) transfer proof + E1 verdict;
#             releases s4/s5 sentinels
#   stage E   s4 (chained) + s5 (chained) candidates, runtime-gated per LR
#
# Decisions unknowable at submit time are enforced by runtime sentinel files
# (launch_sweep --require). s4/s5 rely on chained checkpoint resume.
#
# PREREQUISITE: confirm a real train.py end-to-end run + a checkpoint→resume
# cycle on the server before submitting (see docs/vision_experiment_plan.md gates).
#
# Usage (login node):  bash experiments/vision/pipeline.sh
# Dry run (no sbatch): DRY=1 bash experiments/vision/pipeline.sh
set -euo pipefail

# --- config (env-overridable) ----------------------------------------------
REPO="${REPO:-/net/storage/pr3/plgrid/plggadlers/maxP}"
RUNS_DIR="${RUNS_DIR:-$REPO/runs}"
VENV="${VENV:-$REPO/.venv}"
DATASET="${DATASET:-imagenet12k}"
EPOCHS="${EPOCHS:-4}"
BATCH="${BATCH:-512}"
STATE="${STATE:-$RUNS_DIR/.vpipeline}"
GATE_TOL="${GATE_TOL:-0.02}"
ACCOUNT="${ACCOUNT:-plgadlers-gpu-gh200}"
PARTITION="${PARTITION:-plgrid-gpu-gh200}"
DRY="${DRY:-0}"

# Candidate LR grid (tags must match launch_sweep ALL_LRS in %.0e form).
GRID_LRS="${GRID_LRS:-3e-04 1e-03 3e-03 1e-02 3e-02 1e-01 3e-01}"

cd "$REPO"
mkdir -p "$STATE"
log() { echo "[vpipeline] $*"; }

LAUNCH=(python experiments/vision/launch_sweep.py
        --runs-dir "$RUNS_DIR" --venv-path "$VENV" --repo-path "$REPO"
        --dataset "$DATASET" --epochs "$EPOCHS" --batch-size "$BATCH")
[ "$DRY" = "1" ] && LAUNCH+=(--dry-run)

submit() {  # submit a standalone coordinator script, echo job id
    if [ "$DRY" = "1" ]; then echo "DRY"; log "[dry] sbatch $1" >&2; return; fi
    sbatch "$@" | awk '{print $NF}'
}

dep_from() {  # ids file -> afterany:id:id:...
    [ "$DRY" = "1" ] && { echo "afterany:DRY"; return; }
    [ -s "$1" ] || { echo "ERROR: no job ids in $1 — refusing to wire dependencies" >&2; exit 1; }
    echo "afterany:$(paste -sd: "$1")"
}

coord_header() {  # $1 job name, $2 dependency
    cat <<EOF
#!/bin/bash -l
#SBATCH --job-name $1
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 16
#SBATCH --mem-per-gpu 118GB
#SBATCH --time 01:00:00
#SBATCH --account $ACCOUNT
#SBATCH --partition $PARTITION
#SBATCH --gres gpu:1
#SBATCH --dependency $2
#SBATCH --output $STATE/$1.out
#SBATCH --error  $STATE/$1.err
module add ML-bundle/25.10
source "$VENV/bin/activate"
cd "$REPO"
set -ex
EOF
}

ANALYZE="python experiments/vision/pipeline_analyze.py"
EXPORT="python experiments/vision/export_alignment.py"

# ============ STAGE A: s1+s2 mup-no sweeps (alignment sources) ==============
: > "$STATE/A.ids"
log "stage A: s1 + s2 mup-no sweeps (--measure-only sources)"
"${LAUNCH[@]}" --scale s1 --methods mup-no --measure-only --tag meas --job-ids-file "$STATE/A.ids"
"${LAUNCH[@]}" --scale s2 --methods mup-no --measure-only --tag meas --job-ids-file "$STATE/A.ids"

# ============ COORD B: pick source LRs, export tables, gates ================
{
    coord_header vmaxp_coordB "$(dep_from "$STATE/A.ids")"
    cat <<EOF
BEST_S2=\$($ANALYZE best-lr --glob "$RUNS_DIR/*_s2_mupno_lr*_s*_meas")
BEST_S1=\$($ANALYZE best-lr --glob "$RUNS_DIR/*_s1_mupno_lr*_s*_meas")
echo "\$BEST_S2" > "$STATE/best_lr_mupno_s2.txt"
echo "\$BEST_S1" > "$STATE/best_lr_mupno_s1.txt"
$EXPORT $RUNS_DIR/*_s2_mupno_lr\${BEST_S2}_s*_meas -o "$STATE/s2_align.json"
$EXPORT $RUNS_DIR/*_s1_mupno_lr\${BEST_S1}_s*_meas -o "$STATE/s1_align.json"
ADJ_S2=\$(python -c "g='$GRID_LRS'.split(); i=g.index('\$BEST_S2'); print(g[i+1] if i+1 < len(g) else g[i-1])")
{
    echo "=== Gate report (\$(date)) ==="
    echo "mup-no optima: s1=\$BEST_S1 s2=\$BEST_S2 \$([ "\$BEST_S1" = "\$BEST_S2" ] && echo '[consistent]' || echo '[WARN: differ across scale]')"
    $ANALYZE gates \\
        --runs-glob "$RUNS_DIR/*_s2_mupno_lr\${BEST_S2}_s*_meas" \\
        --adjacent-glob "$RUNS_DIR/*_s2_mupno_lr\${ADJ_S2}_s*_meas"
    python - "$STATE/s1_align.json" "$STATE/s2_align.json" <<'PYEOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
worst = max(abs(a[k][i] - b[k][i]) for k in set(a) & set(b) for i in range(3))
ok = "OK - tiny-proxy story holds" if worst < 0.05 else "WARN - keep s2 as canonical source"
print(f"G3 s1-vs-s2: max |alignment delta| = {worst:.4f}  [{ok}]")
PYEOF
} >> "$STATE/report.txt" 2>&1
EOF
} > "$STATE/coordB.sh"
BJOB=$(submit "$STATE/coordB.sh")
log "coord B: $BJOB"

# ============ STAGE C: s1+s2 maxP-meas sweeps (prefactor re-tune) ===========
: > "$STATE/C.ids"
log "stage C: s1 + s2 maxP-meas sweeps (re-tune under measured table)"
for SC in s1 s2; do
    "${LAUNCH[@]}" --scale "$SC" --methods maxP-meas \
        --alignment-table "$STATE/s2_align.json" --tag transfer-s2 \
        --dependency "afterany:$BJOB" --require "$STATE/s2_align.json" \
        --job-ids-file "$STATE/C.ids"
done

# ============ COORD C: pick transfer LR at small scale, release s3 ==========
{
    coord_header vmaxp_coordC "$(dep_from "$STATE/C.ids")"
    cat <<EOF
T_S2=\$($ANALYZE best-lr --glob "$RUNS_DIR/*_s2_maxPmeas_lr*_s*_transfer-s2")
T_S1=\$($ANALYZE best-lr --glob "$RUNS_DIR/*_s1_maxPmeas_lr*_s*_transfer-s2")
B_S2=\$(cat "$STATE/best_lr_mupno_s2.txt")
echo "\$T_S2" > "$STATE/best_lr_meas_s2.txt"
echo "\$T_S1" > "$STATE/best_lr_meas_s1.txt"
echo "maxP-meas optima: s1=\$T_S1 s2=\$T_S2 \$([ "\$T_S1" = "\$T_S2" ] && echo '[consistent]' || echo '[WARN: differ across scale]')" >> "$STATE/report.txt"
release() {  # \$1 stage prefix, \$2 arm name, \$3 center
    BRACKET=\$(python -c "g='$GRID_LRS'.split(); i=g.index('\$3'); print(' '.join(g[max(0, i-1):i+2]))")
    for lr in \$BRACKET; do touch "$STATE/\${1}_go_\${2}_\${lr}"; done
    EDGE=""
    python -c "g='$GRID_LRS'.split(); import sys; sys.exit(0 if g.index('\$3') in (0, len(g)-1) else 1)" \
        && EDGE=" [WARN: center at grid edge]"
    echo "\$1 \$2 bracket released: \$BRACKET (center \$3)\$EDGE" >> "$STATE/report.txt"
}
release s3 base "\$B_S2"
release s3 transfer "\$T_S2"
EOF
} > "$STATE/coordC.sh"
CJOB=$(submit "$STATE/coordC.sh")
log "coord C: $CJOB"

# ============ STAGE D: s3 verification brackets ============================
: > "$STATE/D.ids"
log "stage D: s3 brackets (both arms, gated per LR)"
"${LAUNCH[@]}" --scale s3 --methods mup-no --lrs $GRID_LRS --tag base \
    --dependency "afterany:$CJOB" --require "$STATE/s3_go_base_{lr}" \
    --job-ids-file "$STATE/D.ids"
"${LAUNCH[@]}" --scale s3 --methods maxP-meas --alignment-table "$STATE/s2_align.json" \
    --lrs $GRID_LRS --tag transfer-s2 \
    --dependency "afterany:$CJOB" --require "$STATE/s3_go_transfer_{lr}" \
    --job-ids-file "$STATE/D.ids"

# ============ COORD D: transfer proof + E1 verdict, release s4/s5 ===========
{
    coord_header vmaxp_coordD "$(dep_from "$STATE/D.ids")"
    cat <<EOF
BEST3_BASE=\$($ANALYZE best-lr --glob "$RUNS_DIR/*_s3_mupno_lr*_s*_base")
BEST3_TRAN=\$($ANALYZE best-lr --glob "$RUNS_DIR/*_s3_maxPmeas_lr*_s*_transfer-s2")
C_BASE=\$(cat "$STATE/best_lr_mupno_s2.txt")
C_TRAN=\$(cat "$STATE/best_lr_meas_s2.txt")
{
    echo "transfer proof: s3 base argmin=\$BEST3_BASE vs transferred \$C_BASE \$([ "\$BEST3_BASE" = "\$C_BASE" ] && echo '[TRANSFERS]' || echo '[WARN: moved]')"
    echo "transfer proof: s3 meas argmin=\$BEST3_TRAN vs transferred \$C_TRAN \$([ "\$BEST3_TRAN" = "\$C_TRAN" ] && echo '[TRANSFERS]' || echo '[WARN: moved]')"
} >> "$STATE/report.txt"
if ! $ANALYZE e1 \\
        --baseline-glob "$RUNS_DIR/*_s3_mupno_lr*_s*_base" \\
        --transfer-glob "$RUNS_DIR/*_s3_maxPmeas_lr*_s*_transfer-s2" \\
        --tol $GATE_TOL >> "$STATE/report.txt" 2>&1; then
    echo "E1 FAILED - s4/s5 will self-skip" >> "$STATE/report.txt"
    exit 0
fi
release() {  # \$1 stage prefix, \$2 arm name, \$3 center
    BRACKET=\$(python -c "g='$GRID_LRS'.split(); i=g.index('\$3'); print(' '.join(g[max(0, i-1):i+2]))")
    for lr in \$BRACKET; do touch "$STATE/\${1}_go_\${2}_\${lr}"; done
    echo "\$1 \$2 bracket released: \$BRACKET (center \$3)" >> "$STATE/report.txt"
}
release s4 base "\$BEST3_BASE"
release s4 transfer "\$BEST3_TRAN"
touch "$STATE/s5_go_base_\${BEST3_BASE}"
touch "$STATE/s5_go_transfer_\${BEST3_TRAN}"
echo "s5 released: base=\${BEST3_BASE} transfer=\${BEST3_TRAN}" >> "$STATE/report.txt"
EOF
} > "$STATE/coordD.sh"
DJOB=$(submit "$STATE/coordD.sh")
log "coord D: $DJOB"

# ============ STAGE E: s4 + s5 candidates (chained, runtime-gated) ==========
log "stage E: s4 + s5 candidates (per-LR gated by coord D, chained resume)"
"${LAUNCH[@]}" --scale s4 --methods mup-no --lrs $GRID_LRS --tag base \
    --dependency "afterany:$DJOB" --require "$STATE/s4_go_base_{lr}"
"${LAUNCH[@]}" --scale s4 --methods maxP-meas --alignment-table "$STATE/s2_align.json" \
    --lrs $GRID_LRS --tag transfer-s2 \
    --dependency "afterany:$DJOB" --require "$STATE/s4_go_transfer_{lr}"
"${LAUNCH[@]}" --scale s5 --methods mup-no --lrs $GRID_LRS --tag base \
    --dependency "afterany:$DJOB" --require "$STATE/s5_go_base_{lr}"
"${LAUNCH[@]}" --scale s5 --methods maxP-meas --alignment-table "$STATE/s2_align.json" \
    --lrs $GRID_LRS --tag transfer-s2 \
    --dependency "afterany:$DJOB" --require "$STATE/s5_go_transfer_{lr}"

log "DAG fully submitted. Nothing else to do on the login node."
log "Progress:  squeue --me | grep maxp"
log "Verdicts:  cat $STATE/report.txt   (after each coordinator runs)"
