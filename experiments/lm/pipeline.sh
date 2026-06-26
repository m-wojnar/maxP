#!/usr/bin/env bash
# Unattended pipeline — "real-life" muTransfer protocol:
# tune everything at small scale (s1+s2), only VERIFY at s3+.
#
# LOGIN-NODE SUBMIT-ONLY DESIGN: this script only calls sbatch (no compute,
# no polling) and exits in seconds. The whole DAG is submitted upfront:
#
#   stage A   s1 + s2 mup-no sweeps (--measure-only) ........ 42 GPU jobs
#   coord B   (afterany: A) best mup-no LR per scale, export
#             alignment tables, gate report G1-G3 .......... 1 small job
#   stage C   s1 + s2 maxP-meas sweeps — the prefactor
#             RE-TUNE under the measured table ............. 42 GPU jobs
#   coord C   (afterany: C) best maxP-meas LR per scale,
#             s1-vs-s2 consistency check, writes s3 bracket
#             sentinels (3-pt bracket per arm) ............. 1 small job
#   stage D   s3 full-grid candidates; coord C releases only
#             the 3-pt bracket around each arm's s2 optimum   28 jobs (12 train)
#   coord D   (afterany: D) transfer proof (s3 argmin ==
#             transferred LR?) + E1 verdict; releases s4/s5  1 small job
#   stage E   s4 full-grid candidates (chain=2, bracket
#             released by coord D) + s5 full-grid candidates
#             (chain=11, single LR released) ................ 210 jobs (26 train)
#
# Decisions that can't be known at submit time are enforced by runtime
# sentinel files (see launch_sweep --require).
#
# PREREQUISITE: run experiments/lm/verify_resume.sh manually and confirm
# PASS before submitting — s4/s5 rely on chained checkpoint resume.
#
# Usage (login node):     bash experiments/lm/pipeline.sh
# Dry run (no sbatch):    DRY=1 bash experiments/lm/pipeline.sh
#
# Idempotent-ish: launch_sweep skips runs that already have checkpoints,
# but coordinator jobs are resubmitted on every invocation — do not run
# this script twice while the pipeline is in flight.
set -euo pipefail

# --- config (env-overridable) ----------------------------------------------
REPO="${REPO:-/net/storage/pr3/plgrid/plggadlers/maxP}"
RUNS_DIR="${RUNS_DIR:-$REPO/runs}"
VENV="${VENV:-$REPO/.venv}"
HF_ASSETS="${HF_ASSETS:-$REPO/experiments/lm/assets/hf/Llama-3.1-8B}"
DATASET="${DATASET:-fineweb-edu}"
STATE="${STATE:-$RUNS_DIR/.pipeline}"
GATE_TOL="${GATE_TOL:-0.02}"
ACCOUNT="${ACCOUNT:-plgadlers-gpu-gh200}"
PARTITION="${PARTITION:-plgrid-gpu-gh200}"
DRY="${DRY:-0}"
RUN_DATE="${RUN_DATE:-}"

# Candidate LR grid for s3/s4/s5 (must match launch_sweep ALL_LRS; tags in
# launch_sweep's printf %.0e format so sentinel and run names line up).
# No hard-coded optima: jobs are pre-submitted at EVERY grid LR and the
# coordinators release only the 3-point bracket (argmin +/- 1 grid step)
# computed from this experiment's own small-scale results. Non-released
# candidates exit in ~1 s at runtime.
GRID_LRS="${GRID_LRS:-1e-03 3e-03 1e-02 3e-02 1e-01 3e-01 1e00}"

cd "$REPO"
mkdir -p "$STATE"
log() { echo "[pipeline] $*"; }

# Login node has no python by default; load a stdlib interpreter for launch_sweep.py.
command -v python >/dev/null 2>&1 || module add GCCcore/13.2.0 Python/3.11.5

LAUNCH=(python experiments/lm/launch_sweep.py
        --runs-dir "$RUNS_DIR" --hf-assets-path "$HF_ASSETS"
        --venv-path "$VENV" --repo-path "$REPO" --dataset "$DATASET")
[ "$DRY" = "1" ] && LAUNCH+=(--dry-run)
# Pin run-date prefix so a resubmit reuses existing run dirs (skip done / resume chains).
[ -n "$RUN_DATE" ] && LAUNCH+=(--run-date "$RUN_DATE")

submit() {  # submit a standalone job script, echo job id
    if [ "$DRY" = "1" ]; then echo "DRY"; log "[dry] sbatch $1" >&2; return; fi
    sbatch "$@" | awk '{print $NF}'
}

dep_from() {  # ids file -> afterany:id:id:...
    [ "$DRY" = "1" ] && { echo "afterany:DRY"; return; }
    [ -s "$1" ] || { echo "ERROR: no job ids in $1 — refusing to wire dependencies (re-run on a dirty runs dir?)" >&2; exit 1; }
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

# ============ STAGE A: s1+s2 mup-no sweeps (baseline + sources) =============
: > "$STATE/A.ids"
log "stage A: s1 + s2 mup-no sweeps (sources)"
"${LAUNCH[@]}" --scale s1 --measure-only --tag meas --job-ids-file "$STATE/A.ids"
"${LAUNCH[@]}" --scale s2 --measure-only --tag meas --job-ids-file "$STATE/A.ids"

# ============ COORD B: pick source LRs, export tables, gates ================
{
    coord_header maxp_coordB "$(dep_from "$STATE/A.ids")"
    cat <<EOF
BEST_S2=\$(python experiments/lm/pipeline_analyze.py best-lr --glob "$RUNS_DIR/*_s2_mupno_lr*_meas")
BEST_S1=\$(python experiments/lm/pipeline_analyze.py best-lr --glob "$RUNS_DIR/*_s1_mupno_lr*_meas")
echo "\$BEST_S2" > "$STATE/best_lr_mupno_s2.txt"
echo "\$BEST_S1" > "$STATE/best_lr_mupno_s1.txt"
python experiments/lm/export_alignment.py $RUNS_DIR/*_s2_mupno_lr\${BEST_S2}_s*_meas -o "$STATE/s2_align.json"
python experiments/lm/export_alignment.py $RUNS_DIR/*_s1_mupno_lr\${BEST_S1}_s*_meas -o "$STATE/s1_align.json"
# adjacent LR tag for gate G2 (grid must match launch_sweep ALL_LRS)
ADJ_S2=\$(python -c "g='$GRID_LRS'.split(); i=g.index('\$BEST_S2'); print(g[i+1] if i+1 < len(g) else g[i-1])")
{
    echo "=== Gate report (\$(date)) ==="
    echo "mup-no optima: s1=\$BEST_S1 s2=\$BEST_S2 \$([ "\$BEST_S1" = "\$BEST_S2" ] && echo '[consistent]' || echo '[WARN: differ across scale]')"
    python experiments/lm/pipeline_analyze.py gates \\
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
        --alignment-table "$STATE/s2_align.json" --measure-only --tag transfer-s2 \
        --dependency "afterany:$BJOB" --require "$STATE/s2_align.json" \
        --job-ids-file "$STATE/C.ids"
done

# ============ COORD C: pick transfer LR at small scale, release s3 ==========
{
    coord_header maxp_coordC "$(dep_from "$STATE/C.ids")"
    cat <<EOF
T_S2=\$(python experiments/lm/pipeline_analyze.py best-lr --glob "$RUNS_DIR/*_s2_maxPmeas_lr*_transfer-s2")
T_S1=\$(python experiments/lm/pipeline_analyze.py best-lr --glob "$RUNS_DIR/*_s1_maxPmeas_lr*_transfer-s2")
B_S2=\$(cat "$STATE/best_lr_mupno_s2.txt")
echo "\$T_S2" > "$STATE/best_lr_meas_s2.txt"
echo "\$T_S1" > "$STATE/best_lr_meas_s1.txt"
{
    echo "maxP-meas optima: s1=\$T_S1 s2=\$T_S2 \$([ "\$T_S1" = "\$T_S2" ] && echo '[consistent]' || echo '[WARN: differ across scale]')"
} >> "$STATE/report.txt"
# Release the 3-point s3 bracket (argmin +/- 1 grid step) around each arm's
# s2 optimum (the tuned scale closest to target). Edge-of-grid centers get a
# truncated 2-point bracket + warning (optimum may lie outside the grid).
release() {  # \$1 stage prefix (s3), \$2 arm name, \$3 center
    BRACKET=\$(python -c "g='$GRID_LRS'.split(); i=g.index('\$3'); print(' '.join(g[max(0, i-1):i+2]))")
    for lr in \$BRACKET; do touch "$STATE/\${1}_go_\${2}_\${lr}"; done
    EDGE=""
    python -c "g='$GRID_LRS'.split(); import sys; sys.exit(0 if g.index('\$3') in (0, len(g)-1) else 1)" \
        && EDGE=" [WARN: center at grid edge - optimum may lie outside grid]"
    echo "\$1 \$2 bracket released: \$BRACKET (center \$3)\$EDGE" >> "$STATE/report.txt"
}
release s3 base "\$B_S2"
release s3 transfer "\$T_S2"
EOF
} > "$STATE/coordC.sh"
CJOB=$(submit "$STATE/coordC.sh")
log "coord C: $CJOB"

# ============ STAGE D: s3 verification brackets =============================
: > "$STATE/D.ids"
log "stage D: s3 brackets (both arms, gated per LR)"
"${LAUNCH[@]}" --scale s3 --lrs $GRID_LRS --measure-only --tag meas \
    --dependency "afterany:$CJOB" --require "$STATE/s3_go_base_{lr}" \
    --job-ids-file "$STATE/D.ids"
"${LAUNCH[@]}" --scale s3 --methods maxP-meas --alignment-table "$STATE/s2_align.json" \
    --lrs $GRID_LRS --measure-only --tag transfer-s2 \
    --dependency "afterany:$CJOB" --require "$STATE/s3_go_transfer_{lr}" \
    --job-ids-file "$STATE/D.ids"

# ============ COORD D: transfer proof + E1 verdict, release s4/s5 ===========
{
    coord_header maxp_coordD "$(dep_from "$STATE/D.ids")"
    cat <<EOF
BEST3_BASE=\$(python experiments/lm/pipeline_analyze.py best-lr --glob "$RUNS_DIR/*_s3_mupno_lr*_meas")
BEST3_TRAN=\$(python experiments/lm/pipeline_analyze.py best-lr --glob "$RUNS_DIR/*_s3_maxPmeas_lr*_transfer-s2")
C_BASE=\$(cat "$STATE/best_lr_mupno_s2.txt")
C_TRAN=\$(cat "$STATE/best_lr_meas_s2.txt")
{
    echo "transfer proof: s3 base argmin=\$BEST3_BASE vs transferred \$C_BASE \$([ "\$BEST3_BASE" = "\$C_BASE" ] && echo '[TRANSFERS]' || echo '[WARN: moved]')"
    echo "transfer proof: s3 meas argmin=\$BEST3_TRAN vs transferred \$C_TRAN \$([ "\$BEST3_TRAN" = "\$C_TRAN" ] && echo '[TRANSFERS]' || echo '[WARN: moved]')"
} >> "$STATE/report.txt"
if ! python experiments/lm/pipeline_analyze.py e1 \\
        --baseline-glob "$RUNS_DIR/*_s3_mupno_lr*_meas" \\
        --transfer-glob "$RUNS_DIR/*_s3_maxPmeas_lr*_transfer-s2" \\
        --tol $GATE_TOL >> "$STATE/report.txt" 2>&1; then
    echo "E1 FAILED - s4/s5 will self-skip" >> "$STATE/report.txt"
    exit 0
fi
# s4 brackets re-centered on the s3 argmin (best knowledge at release time)
release() {  # \$1 stage prefix, \$2 arm name, \$3 center
    BRACKET=\$(python -c "g='$GRID_LRS'.split(); i=g.index('\$3'); print(' '.join(g[max(0, i-1):i+2]))")
    for lr in \$BRACKET; do touch "$STATE/\${1}_go_\${2}_\${lr}"; done
    EDGE=""
    python -c "g='$GRID_LRS'.split(); import sys; sys.exit(0 if g.index('\$3') in (0, len(g)-1) else 1)" \
        && EDGE=" [WARN: center at grid edge - optimum may lie outside grid]"
    echo "\$1 \$2 bracket released: \$BRACKET (center \$3)\$EDGE" >> "$STATE/report.txt"
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

# ============ STAGE E: s4 brackets + s5 single-LR candidates ================
log "stage E: s4 + s5 full-grid candidates (runtime-gated per LR by coord D)"
"${LAUNCH[@]}" --scale s4 --lrs $GRID_LRS --tag base \
    --dependency "afterany:$DJOB" --require "$STATE/s4_go_base_{lr}"
"${LAUNCH[@]}" --scale s4 --methods maxP-meas --alignment-table "$STATE/s2_align.json" \
    --lrs $GRID_LRS --tag transfer-s2 \
    --dependency "afterany:$DJOB" --require "$STATE/s4_go_transfer_{lr}"
# s5: pre-submit every candidate LR; only the one coord D blesses will run.
"${LAUNCH[@]}" --scale s5 --lrs $GRID_LRS --tag base \
    --dependency "afterany:$DJOB" --require "$STATE/s5_go_base_{lr}"
"${LAUNCH[@]}" --scale s5 --methods maxP-meas --alignment-table "$STATE/s2_align.json" \
    --lrs $GRID_LRS --tag transfer-s2 \
    --dependency "afterany:$DJOB" --require "$STATE/s5_go_transfer_{lr}"

log "DAG fully submitted. Nothing else to do on the login node."
log "Progress:  squeue --me | grep maxp"
log "Verdicts:  cat $STATE/report.txt   (after each coordinator runs)"
