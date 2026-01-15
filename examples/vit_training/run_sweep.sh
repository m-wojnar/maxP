#!/bin/bash
#
# Run parallel hyperparameter sweep with multiple workers.
# Each worker gets assigned to a GPU and processes a subset of the grid.
#
# Usage:
#   ./run_sweep.sh <N_WORKERS> [SWEEP_CONFIG]
#
# Examples:
#   ./run_sweep.sh 4                        # 4 workers, default sweep_config.yaml
#   ./run_sweep.sh 8 my_sweep.yaml          # 8 workers, custom config
#

set -e

# Function to handle Ctrl+C and terminate all child processes
cleanup() {
    echo ""
    echo "Caught Ctrl+C! Terminating all jobs..."
    kill 0
    exit 1
}

# Trap Ctrl+C signal (SIGINT) and call the cleanup function
trap cleanup SIGINT

# Parse arguments
N_WORKERS=${1:-1}
SWEEP_CONFIG=${2:-sweep_config.yaml}

if [ "$N_WORKERS" -lt 1 ]; then
    echo "Error: N_WORKERS must be at least 1"
    exit 1
fi

echo "=============================================="
echo "Starting sweep with $N_WORKERS workers"
echo "Config: $SWEEP_CONFIG"
echo "=============================================="
echo ""

# Launch workers
for ((i=0; i<N_WORKERS; i++))
do
    GPU_ID=$((i % 8))  # Cycle through GPUs 0-7
    echo "Launching worker $i on GPU $GPU_ID"

    # Set thread limits to avoid oversubscription
    OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} \
    MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} \
    NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1} \
    TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1} \
    TORCH_INTEROP_THREADS=${TORCH_INTEROP_THREADS:-1} \
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    WORKER_ID=$i \
    N_WORKERS=$N_WORKERS \
    python sweep.py --config "$SWEEP_CONFIG" &
done

echo ""
echo "All workers launched. Waiting for completion..."
echo "(Press Ctrl+C to terminate all jobs)"
echo ""

wait

echo ""
echo "=============================================="
echo "All training jobs completed."
echo "=============================================="
