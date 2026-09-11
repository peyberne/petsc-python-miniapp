#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --partition boost_fua_prod
#SBATCH --time=2:00:00
#SBATCH --output=benchmark_%j.log
#SBATCH --error=benchmark_%j.err

SCRIPT_DIR=/pitagora/home/userexternal/mpeybern/petsc-python-miniapp
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/env_pitagora_python-petsc.sh"
set -euo pipefail

# Small test data (copy from Kuma or use local)
DATA_DIR="/pitagora/home/userexternal/mpeybern/data/TCV_3D_fine_ES/vorticity"
TEST_CASE="TCV_3D_fine_ES/vorticity"
MATRIX_FILE="$DATA_DIR/mat_vorticity.dat"
RHS_FILE="$DATA_DIR/rhs_vorticity.dat"
GUESS_FILE="$DATA_DIR/guess_vorticity.dat"
REF_FILE="$DATA_DIR/sol_vorticity.dat"
CONFIG_FILE="$SCRIPT_DIR/data/options.json"
MPI_COUNTS=${MPI_COUNTS:-1:2:4}
REPETITIONS=${REPETITIONS:-1}
PETSC_EXTRA_OPTS=${PETSC_EXTRA_OPTS:-}

echo "=========================================="
echo "Starting PETSc benchmark on Pitagora"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Allocated MPI processes: $SLURM_NTASKS"
echo "MPI scaling points: $MPI_COUNTS"
echo "=========================================="

for input_file in "$MATRIX_FILE" "$RHS_FILE" "$GUESS_FILE" "$REF_FILE" "$CONFIG_FILE"; do
    if [ ! -f "$input_file" ]; then
        echo "Error: Input file $input_file not found!"
        exit 1
    fi
done

IFS=',:' read -ra MPI_COUNT_LIST <<< "$MPI_COUNTS"
RESULT_FILES=()
for MPI_COUNT in "${MPI_COUNT_LIST[@]}"; do
    if ! [[ "$MPI_COUNT" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: Invalid MPI count: $MPI_COUNT"
        exit 1
    fi
    if (( MPI_COUNT > SLURM_NTASKS )); then
        echo "Error: MPI count $MPI_COUNT exceeds allocation of $SLURM_NTASKS tasks"
        exit 1
    fi

    RESULT_FILE="$SCRIPT_DIR/results/scaling_${SLURM_JOB_ID}_${MPI_COUNT}.json"
    RESULT_FILES+=("$RESULT_FILE")
    echo "Running benchmark with $MPI_COUNT MPI process(es)"
    PETSC_ARGS=()
    if [ -n "$PETSC_EXTRA_OPTS" ]; then
        IFS=':' read -ra PETSC_ARGS <<< "$PETSC_EXTRA_OPTS"
    fi
    srun --ntasks="$MPI_COUNT" \
        --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        --gpus-per-task=1 \
        --gpu-bind=closest \
        python3 "$SCRIPT_DIR/benchmark_petsc.py" \
        --mat "$MATRIX_FILE" \
        --rhs "$RHS_FILE" \
        --guess "$GUESS_FILE" \
        --ref "$REF_FILE" \
        --config "$CONFIG_FILE" \
        --repetitions "$REPETITIONS" \
        --results-json "$RESULT_FILE" \
        --test-case "$TEST_CASE" \
        --petsc-options "${PETSC_ARGS[@]+"${PETSC_ARGS[@]}"}" \
        --gpu
done

python3 "$SCRIPT_DIR/benchmark_petsc.py" \
    --plot-results "${RESULT_FILES[@]}" \
    --output "$SCRIPT_DIR/results/benchmark_results_pitagora.png"

echo "=========================================="
echo "Benchmark completed at $(date)"
echo "=========================================="
