#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition h100
#SBATCH --time=1:00:00
#SBATCH --output=benchmark_%j.log
#SBATCH --error=benchmark_%j.err
source env_kuma_python-petsc.sh

# Input files (adapt according to your files)
MATRIX_FILE="mat.dat"
RHS_FILE="rhs.dat"
GUESS_FILE="guess.dat"  # Optional - initial guess for solver
REF_FILE="sol.dat"      # reference solution 

echo "=========================================="
echo "Starting PETSc benchmark"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Number of MPI processes: $SLURM_NTASKS"
echo "=========================================="

# Check that files exist
if [ ! -f "$MATRIX_FILE" ]; then
    echo "Error: Matrix file $MATRIX_FILE not found!"
    exit 1
fi

if [ ! -f "$RHS_FILE" ]; then
    echo "Error: RHS file $RHS_FILE not found!"
    exit 1
fi

# Run benchmark
srun -n $SLURM_NTASKS python3 benchmark_petsc.py \
    --mat $MATRIX_FILE \
    --rhs $RHS_FILE \
    --guess $GUESS_FILE \
    --ref $REF_FILE \
    --gpu

echo "=========================================="
echo "Benchmark completed"
echo "Date: $(date)"
echo "=========================================="
