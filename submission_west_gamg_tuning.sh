#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --partition=boost_fua_prod
#SBATCH --time=06:00:00
#SBATCH --output=/pitagora/home/userexternal/mpeybern/petsc-python-miniapp/gamg-west-%x-%j.out
#SBATCH --error=/pitagora/home/userexternal/mpeybern/petsc-python-miniapp/gamg-west-%x-%j.err

set -euo pipefail

BASE=/pitagora/home/userexternal/mpeybern/petsc-python-miniapp
DATA=/pitagora_scratch/userexternal/mpeybern/test_miniapp/west/vorticity
: "${CONFIG:?CONFIG must name a JSON file under $BASE/data}"
LABEL=${LABEL:-gamg}
RESULT="$BASE/results/gamg_west_${LABEL}_${SLURM_JOB_ID}.json"

cd "$BASE"
source ./env_pitagora_python-petsc.sh

srun python "$BASE/benchmark_petsc.py" \
    --mat "$DATA/mat_vorticity.dat" \
    --rhs "$DATA/rhs_vorticity.dat" \
    --guess "$DATA/guess_vorticity.dat" \
    --ref "$DATA/sol_vorticity.dat" \
    --config "$BASE/data/$CONFIG" \
    --repetitions 3 \
    --view-ksp \
    --log-view \
    --results-json "$RESULT" \
    --test-case WEST/vorticity \
    --gpu

echo "Results: $RESULT"
