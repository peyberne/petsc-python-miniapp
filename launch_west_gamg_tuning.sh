#!/bin/bash
set -euo pipefail

BASE=/pitagora/home/userexternal/mpeybern/petsc-python-miniapp
SBATCH="$BASE/submission_west_gamg_tuning.sh"

first=$(sbatch --job-name=gamg-safe \
    --export=ALL,CONFIG=gamg-west-safe.json,LABEL=safe "$SBATCH" | awk '{print $4}')
second=$(sbatch --job-name=gamg-asm --dependency=afterok:$first \
    --export=ALL,CONFIG=gamg-west-asm.json,LABEL=asm "$SBATCH" | awk '{print $4}')
third=$(sbatch --job-name=gamg-asm-smooth --dependency=afterok:$second \
    --export=ALL,CONFIG=gamg-west-asm-smooth.json,LABEL=asm_smooth "$SBATCH" | awk '{print $4}')
sbatch --job-name=gamg-square --dependency=afterok:$third \
    --export=ALL,CONFIG=gamg-west-square.json,LABEL=square "$SBATCH"
echo "Chained: $first -> $second -> $third"
