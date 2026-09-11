#!/bin/bash
# Pitagora build environment — fixes nvhpc 25.11 CUDA 13.0 path contamination
module purge
module load gcc/12.3.0
module load nvhpc/25.11
module load cmake/3.27.9
module load "hdf5/1.14.3--hpcx-mpi--2.25.1--nvhpc--25.11"

NVHPC_BASE=$NVHPC_HOME/Linux_x86_64/25.11

# Strip CUDA 13.0 paths injected by the nvhpc module
CPATH_CLEAN=$(echo "$CPATH" | sed 's|[^ :]*math_libs/13.0[^ :]*||g; s|[^ :]*cuda/13.0[^ :]*||g; s|^:||; s|:$||; s|::|:|g')
LD_CLEAN=$(echo "$LD_LIBRARY_PATH" | sed 's|[^ :]*math_libs/13.0[^ :]*||g; s|[^ :]*cuda/13.0[^ :]*||g; s|^:||; s|:$||; s|::|:|g')

# Find libnvToolsExt from system CUDA 12.6 (not bundled in nvhpc 25.11)
CUDA126_LIB=$(dirname "$(find /pitagora/prod/spack -path "*/cuda-12.6.0*/lib64/libnvToolsExt.so" -print -quit 2>/dev/null)" 2>/dev/null)

# Use only CUDA 12.9 (nvhpc bundled default)
export CPATH=$NVHPC_BASE/cuda/12.9/include:$NVHPC_BASE/math_libs/12.9/targets/x86_64-linux/include:${CPATH_CLEAN}
export LD_LIBRARY_PATH=$NVHPC_BASE/cuda/12.9/lib64:$NVHPC_BASE/math_libs/12.9/targets/x86_64-linux/lib:${CUDA126_LIB:+$CUDA126_LIB}:${LD_CLEAN}
export CUDA_HOME=$NVHPC_BASE/cuda/12.9
