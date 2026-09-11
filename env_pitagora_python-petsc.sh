#!/bin/bash
# Pitagora runtime environment for petsc-python-miniapp
# Does NOT source env_pitagora_build.sh (module purge not needed at runtime)

export PETSC_DIR=/pitagora/home/userexternal/mpeybern/petsc-install-python
export PETSC_ARCH=
export PYTHONPATH=$PETSC_DIR/lib:$PYTHONPATH

source /pitagora/home/userexternal/mpeybern/petsc-python-miniapp/myenv/bin/activate

# CUDA 12.9 (nvhpc bundled) for runtime
NVHPC_BASE=${NVHPC_HOME:-/pitagora/prod/spack/6.1/install/0.22/linux-rhel9-zen4/gcc-11.4.1/nvhpc-25.11-ntshdsgl52b6ckb6iu7xazd6uvqi3wqi/Linux_x86_64/25.11}
CUDA126_LIB=/pitagora/prod/spack/6.1/install/0.22/linux-rhel9-zen4/gcc-11.4.1/cuda-12.6.0-mpuxcbikjk6ksvpkfapavgd7s27by2ac/lib64

# Fix nvhpc OpenMPI hardcoded /proj/nv path
OMPI_ROOT=$NVHPC_BASE/comm_libs/13.0/hpcx/hpcx-2.25.1/ompi
export OPAL_PREFIX=$OMPI_ROOT

export LD_LIBRARY_PATH=$PETSC_DIR/lib:$OMPI_ROOT/lib:$NVHPC_BASE/cuda/12.9/lib64:$NVHPC_BASE/math_libs/12.9/targets/x86_64-linux/lib:${CUDA126_LIB}:${LD_LIBRARY_PATH:-}

# GPU Direct
export MPICH_GPU_SUPPORT_ENABLED=1
export OMPI_MCA_opal_cuda_support=true
export UCX_TLS=rc_x,cuda_copy
export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=yes
