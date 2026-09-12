#!/bin/bash
# Pitagora runtime environment for petsc-python-miniapp
module purge
module load nvhpc/25.11
module load petsc/3.22.0--hpcx-mpi--2.25.1--nvhpc--25.11-mumps

export PETSC_ARCH=
source /pitagora/home/userexternal/mpeybern/petsc-python-miniapp/myenv/bin/activate

NVHPC_BASE=/pitagora/prod/spack/6.1/install/0.22/linux-rhel9-zen4/gcc-11.4.1/nvhpc-25.11-ntshdsgl52b6ckb6iu7xazd6uvqi3wqi/Linux_x86_64/25.11

# Fix nvhpc OpenMPI hardcoded /proj/nv path
OMPI_ROOT=$NVHPC_BASE/comm_libs/13.0/hpcx/hpcx-2.25.1/ompi
export OPAL_PREFIX=$OMPI_ROOT

export LD_LIBRARY_PATH=$OMPI_ROOT/lib:${LD_LIBRARY_PATH:-}

# GPU Direct
export MPICH_GPU_SUPPORT_ENABLED=1
export OMPI_MCA_opal_cuda_support=true
export UCX_TLS=rc_x,cuda_copy
export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=yes
