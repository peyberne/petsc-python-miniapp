module load nvhpc
module load cuda
module load cmake
module load hdf5/1.14.3-mpi 
export PETSC_DIR=/home/peyberne/Codes/petsc-install-3.21.4-python
export PETSC_ROOT=/home/peyberne/Codes/petsc-install-3.21.4-python
export PETSC_HOME=/home/peyberne/Codes/petsc-install-3.21.4-python
. /home/peyberne/spack/share/spack/setup-env.sh 
spack load json-fortran@8.3.0%nvhpc@24.7
source /home/peyberne/Codes/petsc-python-miniapp/myenv/bin/activate
export PYTHONPATH=$PETSC_DIR/$PETSC_ARCH/lib:$PYTHONPATH
# GPU DIRECT
# For MPICH/Cray MPICH:
export MPICH_GPU_SUPPORT_ENABLED=1
# For Open MPI:
export OMPI_MCA_opal_cuda_support=true   # some builds still look at this

export UCX_TLS=rc_x,cuda_copy
export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=yes
