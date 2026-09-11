#!/bin/bash
set -euo pipefail
SCRIPT_DIR=${BASH_SOURCE[0]%/*}
source "$SCRIPT_DIR/env_pitagora_build.sh"

PETSC_SRC=/pitagora/home/userexternal/mpeybern/petsc
PETSC_PREFIX=/pitagora/home/userexternal/mpeybern/petsc-install-python

cd "$PETSC_SRC"

echo "Cleaning previous build..."
make PETSC_ARCH=arch-linux-c-opt clean 2>/dev/null || true
rm -rf arch-linux-c-opt configure.log

# nvhpc 25.11 doesn't bundle nvToolsExt; CUDA 12.6 has it.
# Use merged CUDA dir with both CUDA 12.9 + nvToolsExt from 12.6
CUDA_DIR=/pitagora/home/userexternal/mpeybern/cuda-12.9-merged
CUDA126_LIB64=/pitagora/prod/spack/6.1/install/0.22/linux-rhel9-zen4/gcc-11.4.1/cuda-12.6.0-mpuxcbikjk6ksvpkfapavgd7s27by2ac/lib64

echo "Configuring PETSc..."
./configure \
  --with-clean \
  --with-cc=mpicc \
  --with-cxx=mpicxx \
  --with-fc=mpif90 \
  --with-cuda=1 \
  --with-cudac=nvcc \
  --with-cuda-dir="$CUDA_DIR" \
  --with-cuda-arch=90 \
  --with-debugging=0 \
  --with-petsc4py=1 \
  --ignoreLinkOutput=1 \
  "LDFLAGS=-L$CUDA126_LIB64 -Wl,-rpath,$CUDA126_LIB64" \
  --prefix="$PETSC_PREFIX"

echo "Building PETSc..."
make PETSC_DIR="$PETSC_SRC" PETSC_ARCH=arch-linux-c-opt all

echo "Installing PETSc..."
source /pitagora/home/userexternal/mpeybern/petsc-python-miniapp/myenv/bin/activate

# Fix petscrules: replace hardcoded /usr/bin/python3 with venv python
VENV_PYTHON=$(which python3)
sed -i "s|/usr/bin/python3|$VENV_PYTHON|g" "$PETSC_SRC/arch-linux-c-opt/lib/petsc/conf/petscrules"

# nvc doesn't support -fwrapv (GCC flag from Python sysconfig)
export CC=nvc CXX=nvc++ FC=nvfortran
export CFLAGS="-O -DNDEBUG" CXXFLAGS="-O -DNDEBUG" FOPTFLAGS="-O"

make PETSC_DIR="$PETSC_SRC" PETSC_ARCH=arch-linux-c-opt install

echo "=========================================="
echo "Done. Source env_pitagora_python-petsc.sh to use."
echo "=========================================="
