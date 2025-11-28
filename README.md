# petsc-python-miniapp

A small Python mini-application for testing PETSc solvers, built with CUDA, MPI, and petsc4py support.  
This README provides all the steps needed to set up the environment, install PETSc, prepare input data, and run the test on the target HPC system.

---

## 1. Load required modules

```
module load nvhpc
module load cuda
module load cmake
module load hdf5/1.14.3-mpi
```

---

## 2. Create and activate the Python environment

```
/ssoft/spack/pinot-noir/kuma-h100/v1/spack/opt/spack/linux-rhel9-zen4/gcc-13.2.0/python-3.11.7-wpgsyqek7spdydbmic66srcfb3v7kzoi/bin/python3.11 -m venv myenv
source myenv/bin/activate

pip install numpy
pip install 'Cython>=3.0.0,<3.1.0'
```

---

## 3. Install PETSc (with CUDA + petsc4py)

Checkout PETSc version:

```
git checkout v3.21.4
```

Configure PETSc:

```
./configure   --with-clean   --with-cc=mpicc   --with-cxx=mpicxx   --with-fc=mpif90   --with-cuda=1   --with-cudac=nvcc   --with-cuda-arch=90   --with-debugging=0   --with-petsc4py=1   --pre
fix=/scratch/peyberne/petsc-install-python
```

Build and install:

```
make
export CFLAGS=$(echo $CFLAGS | sed 's/-fwrapv//g')
make install
```

Verify petsc4py:

```
python3 -c "import petsc4py; print(petsc4py.__version__)"
```

---

## 4. Copy matrix, RHS, and initial guess

```
cp ../petsc_miniapp/python_test_solver/mat mat.dat
cp ../petsc_miniapp/python_test_solver/rhs rhs.dat
cp ../petsc_miniapp/python_test_solver/sol sol.dat
```

---

## 5. Run the test

```
sbatch submission_script.sh
```

---

## Notes

- Ensure that the PETSc installation directory (`petsc-install-python`) matches the configured prefix.
- petsc4py must be installed through PETSc’s configure step, not via pip.
- All steps assume an HPC environment with SLURM and NVHPC modules.
