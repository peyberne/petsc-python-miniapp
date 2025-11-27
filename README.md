# petsc-python-miniapp
miniapp testing petsc solvers

# 1- load modules 
module load nvhpc
module load cuda
module load cmake
module load hdf5/1.14.3-mpi 

# source python
/ssoft/spack/pinot-noir/kuma-h100/v1/spack/opt/spack/linux-rhel9-zen4/gcc-13.2.0/python-3.11.7-wpgsyqek7spdydbmic66srcfb3v7kzoi/bin/python3.11 -m venv myenv
source myenv/bin/activate
pip install numpy
pip install 'Cython>=3.0.0,<3.1.0'

# 2- install petsc
git clone v3.21.4
./configure --with-clean --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --with-cudac=nvcc --with-cuda-arch=90  --with-debugging=0 --with-petsc4py=1 --prefix=/scratch/peyberne/petsc-install-python
make
export CFLAGS="-O3 -fPIC"
export CPPFLAGS="-DMPICH_SKIP_MPICXX=1 -DOMPI_SKIP_MPICXX=1"
make install
python3 -c "import petsc4py; print(petsc4py.__version__)"

# 3- copy matrix, RHS and initial guess
cp ../petsc_miniapp/python_test_solver/mat mat.dat
cp ../petsc_miniapp/python_test_solver/rhs rhs.dat
cp ../petsc_miniapp/python_test_solver/sol sol.dat

# 5- run test
sbatch submission_script.sh