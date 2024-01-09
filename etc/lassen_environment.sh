module load gcc/12.2.1 cuda/12.2.2 cmake/3.23.1
export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_ACCEL_TARGET=nvidia70
export AMREX_CUDA_ARCH=7.0
export CC=gcc
export CXX=g++
export FC=gfortran
export CUDACXX=$(which nvcc)
export CUDAHOSTCXX=g++
