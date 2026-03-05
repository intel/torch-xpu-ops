#!/bin/bash
# P2P cache coherency test - build and run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Specify MPI path explicitly
MPI_HOME=/home/sdp/hanchao/2025.3/intel/oneapi/mpi/2021.17

# Build
icpx -fsycl -O2 \
    -I${MPI_HOME}/include \
    -L${MPI_HOME}/lib -lmpi \
    -Wl,-rpath,${MPI_HOME}/lib \
    -lze_loader \
    -o test_p2p_cache_coherency test_p2p_cache_coherency.cpp

# Run
# export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
${MPI_HOME}/bin/mpirun -np 2 ./test_p2p_cache_coherency

