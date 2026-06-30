#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

: "${ISHMEM_HOME:=/root/cherry/ep_ws/ishmem_ibgda/build}"
: "${NPES:=2}"
: "${TIMEOUT:=60s}"
: "${RMA_BYTES:=4194304}"
: "${RMA_CHUNK_BYTES:=4096}"
: "${QUIET_REPRO_ITERS:=2}"
: "${ZE_AFFINITY_MASK:=6,7}"

MPI_FLAGS="$(
  pkg-config --cflags --libs impi 2>/dev/null ||
    echo -I/opt/intel/oneapi/mpi/2021.18/include \
      -L/opt/intel/oneapi/mpi/2021.18/lib -lmpi -lmpicxx -lmpifort
)"

icpx -fsycl -std=c++17 -O2 -I"${ISHMEM_HOME}/include" \
  ishmem_quiet_hang_repro.cpp -o ishmem_quiet_hang_repro \
  "${ISHMEM_HOME}/src/libishmem.a" ${MPI_FLAGS} -lze_loader

export RMA_BYTES
export RMA_CHUNK_BYTES
export QUIET_REPRO_ITERS
export ZE_AFFINITY_MASK
export ISHMEM_IB_ENABLE_IBGDA=1
export ISHMEM_IBGDA_DIRECT_DOORBELL=1
export ISHMEM_ENABLE_GPU_IPC=0
export ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP=1
export ISHMEM_SYMMETRIC_SIZE=536870912
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ISHMEM_IBGDA_QPS_PER_PE=1
export ISHMEM_IBGDA_DB_BATCH_SIZE=0
export ISHMEM_IBGDA_BAR_BACKEND=igub
export I_MPI_FABRICS=shm

timeout "${TIMEOUT}" mpirun -np "${NPES}" --prepend-rank ./ishmem_quiet_hang_repro
