#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

. /root/cherry/ishmem_ws/ishmem_ibgda/build/_install/env/vars.sh

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
export ISHMEM_DEBUG=1
export ZE_AFFINITY_MASK=4,5,6,7

mpirun -np 4 --prepend-rank python test_allgather_permute_ishmem_mlx5dv_repro.py
