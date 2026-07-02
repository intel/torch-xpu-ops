#!/usr/bin/env bash
set -e

export TOKENS_PER_RANK=1024
export HIDDEN_SIZE=2048
export TOPK=8
export NUM_EXPERTS=128
export DTYPE=bfloat16
export LOOP=40
export WARMUP=20

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
export ISHMEM_DEBUG=0
export ZE_AFFINITY_MASK=6,7


mpirun -np 2 --prepend-rank python test_allgather_permute_ishmem_perf.py

'''
timeout 120 mpirun \
    -n 1 -genv ZE_AFFINITY_MASK 6 -genv ISHMEM_IBGDA_NIC mlx5_6 python test_allgather_permute_ishmem_perf.py \
    : \
    -n 1 -genv ZE_AFFINITY_MASK 7 -genv ISHMEM_IBGDA_NIC mlx5_7 python test_allgather_permute_ishmem_perf.py
'''
