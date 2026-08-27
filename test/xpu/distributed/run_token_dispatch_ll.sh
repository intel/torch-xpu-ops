#!/usr/bin/env bash
set -uo pipefail
cd /root/cherry/torch-xpu-ops/test/xpu/distributed

export ISHMEM_IB_ENABLE_IBGDA=1
export ISHMEM_IBGDA_DIRECT_DOORBELL=1
export ISHMEM_ENABLE_GPU_IPC=0
export ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP=0
export ISHMEM_SYMMETRIC_SIZE=536870912
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ISHMEM_IBGDA_BAR_BACKEND=igub
export ISHMEM_IB_TRAFFIC_CLASS="${ISHMEM_IB_TRAFFIC_CLASS:-96}"
export I_MPI_FABRICS=shm
export ISHMEM_DEBUG=0

WORLD_SIZE=4
# Reproduce BK's multi-QP fan-out slowdown by default (opt-in toggle in
# TokenDispatchIshmem.cpp): qp_id = local_expert instead of round-robin.
export TOKEN_DISPATCH_QP_PER_EXPERT="${TOKEN_DISPATCH_QP_PER_EXPERT:-1}"
export ISHMEM_IBGDA_QPS_PER_PE="${ISHMEM_IBGDA_QPS_PER_PE:-16}"
export ISHMEM_IBGDA_DB_BATCH_SIZE="${ISHMEM_IBGDA_DB_BATCH_SIZE:-4}"

export TOKENS_PER_RANK="${TOKENS_PER_RANK:-8}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
export TOPK="${TOPK:-4}"
export NUM_EXPERTS="${NUM_EXPERTS:-64}"

GPU_IDS=(0 2 4 6)
NIC_IDS=(0 2 4 6)
export _TD_GPU_IDS="${GPU_IDS[*]}"
export _TD_NIC_IDS="${NIC_IDS[*]}"

TORCH_LIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

mpirun -np "${WORLD_SIZE}" --prepend-rank bash -c '
  gpu_ids=($_TD_GPU_IDS)
  nic_ids=($_TD_NIC_IDS)
  export ZE_AFFINITY_MASK=${gpu_ids[PMI_RANK]}
  export ISHMEM_IBGDA_NIC=mlx5_${nic_ids[PMI_RANK]}
  exec python test_token_dispatch_ishmem.py
'
