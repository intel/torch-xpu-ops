#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u
export PATH=/root/miniforge3/envs/hanchao/bin:$PATH

export ISHMEM_IB_ENABLE_IBGDA=1
export ISHMEM_IBGDA_DIRECT_DOORBELL=1
export ISHMEM_ENABLE_GPU_IPC=1
export ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP=0
export ISHMEM_SYMMETRIC_SIZE="${ISHMEM_SYMMETRIC_SIZE:-536870912}"
export ISHMEM_IB_TRAFFIC_CLASS="${ISHMEM_IB_TRAFFIC_CLASS:-96}"
export I_MPI_FABRICS="${I_MPI_FABRICS:-shm}"
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export DISPATCH_RDMA_NVL_CHANNELS="${DISPATCH_RDMA_NVL_CHANNELS:-16}"
export DISPATCH_RDMA_NVL_CHUNK_TOKENS="${DISPATCH_RDMA_NVL_CHUNK_TOKENS:-8}"
export DISPATCH_RDMA_NVL_THREADS="${DISPATCH_RDMA_NVL_THREADS:-512}"
export ISHMEM_IBGDA_QPS_PER_PE="${ISHMEM_IBGDA_QPS_PER_PE:-32}"
export ISHMEM_IBGDA_DB_BATCH_SIZE="${ISHMEM_IBGDA_DB_BATCH_SIZE:-4}"
export NUM_MAX_TOKENS_PER_RANK="${NUM_MAX_TOKENS_PER_RANK:-${NUM_TOKENS:-4096}}"

read -r -a GPU_IDS <<< "${RING_GPU_IDS:-0 1 2 3}"
read -r -a NIC_IDS <<< "${RING_NIC_IDS:-0 1 2 3}"
if (( ${#GPU_IDS[@]} < 4 || ${#NIC_IDS[@]} < 4 )); then
  echo "RING_GPU_IDS and RING_NIC_IDS must each contain four entries" >&2
  exit 1
fi

for rank in 0 1 2 3; do
  nic="mlx5_${NIC_IDS[rank]}"
  if [[ ! -d "/sys/class/infiniband/${nic}" ]]; then
    echo "Missing NIC ${nic} required by rank ${rank}" >&2
    exit 1
  fi
done

ISHMEM_HOME="${ISHMEM_HOME:-/opt/intel/ishmem_ibgda}"
output=../csrc/libdispatch_rdma_nvl.so
source_file=../csrc/DispatchRdmaNvl.cpp
build_helper=../csrc/build.py
ishmem_static="${ISHMEM_HOME%/}/lib/libishmem.a"
ishmem_include="${ISHMEM_HOME%/}/include"
if [[ ! -f "$ishmem_static" || ! -d "$ishmem_include" ]]; then
  echo "Invalid ISHMEM_HOME=${ISHMEM_HOME}" >&2
  exit 1
fi

rebuild=0
if [[ ! -f "$output" || "$source_file" -nt "$output" ||
      "$build_helper" -nt "$output" || "$ishmem_static" -nt "$output" ]]; then
  rebuild=1
elif [[ -n "$(find "$ishmem_include" -type f -newer "$output" -print -quit)" ]]; then
  rebuild=1
fi

if (( rebuild )); then
  (cd ../csrc && ISHMEM_HOME="$ISHMEM_HOME" python - <<'PY'
import build

build.build_one_ishmem(
    build.get_build_config(),
    build.get_ishmem_config(),
    "DispatchRdmaNvl.cpp",
    "libdispatch_rdma_nvl.so",
    "DispatchRdmaNvl",
)
PY
  )
fi

echo "[dispatch_rdma_nvl] rank pairs: (0,1) IPC, (2,3) IPC; cross-pair RDMA"
for rank in 0 1 2 3; do
  echo "[dispatch_rdma_nvl] rank ${rank}: GPU ${GPU_IDS[rank]}, mlx5_${NIC_IDS[rank]}"
done

export RING_GPU_IDS="${GPU_IDS[*]}"
export RING_NIC_IDS="${NIC_IDS[*]}"
mpirun -np 4 --prepend-rank bash -c '
  gpu_ids=($RING_GPU_IDS)
  nic_ids=($RING_NIC_IDS)
  export ZE_AFFINITY_MASK=${gpu_ids[PMI_RANK]}
  export ISHMEM_IBGDA_NIC=mlx5_${nic_ids[PMI_RANK]}
  exec python test_dispatch_rdma_nvl.py
'
