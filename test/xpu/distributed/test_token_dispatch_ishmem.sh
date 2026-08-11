#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# --- Runtime environment ----------------------------------------------------
# Source Intel oneAPI (compiler/MPI/ISHMEM/MKL) and select the conda env whose
# torch the prebuilt libring_allgather_ishmem.so was linked against (hanchao).
# setvars.sh references unbound vars, so relax `set -u` only around the source.
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u
export PATH=/root/miniforge3/envs/hanchao/bin:$PATH
# ----------------------------------------------------------------------------

#. /root/cherry/ishmem_ws/ishmem_ibgda/build/_install/env/vars.sh

export ISHMEM_IB_ENABLE_IBGDA=1
export ISHMEM_IBGDA_DIRECT_DOORBELL=1
# --- NIC-only guarantee (no PCIe IPC / P2P) ---------------------------------
# 1) Disable GPU IPC outright.
export ISHMEM_ENABLE_GPU_IPC=0
# 2) accessible_host_heap=1 additionally forces ISHMEM to disable IPC
#    ("Disabling IPC - it is unsupported when shared heap is enabled").
export ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP=0
# 3) STRICT=1 makes the IBGDA direct-doorbell NIC path fail-closed: if the
#    NIC/UAR bring-up cannot be completed, ISHMEM errors out instead of
#    silently falling back to a non-NIC path.
# ----------------------------------------------------------------------------
export ISHMEM_SYMMETRIC_SIZE=536870912
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ISHMEM_IBGDA_BAR_BACKEND=igub
export ISHMEM_IB_TRAFFIC_CLASS="${ISHMEM_IB_TRAFFIC_CLASS:-96}"
export I_MPI_FABRICS=shm
export ISHMEM_DEBUG=0
# Print per-PE debug lines from RingAllgatherIshmem.cpp (ring launch/finalize
# steps) so a hang can be localized to a specific rank/step.
export RING_ALLGATHER_ISHMEM_DEBUG=1

RING_WORLD_SIZE="${RING_WORLD_SIZE:-4}"
#GPU_IDS=(0 1 2 3 4 5 6 7)
#NIC_IDS=(0 1 2 3 4 5 6 7)
GPU_IDS=(0 1 2 3)
NIC_IDS=(0 1 2 3)

#GPU_IDS=(4 6)
#NIC_IDS=(4 6)

if (( RING_WORLD_SIZE < 2 || RING_WORLD_SIZE > ${#GPU_IDS[@]} )); then
  echo "RING_WORLD_SIZE must be in [2, ${#GPU_IDS[@]}], got ${RING_WORLD_SIZE}" >&2
  exit 1
fi

if (( RING_WORLD_SIZE == 2 )); then
  default_qps_per_pe=1
  default_db_batch_size=1
else
  default_qps_per_pe=2
  default_db_batch_size=1
fi
export ISHMEM_IBGDA_QPS_PER_PE="${RING_IBGDA_QPS_PER_PE:-$default_qps_per_pe}"
export ISHMEM_IBGDA_DB_BATCH_SIZE="${RING_IBGDA_DB_BATCH_SIZE:-$default_db_batch_size}"

# Keep each rank on a distinct PCIe-local GPU/NIC port:
# rank 0..7 -> GPU 0..7 / mlx5_0..7. The pairs are 1a/20, 1e/20,
# 3a/3c, 3f/3c, 9a/a0, 9e/a0, ba/c0, and be/c0 (GPU/NIC BDF prefixes).
for ((rank = 0; rank < RING_WORLD_SIZE; ++rank)); do
  nic="mlx5_${NIC_IDS[rank]}"
  if [[ ! -d "/sys/class/infiniband/${nic}" ]]; then
    echo "Missing NIC ${nic} required by rank ${rank}" >&2
    exit 1
  fi
done

echo "[test_ishmem] world_size=${RING_WORLD_SIZE}"
echo "[test_ishmem] ISHMEM_IBGDA_QPS_PER_PE=${ISHMEM_IBGDA_QPS_PER_PE}"
echo "[test_ishmem] ISHMEM_IBGDA_DB_BATCH_SIZE=${ISHMEM_IBGDA_DB_BATCH_SIZE}"
for ((rank = 0; rank < RING_WORLD_SIZE; ++rank)); do
  echo "[test_ishmem] rank ${rank}: GPU ${GPU_IDS[rank]}, mlx5_${NIC_IDS[rank]}"
done

# The extension statically links libishmem.a, so relink whenever the library,
# public headers, build helper, or ring source is newer than the existing .so.
ISHMEM_HOME="${ISHMEM_HOME:-/opt/intel/ishmem}"
ring_so=../csrc/libring_allgather_ishmem.so
ring_src=../csrc/RingAllgatherIshmem.cpp
build_helper=../csrc/build.py
ishmem_static="${ISHMEM_HOME%/}/lib/libishmem.a"
ishmem_include="${ISHMEM_HOME%/}/include"

if [[ ! -f "$ishmem_static" || ! -d "$ishmem_include" ]]; then
  echo "Invalid ISHMEM_HOME=${ISHMEM_HOME}: expected include/ and lib/libishmem.a" >&2
  exit 1
fi

rebuild_ring=0
if [[ ! -f "$ring_so" || "$ring_src" -nt "$ring_so" ||
      "$build_helper" -nt "$ring_so" || "$ishmem_static" -nt "$ring_so" ]]; then
  rebuild_ring=1
elif [[ -n "$(find "$ishmem_include" -type f -newer "$ring_so" -print -quit)" ]]; then
  rebuild_ring=1
fi

if (( rebuild_ring )); then
  echo "[test_ishmem] rebuilding libring_allgather_ishmem.so from ${ISHMEM_HOME}"
  ( cd ../csrc && ISHMEM_HOME="$ISHMEM_HOME" python - <<'PY'
import build
cfg = build.get_build_config()
ishmem_cfg = build.get_ishmem_config()
build.build_one_ishmem(
    cfg,
    ishmem_cfg,
    "RingAllgatherIshmem.cpp",
    "libring_allgather_ishmem.so",
    "RingAllgatherIshmem",
)
PY
  )
fi


# Same-node ring with one explicit PCIe-local GPU/NIC pair per rank.
export RING_GPU_IDS="${GPU_IDS[*]}"
export RING_NIC_IDS="${NIC_IDS[*]}"
mpirun -np "${RING_WORLD_SIZE}" --prepend-rank bash -c '
  gpu_ids=($RING_GPU_IDS)
  nic_ids=($RING_NIC_IDS)
  export ZE_AFFINITY_MASK=${gpu_ids[PMI_RANK]}
  export ISHMEM_IBGDA_NIC=mlx5_${nic_ids[PMI_RANK]}
  exec python test_token_dispatch_ishmem.py 

# test_ring_allgather_ishmem.py
'
