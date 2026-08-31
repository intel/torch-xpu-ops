#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# --- Runtime environment ----------------------------------------------------
# Source Intel oneAPI (compiler/MPI/ISHMEM/MKL) and select the conda env whose
# torch the prebuilt libinternode_dispatch_rdma_sender.so was linked against
# (hanchao). setvars.sh references unbound vars, so relax `set -u` only around
# the source.
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
#export ISHMEM_IBGDA_BAR_BACKEND=igub
export ISHMEM_IB_TRAFFIC_CLASS="${ISHMEM_IB_TRAFFIC_CLASS:-96}"
export I_MPI_FABRICS=shm
export ISHMEM_DEBUG=0
# Print per-PE debug lines from InternodeDispatchRdmaSender.cpp (launch/
# finalize steps) so a hang can be localized to a specific rank/step.
# Respects a caller-provided override (e.g. INTERNODE_DISPATCH_RDMA_SENDER_DEBUG=0).
export INTERNODE_DISPATCH_RDMA_SENDER_DEBUG="${INTERNODE_DISPATCH_RDMA_SENDER_DEBUG:-1}"

# test_internode_dispatch_rdma_sender.py's default MoE-style random routing
# (NUM_TOKENS=1024, TOPK=8, EXPERTS_PER_RANK=64) makes almost every token hit
# almost every destination node, so the true per-node hit count is typically
# ~900-1024 -- far above the receive buffer's default NUM_MAX_TOKENS_PER_RANK
# (128). When that happens, recv_x/recv_topk_idx/etc. get correctly truncated
# to 128 by the op, but the test's expected-value construction does NOT
# truncate the same way, so the correctness check spuriously fails with a
# "nvl bits mismatch" AssertionError even though the RDMA path is fine. Size
# the receive buffer to comfortably cover the expected hit count so the test
# doesn't hit this overflow.
export NUM_MAX_TOKENS_PER_RANK="${RING_NUM_MAX_TOKENS_PER_RANK:-1024}"

RING_WORLD_SIZE="${RING_WORLD_SIZE:-4}"
# Ranks are grouped into nodes of this many NVL peers each (num_rdma_ranks =
# world_size / NUM_MAX_NVL_PEERS); the compiled-in default is 2, which means a
# 2-rank run puts both ranks on the SAME node (num_rdma_ranks=1) and only ever
# takes the own-node local-copy branch -- no RDMA put is ever issued. Set
# RING_NUM_MAX_NVL_PEERS=1 (e.g. for a 2-GPU smoke test) to make every rank
# its own node so RDMA is actually exercised.
export INTERNODE_DISPATCH_NUM_MAX_NVL_PEERS="${RING_NUM_MAX_NVL_PEERS:-2}"
#GPU_IDS=(0 1 2 3 4 5 6 7)
#NIC_IDS=(0 1 2 3 4 5 6 7)
# Overridable so a run can pick a specific GPU/NIC subset (e.g.
# RING_GPU_IDS="0 2" RING_NIC_IDS="0 2" for a 2-GPU test on GPU0+GPU2 instead
# of the default adjacent GPU0+GPU1 pair).
read -r -a GPU_IDS <<< "${RING_GPU_IDS:-0 1 2 3}"
read -r -a NIC_IDS <<< "${RING_NIC_IDS:-0 1 2 3}"

#GPU_IDS=(4 6)
#NIC_IDS=(4 6)

if (( RING_WORLD_SIZE < 2 || RING_WORLD_SIZE > ${#GPU_IDS[@]} )); then
  echo "RING_WORLD_SIZE must be in [2, ${#GPU_IDS[@]}], got ${RING_WORLD_SIZE}" >&2
  exit 1
fi

# internode_dispatch_rdma_sender groups ranks into nodes of
# INTERNODE_DISPATCH_NUM_MAX_NVL_PEERS PEs, so world_size must be a multiple
# of it.
if (( RING_WORLD_SIZE % INTERNODE_DISPATCH_NUM_MAX_NVL_PEERS != 0 )); then
  echo "RING_WORLD_SIZE must be a multiple of INTERNODE_DISPATCH_NUM_MAX_NVL_PEERS=${INTERNODE_DISPATCH_NUM_MAX_NVL_PEERS}, got ${RING_WORLD_SIZE}" >&2
  exit 1
fi

# InternodeDispatchRdmaSender pins one QP per channel (qp == channel) and
# defaults to kDefaultChannels=8 channels (test_internode_dispatch_rdma_sender.py
# doesn't override num_channels and uses NUM_TOKENS=16 > 8, so num_channels
# stays at 8 regardless of world size) -- ISHMEM must provision at least that
# many QPs/PE or the op aborts with "ISHMEM_IBGDA_QPS_PER_PE must be >=
# num_channels".
#
# A channels sweep (2-rank RDMA, NUM_TOKENS=4096/HIDDEN_SIZE=4096) found 16
# channels to be the sweet spot vs. the compiled-in default of 8 -- ~35-55%
# higher RDMA BW, since the default only launches num_channels*2=16 WGs (~6%
# of this GPU's 256 compute units) while 16 channels (32 WGs) uses more of
# the QP/compute parallelism without hitting the instability seen at 18+.
# Override via RING_CHANNELS/RING_IBGDA_QPS_PER_PE if re-tuning is needed.
default_channels=16
default_qps_per_pe=16
default_db_batch_size=4
export INTERNODE_DISPATCH_RDMA_SENDER_CHANNELS="${RING_CHANNELS:-$default_channels}"
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
# public headers, build helper, or the sender source is newer than the
# existing .so.
ISHMEM_HOME="${ISHMEM_HOME:-/opt/intel/ishmem_ibgda}"
sender_so=../csrc/libinternode_dispatch_rdma_sender.so
sender_src=../csrc/InternodeDispatchRdmaSender.cpp
build_helper=../csrc/build.py
ishmem_static="${ISHMEM_HOME%/}/lib/libishmem.a"
ishmem_include="${ISHMEM_HOME%/}/include"

if [[ ! -f "$ishmem_static" || ! -d "$ishmem_include" ]]; then
  echo "Invalid ISHMEM_HOME=${ISHMEM_HOME}: expected include/ and lib/libishmem.a" >&2
  exit 1
fi

rebuild_sender=0
if [[ ! -f "$sender_so" || "$sender_src" -nt "$sender_so" ||
      "$build_helper" -nt "$sender_so" || "$ishmem_static" -nt "$sender_so" ]]; then
  rebuild_sender=1
elif [[ -n "$(find "$ishmem_include" -type f -newer "$sender_so" -print -quit)" ]]; then
  rebuild_sender=1
fi

if (( rebuild_sender )); then
  echo "[test_ishmem] rebuilding libinternode_dispatch_rdma_sender.so from ${ISHMEM_HOME}"
  ( cd ../csrc && ISHMEM_HOME="$ISHMEM_HOME" python - <<'PY'
import build
cfg = build.get_build_config()
ishmem_cfg = build.get_ishmem_config()
build.build_one_ishmem(
    cfg,
    ishmem_cfg,
    "InternodeDispatchRdmaSender.cpp",
    "libinternode_dispatch_rdma_sender.so",
    "InternodeDispatchRdmaSender",
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
  exec python test_internode_dispatch_rdma_sender.py

# test_token_dispatch_ishmem_hier.py

# test_token_dispatch_ishmem.py 

# test_ring_allgather_ishmem.py
'
