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
export ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP=1
# 3) STRICT=1 makes the IBGDA direct-doorbell NIC path fail-closed: if the
#    NIC/UAR bring-up cannot be completed, ISHMEM errors out instead of
#    silently falling back to a non-NIC path.
export ISHMEM_IBGDA_STRICT=1
# ----------------------------------------------------------------------------
export ISHMEM_SYMMETRIC_SIZE=536870912
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ISHMEM_IBGDA_QPS_PER_PE=1
export ISHMEM_IBGDA_DB_BATCH_SIZE=0
export ISHMEM_IBGDA_BAR_BACKEND=igub
export I_MPI_FABRICS=shm
export ISHMEM_DEBUG=0
#export ZE_AFFINITY_MASK=4,5,6,7

# --- Optional: pin each GPU/rank to a specific NIC --------------------------
# On this node all 4 GPUs (dev 0-3) sit on NUMA/socket 0. There are only 2
# physical NIC cards per socket:
#   socket0: mlx5_0/1 (BDF 31:00.0/.1), mlx5_2/3 (42:00.0/.1)
#   socket1: mlx5_4/5 (b2:00.0/.1),     mlx5_6/7 (ba:00.0/.1)
#
# TUNED RESULT (pure allgather 120MB/PE, best-of-3; end-to-end in []):
#   AUTO  mlx5_0-3 (socket0, near)     10.2 GB/s   [full 14.6 ms]
#   C     mlx5_0,2,4,6 (4 cards)       11.6 GB/s
#   E     mlx5_4-7 (socket1, far)      13.5 GB/s   [full 10.66 ms]  <-- BEST
# Counter-intuitively the socket-1 NICs win by ~27%: with socket-0 NICs the
# NIC DMA contends with GPU memory/compute traffic on the socket-0 PCIe/IIO
# complex; moving NIC DMA to socket 1 offloads that. Correctness verified
# (exact match) under the tuned map.
#
# PIN_NIC=1 uses ISHMEM_NIC_MAP below (rank i -> i-th NIC) via _pin_nic_launch.sh.
# Set PIN_NIC=0 to fall back to ISHMEM's automatic PCIe/NUMA affinity selection.
PIN_NIC="${PIN_NIC:-1}"
export ISHMEM_NIC_MAP="${ISHMEM_NIC_MAP:-mlx5_4 mlx5_5 mlx5_6 mlx5_7}"
# ----------------------------------------------------------------------------

# How to double-check the traffic really went over the NICs (mlx5_0..3):
#   snapshot the counters before/after a run and confirm they grow by the
#   transferred byte count (delta * 4 = bytes; IB counters are in 4-octet
#   units). If data took a PCIe IPC/P2P path, these counters stay flat.
#     for n in mlx5_0 mlx5_1 mlx5_2 mlx5_3; do
#       echo "$n $(cat /sys/class/infiniband/$n/ports/1/counters/port_xmit_data)"
#     done
#   Or run the bundled per-NIC check:
#     mpirun -np 4 --prepend-rank python _nic_traffic_check.py

# Build liballgather_permute_ishmem.so only if it is missing.
export ISHMEM_HOME="${ISHMEM_HOME:-/root/jiafuzha/code-repo/ishmem_ibgda/build/_install}"
if [ ! -f ../csrc/libring_allgather_ishmem.so ]; then
  ( cd ../csrc && python - <<'PY'
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


mpirun -np 2 --prepend-rank python test_ring_allgather_ishmem.py

'''
if [ "$PIN_NIC" = "1" ]; then
  echo "[test_ishmem] PIN_NIC=1  mapping: ISHMEM_NIC_MAP=\"$ISHMEM_NIC_MAP\""
  RING_ALLGATHER_ISHMEM_DEBUG=0 mpirun -np 2 --prepend-rank ./_pin_nic_launch.sh python test_ring_allgather_ishmem.py
else
  echo "[test_ishmem] PIN_NIC=0  (ISHMEM automatic PCIe/NUMA NIC affinity)"
  RING_ALLGATHER_ISHMEM_DEBUG=0 mpirun -np 2 --prepend-rank python test_ring_allgather_ishmem.py
fi
'''
#mpirun -np 4 --prepend-rank python test_allgather_permute_ishmem_mlx5dv_repro.py
