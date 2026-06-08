#!/usr/bin/env bash
# Build + run native oneCCL allreduce / allgather / reducescatter benchmark.
#
# Usage:
#   bash build_and_run_ccl.sh              # build + run with defaults (ws=4)
#   bash build_and_run_ccl.sh build        # build only
#   WS=8 ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7 bash build_and_run_ccl.sh
#
# Environment overrides (all optional):
#   WS                      number of MPI ranks (default 4)
#   ZE_AFFINITY_MASK        e.g. 0,1,2,3  (default: 0..WS-1)
#   CCL_ATL_TRANSPORT       mpi | ofi  (default mpi — avoids UCX issues in docker)
#   CCL_ROOT                path to oneCCL install (default /opt/intel/oneapi/2025.3)
#   I_MPI_ROOT              path to Intel MPI install (default $CCL_ROOT)
set -euo pipefail

cd "$(dirname "$0")"

CCL_ROOT=${CCL_ROOT:-/opt/intel/oneapi/2025.3}
MPI_ROOT=${I_MPI_ROOT:-${CCL_ROOT}}

icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -std=c++17 \
     -I"${CCL_ROOT}/include" \
     -I"${MPI_ROOT}/include" \
     -L"${CCL_ROOT}/lib" \
     -L"${MPI_ROOT}/lib" \
     -lccl -lmpi \
     bench_ccl_collectives.cpp -o bench_ccl_collectives

echo "[build] OK -> $(pwd)/bench_ccl_collectives"

if [[ "${1:-}" == "build" ]]; then
    exit 0
fi

WS=${WS:-4}

# Build a default affinity mask 0,1,...,WS-1 if not set
if [[ -z "${ZE_AFFINITY_MASK:-}" ]]; then
    MASK=$(seq -s, 0 $((WS - 1)))
    export ZE_AFFINITY_MASK="${MASK}"
fi

export CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT:-mpi}

echo "[run]   WS=${WS}  ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}  CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT}"
echo "[run]   args: ${BENCH_ARGS:---min 7 --max 20 --warmup 20 --iters 100}"
echo ""

# Extra args can be passed via BENCH_ARGS, e.g.:
#   BENCH_ARGS="--op ar --min 10 --max 20" bash build_and_run_ccl.sh
# shellcheck disable=SC2086
mpirun -n "${WS}" ./bench_ccl_collectives ${BENCH_ARGS:---min 7 --max 20 --warmup 20 --iters 100}
