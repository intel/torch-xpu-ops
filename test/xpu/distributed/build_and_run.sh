#!/usr/bin/env bash
# Build + run native oneshot/CCL bench inside docker container `hanchao`.
#
# Usage: bash build_and_run.sh            # build + run default sizes
#        bash build_and_run.sh build      # build only
set -euo pipefail

cd "$(dirname "$0")"

CCL_ROOT=${CCL_ROOT:-/opt/intel/oneapi/2025.3}
MPI_ROOT=${I_MPI_ROOT:-/opt/intel/oneapi/2025.3}
SYCL_TLA_INC=/root/hanchao/sycl-tla/examples/00_bmg_gemm

icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -std=c++17 \
     -I"${CCL_ROOT}/include" -I"${MPI_ROOT}/include" -I"${SYCL_TLA_INC}" \
     -L"${CCL_ROOT}/lib" -L"${MPI_ROOT}/lib" \
     -lccl -lmpi -lze_loader \
     bench_oneshot_vs_ccl.cpp -o bench_oneshot_vs_ccl

echo "[build] OK -> $(pwd)/bench_oneshot_vs_ccl"

if [[ "${1:-}" == "build" ]]; then
  exit 0
fi

export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0,1,2,3}
# Workaround UCX thread-owner assertion inside this docker image: force CCL to
# use the MPI ATL instead of UCX/OFI.
export CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT:-mpi}
mpirun -n 4 ./bench_oneshot_vs_ccl --min 10 --max 22 --warmup 10 --iters 50
