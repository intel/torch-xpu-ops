#!/bin/bash
# XCCL Benchmark Runner
#
# Usage:
#   ./run_benchmark.sh                          # defaults: ws=2,4,8  ops=all  dtype=bfloat16
#   ./run_benchmark.sh -w 2,4 -o allreduce,allgather
#   ./run_benchmark.sh -w 8 -d float32 -n 100 --min 12 --max 30 -s 15
#   ./run_benchmark.sh -h

set -e

# ---- defaults ----
WORLD_SIZES="2,4,8"
OPS=""                          # empty = all ops
DTYPE="bfloat16"
NUM_ITERS=50
NUM_WARMUP=10
MIN_SIZE=10                     # 2^10 = 1KB
MAX_SIZE=28                     # 2^28 = 256MB
SIZE_STEPS=10
OUTPUT_DIR=""                   # auto-generated if empty

usage() {
    cat <<EOF
XCCL Benchmark Runner

Options:
  -w, --world-sizes   Comma-separated world sizes          (default: 2,4,8)
  -o, --ops           Comma-separated ops to benchmark     (default: all)
                      Available: broadcast,allreduce,reduce,allgather,
                      reduce_scatter,alltoall_single,alltoall,gather,
                      scatter,send_recv,batch_isend_irecv,barrier
  -d, --dtype         Data type: float32|float16|bfloat16|int32  (default: bfloat16)
  -n, --num-iters     Measured iterations                  (default: 50)
      --num-warmup    Warmup iterations                    (default: 10)
      --min           Min message size as 2^N bytes        (default: 10, i.e. 1KB)
      --max           Max message size as 2^N bytes        (default: 28, i.e. 256MB)
  -s, --size-steps    Number of size steps                 (default: 10)
      --output-dir    Output directory for CSVs            (default: ./benchmark_results/<timestamp>)
  -h, --help          Show this help

Examples:
  ./run_benchmark.sh
  ./run_benchmark.sh -w 2 -o allreduce,reduce_scatter -d bfloat16
  ./run_benchmark.sh -w 2,4,8 -n 100 --min 14 --max 30 -s 20
EOF
    exit 0
}

# ---- parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--world-sizes)  WORLD_SIZES="$2";  shift 2 ;;
        -o|--ops)          OPS="$2";           shift 2 ;;
        -d|--dtype)        DTYPE="$2";         shift 2 ;;
        -n|--num-iters)    NUM_ITERS="$2";     shift 2 ;;
        --num-warmup)      NUM_WARMUP="$2";    shift 2 ;;
        --min)             MIN_SIZE="$2";      shift 2 ;;
        --max)             MAX_SIZE="$2";      shift 2 ;;
        -s|--size-steps)   SIZE_STEPS="$2";    shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";    shift 2 ;;
        -h|--help)         usage ;;
        *)  echo "Unknown option: $1"; usage ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/bench_c10d_xccl.py"

if [ -z "${OUTPUT_DIR}" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${SCRIPT_DIR}/benchmark_results/${TIMESTAMP}"
fi
mkdir -p "${OUTPUT_DIR}"

NUM_DEVICES=$(python3 -c "import torch; print(torch.xpu.device_count())")

echo "============================================"
echo "  XCCL Benchmark"
echo "  XPU devices : ${NUM_DEVICES}"
echo "  world_sizes : ${WORLD_SIZES}"
echo "  ops         : ${OPS:-all}"
echo "  dtype       : ${DTYPE}"
echo "  iters       : ${NUM_ITERS}  warmup: ${NUM_WARMUP}"
echo "  sizes       : 2^${MIN_SIZE} ~ 2^${MAX_SIZE}  (${SIZE_STEPS} steps)"
echo "  output      : ${OUTPUT_DIR}"
echo "============================================"
echo ""

# build extra args
EXTRA_ARGS=""
if [ -n "${OPS}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --ops ${OPS}"
fi

IFS=',' read -ra WS_ARRAY <<< "${WORLD_SIZES}"
for WS in "${WS_ARRAY[@]}"; do
    if [ "${WS}" -gt "${NUM_DEVICES}" ]; then
        echo "[SKIP] world_size=${WS} > available devices (${NUM_DEVICES})"
        continue
    fi

    CSV_FILE="${OUTPUT_DIR}/xccl_ws${WS}_${DTYPE}.csv"
    echo "========== world_size=${WS} =========="
    torchrun --standalone --nproc-per-node "${WS}" "${BENCH_SCRIPT}" \
        --dtype "${DTYPE}" \
        --num-iters "${NUM_ITERS}" \
        --num-warmup "${NUM_WARMUP}" \
        --min-size "${MIN_SIZE}" \
        --max-size "${MAX_SIZE}" \
        --size-steps "${SIZE_STEPS}" \
        --export-csv "${CSV_FILE}" \
        ${EXTRA_ARGS}
    echo "Saved: ${CSV_FILE}"
    echo ""
done

echo "All benchmarks complete. Results in: ${OUTPUT_DIR}"
