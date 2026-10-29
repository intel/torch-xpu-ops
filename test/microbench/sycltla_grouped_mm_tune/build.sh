#!/bin/bash
# Build the standalone SYCL-TLA grouped GEMM tuner (no PyTorch needed).
#
#   SYCL_TLA_DIR=/path/to/sycl-tla [AOT_DEVICE=pvc] [CONFIGS="p256x256 p128x256"] ./build.sh [outdir]
#
# Needs icpx (oneAPI DPC++ 2025+) on PATH and a sycl-tla checkout (v0.9, the
# tag torch-xpu-ops pins). Flags mirror the torch-xpu-ops-sycltla target.
# Each config becomes libtune_cfg_<name>.so, compiled in parallel; configs
# that fail to compile are reported and skipped.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../../../src/ATen/native/xpu/grouped_mm/sycltla" && pwd)
TLA=${SYCL_TLA_DIR:?set SYCL_TLA_DIR to a sycl-tla checkout (v0.9)}
AOT_DEVICE=${AOT_DEVICE:-pvc}
OUT=${1:-$HERE/build}
JOBS=${JOBS:-16}
ALL=$(sed -n 's/^TUNE_CFG(\([a-z0-9_x]*\),.*/\1/p' "$HERE/tune_configs.h")
CONFIGS=${CONFIGS:-$ALL}
[ -d "$TLA/include/cute" ] || { echo "SYCL_TLA_DIR=$TLA does not look like a sycl-tla checkout"; exit 1; }
command -v icpx >/dev/null || { echo "icpx not on PATH (load oneAPI first)"; exit 1; }
mkdir -p "$OUT/logs"

COMMON=(-fsycl -std=c++20 -O3 -fPIC
        -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET
        -I"$TLA/include" -I"$TLA/tools/util/include" -I"$TLA/applications" -I"$SRC" -I"$HERE/../../../src" -I"$HERE")
DEVICE=(-fsycl-targets=spir64_gen -Xs "-device $AOT_DEVICE"
        -Xspirv-translator "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate")

echo "sycl-tla: $TLA   device: $AOT_DEVICE   out: $OUT"
echo "configs: $(tr '\n' ' ' <<< "$CONFIGS")"
t0=$(date +%s)
build_cfg() {
  local c=$1
  if icpx "${COMMON[@]}" "${DEVICE[@]}" -shared -DTUNE_CFG_NAME="$c" \
       "$HERE/tune_cfg.cpp" -o "$OUT/libtune_cfg_$c.so" > "$OUT/logs/$c.log" 2>&1; then
    local spill
    spill=$(grep -o "spilled around [0-9]*" "$OUT/logs/$c.log" | sort -u | tr '\n' ' ')
    echo "  ok    $c ${spill:+(register spills: $spill)}"
  else
    rm -f "$OUT/libtune_cfg_$c.so"
    echo "  FAIL  $c  (see $OUT/logs/$c.log)"
  fi
}
export -f build_cfg; export HERE OUT
export COMMON_STR="${COMMON[*]}"
for c in $CONFIGS; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 1; done
  build_cfg "$c" &
done
if icpx "${COMMON[@]}" "${DEVICE[@]}" "$HERE/tune_grouped_mm.cpp" "$SRC/grouped_mm_util.cpp" \
     -ldl -o "$OUT/tune_grouped_mm" > "$OUT/logs/driver.log" 2>&1; then
  echo "  ok    driver"
else
  echo "  FAIL  driver (see $OUT/logs/driver.log)"
fi
wait
echo "built in $(( $(date +%s) - t0 )) s:  $OUT/tune_grouped_mm --list"
