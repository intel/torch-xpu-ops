#!/bin/bash
# Run a single clang-tidy check on all files and show verbose diagnostics.
# Usage: ./run-clang-tidy-check.sh <check-name> [--fix] [--jobs N] [file...]
#
# Examples:
#   ./run-clang-tidy-check.sh performance-unnecessary-copy-initialization
#   ./run-clang-tidy-check.sh modernize-concat-nested-namespaces --fix
#   ./run-clang-tidy-check.sh bugprone-narrowing-conversions src/ATen/native/xpu/Sorting.cpp

PYTORCH_BUILD="${PYTORCH_BUILD:-/home/gta/pytorch/build}"
CLANG_TIDY="${CLANG_TIDY:-clang-tidy-18}"
JOBS=$(nproc)
FIX=""
CHECK=""
USER_FILES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fix)      FIX="--fix"; shift;;
        --jobs|-j)  JOBS="$2"; shift 2;;
        -*)         echo "Unknown flag: $1"; exit 1;;
        *)
            if [[ -z "$CHECK" ]]; then
                CHECK="$1"
            else
                USER_FILES+=("$(realpath "$1")")
            fi
            shift;;
    esac
done

if [[ -z "$CHECK" ]]; then
    echo "Usage: $0 <check-name> [--fix] [--jobs N] [file...]"
    echo ""
    echo "Examples:"
    echo "  $0 performance-unnecessary-copy-initialization"
    echo "  $0 modernize-concat-nested-namespaces --fix"
    echo "  $0 bugprone-narrowing-conversions src/ATen/native/xpu/Sorting.cpp"
    exit 1
fi

# Get file list
if [[ ${#USER_FILES[@]} -gt 0 ]]; then
    FILES=("${USER_FILES[@]}")
else
    mapfile -t FILES < <(python3 -c "
import json
with open('${PYTORCH_BUILD}/compile_commands.json') as f:
    data = json.load(f)
files = sorted({e['file'] for e in data if 'torch-xpu-ops' in e['file'] and '/sycl/' not in e['file'] and '/sycltla/' not in e['file']})
for f in files: print(f)
")
fi

TOTAL=${#FILES[@]}
echo "=== Check: $CHECK ==="
echo "=== Files: $TOTAL, Jobs: $JOBS ${FIX:+, FIX MODE} ==="
echo ""

FOUND=0
IDX=0

run_one() {
    local file="$1"
    local short="${file##*/torch-xpu-ops/}"

    local output
    output=$("$CLANG_TIDY" -p "$PYTORCH_BUILD" -checks="-*,$CHECK" \
        -header-filter='.*/torch-xpu-ops/.*' \
        $FIX "$file" 2>&1)

    # Filter to only lines with warnings/errors from our check
    local diags
    diags=$(echo "$output" | grep -E ': (warning|error):')

    if [[ -n "$diags" ]]; then
        local count
        count=$(echo "$diags" | wc -l)
        echo "--- $short ($count diagnostic(s)) ---"
        echo "$output" | grep -v '^$' | grep -v 'warnings generated' | grep -v '^Suppressed' | grep -v '^Use -header'
        echo ""
    fi
}

export -f run_one
export CLANG_TIDY PYTORCH_BUILD FIX CHECK

if [[ $JOBS -le 1 ]]; then
    for i in "${!FILES[@]}"; do
        printf "\r\033[K[%d/%d] Checking..." "$((i+1))" "$TOTAL" >&2
        run_one "${FILES[$i]}"
    done
    printf "\r\033[K" >&2
else
    printf '%s\n' "${FILES[@]}" | \
        xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}
fi

echo "=== Done ==="
