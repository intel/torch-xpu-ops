#!/bin/bash
# Run clang-tidy on all checkable torch-xpu-ops files with progress.
# Usage: ./run-clang-tidy-all.sh [--jobs N]

PYTORCH_BUILD="${PYTORCH_BUILD:-/home/gta/pytorch/build}"
CLANG_TIDY="${CLANG_TIDY:-clang-tidy-18}"
JOBS=$(nproc)
OUTDIR="/tmp/clang-tidy-results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs|-j) JOBS="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/*.txt

mapfile -t FILES < <(python3 -c "
import json
with open('${PYTORCH_BUILD}/compile_commands.json') as f:
    data = json.load(f)
files = sorted({e['file'] for e in data if 'torch-xpu-ops' in e['file'] and '/sycl/' not in e['file'] and '/sycltla/' not in e['file']})
for f in files: print(f)
")

TOTAL=${#FILES[@]}
echo "=== clang-tidy: $TOTAL files, $JOBS job(s) ==="
echo "Results dir: $OUTDIR"
echo ""

run_one() {
    local idx="$1" file="$2" total="$3"
    local base
    base=$(basename "$file" .cpp)
    local out="$OUTDIR/${base}.txt"
    local short="${file##*/torch-xpu-ops/}"

    "$CLANG_TIDY" -p "$PYTORCH_BUILD" "$file" > "$out" 2>&1

    local warns errs
    warns=$(grep -c ': warning:' "$out" 2>/dev/null || true)
    errs=$(grep -c ': error:' "$out" 2>/dev/null || true)

    if [[ "$errs" -gt 0 || "$warns" -gt 0 ]]; then
        printf "[%3d/%d] FAIL  %-60s  (%d warn, %d err)\n" "$idx" "$total" "$short" "$warns" "$errs"
        # Print unique check names that fired
        grep -oP '\[[-a-zA-Z0-9,.*]+\]$' "$out" | sort -u | while read -r check; do
            printf "         %s\n" "$check"
        done
    else
        printf "[%3d/%d] OK    %s\n" "$idx" "$total" "$short"
    fi
}

export -f run_one
export CLANG_TIDY PYTORCH_BUILD OUTDIR

if [[ $JOBS -le 1 ]]; then
    for i in "${!FILES[@]}"; do
        run_one "$((i+1))" "${FILES[$i]}" "$TOTAL"
    done
else
    for i in "${!FILES[@]}"; do
        echo "$((i+1)) ${FILES[$i]} $TOTAL"
    done | xargs -P "$JOBS" -L1 bash -c 'run_one "$@"' _
fi

echo ""
echo "=== Summary: unique checks that fired ==="
grep -rohP '\[[-a-zA-Z0-9,.*]+\]$' "$OUTDIR"/*.txt 2>/dev/null | sort | uniq -c | sort -rn
echo ""
echo "=== Done: $TOTAL files ==="
echo "Per-file details: $OUTDIR/<name>.txt"
