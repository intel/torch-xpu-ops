#!/bin/bash
set -euo pipefail

# Post-op the Linux wheel to change the .so rpath to work with XPU runtime PyPI packages
# Usage: rpath.sh <wheel-file> [output-dir]

# --- Configuration ---
PATCHELF_BIN="${PATCHELF_BIN:-patchelf}"
export FORCE_RPATH="${FORCE_RPATH:---force-rpath}"  # set to empty to disable

# Rpath definitions
XPU_RPATHS=(
    '$ORIGIN/../../../..'
)
XPU_RPATHS_JOINED=$(IFS=: ; echo "${XPU_RPATHS[*]}")

export C_SO_RPATH="${XPU_RPATHS_JOINED}:\$ORIGIN:\$ORIGIN/lib"
export LIB_SO_RPATH="${XPU_RPATHS_JOINED}:\$ORIGIN"

# --- Helper functions ---
usage() {
    echo "Usage: $0 <wheel-file> [output-dir]"
    exit 1
}

check_deps() {
    for cmd in "$PATCHELF_BIN" openssl unzip zip realpath; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            echo "Error: Required command '$cmd' not found." >&2
            exit 1
        fi
    done
}

make_wheel_record() {
    local file="$1"
    if [[ "$file" == *RECORD ]]; then
        echo "$file,,"
    else
        local hash size
        hash=$(openssl dgst -sha256 -binary "$file" | openssl base64 | tr -d '\n' | sed -e 's/+/-/g' -e 's/\//_/g' -e 's/=*$//')
        size=$(stat -c %s "$file" 2>/dev/null || stat -f %z "$file" 2>/dev/null)
        echo "$file,sha256=$hash,$size"
    fi
}

# --- Main ---
[[ $# -ge 1 ]] || usage
input_wheel="$(realpath "$1")"
[[ -f "$input_wheel" ]] || { echo "Error: File not found: $input_wheel" >&2; exit 1; }

output_dir="${2:-.}"
# Create output directory if it doesn't exist
mkdir -p "$output_dir"
output_dir="$(realpath "$output_dir")"

check_deps

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
cd "$workdir"

echo "Processing $input_wheel ..."
cp "$input_wheel" .

wheel_name="$(basename "$input_wheel")"
unzip -q "$wheel_name"
rm -f "$wheel_name"

# Locate .dist-info
dist_info_dir=$(find . -maxdepth 2 -type d -name "*.dist-info" -print -quit)
if [[ -z "$dist_info_dir" ]]; then
    echo "Error: No .dist-info directory found in wheel." >&2
    exit 1
fi

echo "Setting RPATHs on .so files..."
find . -type f -name "*.so*" -print0 | while IFS= read -r -d '' sofile; do
    sofile="${sofile#./}"
    if [[ "$sofile" == lib/* ]]; then
        rpath="$LIB_SO_RPATH"
        echo "  [lib] $sofile -> $rpath"
    else
        rpath="$C_SO_RPATH"
        echo "  [top] $sofile -> $rpath"
    fi
    "$PATCHELF_BIN" --set-rpath "$rpath" ${FORCE_RPATH:+"$FORCE_RPATH"} "$sofile"
done

# Regenerate RECORD
record_file="$dist_info_dir/RECORD"
echo "Regenerating $record_file ..."
: > "$record_file"

find . -type f ! -name RECORD -print0 | sort -z | while IFS= read -r -d '' fname; do
    fname="${fname#./}"
    make_wheel_record "$fname" >> "$record_file"
done
echo "$record_file,," >> "$record_file"

# Repack wheel
echo "Repacking wheel ..."
zip -rq "$wheel_name" . -x "*.DS_Store"

# Move to output directory
output_wheel="$output_dir/$wheel_name"
if [[ -e "$output_wheel" ]]; then
    echo "Warning: $output_wheel already exists, overwriting." >&2
fi
mv "$wheel_name" "$output_wheel"
echo "Done. Modified wheel saved to: $output_wheel"
