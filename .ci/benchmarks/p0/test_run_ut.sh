#!/bin/bash
# Run with: bash .ci/benchmarks/p0/test_run_ut.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/ut_skips.sh"
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT
export PYTORCH_ROOT_DIR="$tmp/pytorch root"
export XML_OUTPUT_DIR="$tmp/xml"
export CALLS="$tmp/calls"
mkdir -p "$PYTORCH_ROOT_DIR/third_party/torch-xpu-ops" "$XML_OUTPUT_DIR"

pytest() { printf '%s\n' "$@" >> "$CALLS"; }
python() { :; }
xpu-smi() { :; }
export -f pytest python xpu-smi

for family in bmg pvc unknown unset; do
    if [[ "$family" == unset ]]; then
        unset XPU_DEVICE_FAMILY
    else
        export XPU_DEVICE_FAMILY="$family"
    fi
    : > "$CALLS"
    (cd "$tmp" && bash "$SCRIPT_DIR/run_ut.sh") > "$tmp/runner.log" 2>&1
    expected=()
    for node in "${PYTEST_DESELECTS[@]}"; do
        prefix=""
        if [[ "$node" == test/xpu/* || "$node" == test/profiling/* ]]; then
            prefix="third_party/torch-xpu-ops/"
        fi
        expected+=("--deselect=${prefix}${node}")
    done
    printf '%s\n' "${expected[@]}" | sort > "$tmp/expected"
    grep '^--deselect=' "$CALLS" | sort > "$tmp/actual"
    diff -u "$tmp/expected" "$tmp/actual"
    file_count="$(grep -c '^test/.*\.py$' "$CALLS")"
    [[ "$(grep -Fxc -- "--rootdir=$PYTORCH_ROOT_DIR" "$CALLS")" == "$file_count" ]]
done

mkdir "$tmp/missing"
cp "$SCRIPT_DIR/run_ut.sh" "$tmp/missing/run_ut.sh"
: > "$CALLS"
if bash "$tmp/missing/run_ut.sh" inductor > "$tmp/missing.log" 2>&1; then
    echo "ERROR: Missing skip list did not fail." >&2
    exit 1
fi
grep -q 'ERROR: Could not load P0 skip list' "$tmp/missing.log"
[[ ! -s "$CALLS" ]]

# Generate failing stand-ins for all 12 added cases, plus one passing control
# per class. Real pytest must collect the controls and deselect every failure.
unset -f pytest python xpu-smi
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTEST_ADDOPTS="-q -p no:cacheprovider"
declare -A class_methods=()
for node in "${PYTEST_DESELECTS[@]: -12}"; do
    file="${node%%::*}"
    rest="${node#*::}"
    class="${rest%%::*}"
    method="${rest#*::}"
    key="$file::$class"
    class_methods["$key"]+=$'    def '"$method"$'(self):\n        assert False, "Skip did not match"\n'
done
declare -A fixture_files=()
for key in "${!class_methods[@]}"; do
    file="${key%%::*}"
    class="${key#*::}"
    cwd="$PYTORCH_ROOT_DIR"
    if [[ "$file" == test/xpu/* ]]; then
        cwd+="/third_party/torch-xpu-ops"
    fi
    mkdir -p "$cwd/$(dirname "$file")"
    printf 'class %s:\n%s    def test_control(self):\n        pass\n\n' \
        "$class" "${class_methods[$key]}" >> "$cwd/$file"
    fixture_files["$file"]="$cwd"
done
source <(sed -n '/^run_pytest() {/,/^}/p' "$SCRIPT_DIR/run_ut.sh")
XML_PREFIX="fixture_"
for file in "${!fixture_files[@]}"; do
    (cd "${fixture_files[$file]}" && run_pytest "$file") >> "$tmp/real.log" 2>&1
done
passed="$(grep -ho 'tests="[0-9]*"' "$XML_OUTPUT_DIR"/fixture_*.xml | awk -F '"' '{n += $2} END {print n}')"
[[ "$passed" == "${#class_methods[@]}" ]]
deselected="$(grep -o '[0-9]* deselected' "$tmp/real.log" | awk '{n += $1} END {print n}')"
[[ "$deselected" == 12 ]]
echo "PASS: all skip arguments preserved across four platform settings; missing-list failure; 12 real pytest deselections with $passed passing controls."
