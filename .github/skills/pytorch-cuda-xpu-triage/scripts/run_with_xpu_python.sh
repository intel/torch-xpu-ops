#!/usr/bin/env bash
# run_with_xpu_python.sh
# Execute a Python script with the XPU-capable interpreter.
# Usage: run_with_xpu_python.sh <script> [args...]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  printf 'Usage: %s <script> [args...]\n' "$0" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_path="$($script_dir/find_xpu_python.sh)"

exec "$python_path" "$@"
