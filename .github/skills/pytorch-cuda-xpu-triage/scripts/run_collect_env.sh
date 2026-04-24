#!/usr/bin/env bash
# run_collect_env.sh
# Run torch.utils.collect_env with the XPU-capable interpreter.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_path="$($script_dir/find_xpu_python.sh)"

exec "$python_path" -W ignore::RuntimeWarning -m torch.utils.collect_env "$@"
