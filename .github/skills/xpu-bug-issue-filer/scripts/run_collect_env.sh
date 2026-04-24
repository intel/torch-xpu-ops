#!/usr/bin/env bash
# run_collect_env.sh
# Run torch.utils.collect_env with the XPU-capable interpreter.
# Output should be included in the Versions section of a filed issue.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_path="$($script_dir/find_xpu_python.sh)"

exec "$python_path" -W ignore::RuntimeWarning -m torch.utils.collect_env "$@"
