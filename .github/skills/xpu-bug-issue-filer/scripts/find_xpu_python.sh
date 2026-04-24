#!/usr/bin/env bash
# find_xpu_python.sh
# Resolve a torch.xpu-capable Python interpreter.
# (Included here as a dependency of run_collect_env.sh.)
set -euo pipefail

check_python() {
  local python_path="$1"
  if [[ ! -x "$python_path" ]]; then return 1; fi
  if "$python_path" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if hasattr(torch, "xpu") and torch.xpu.is_available() else 1)
PY
  then
    printf '%s\n' "$python_path"
    return 0
  fi
  return 1
}

if [[ -n "${PYTORCH_XPU_PYTHON:-}" ]] && check_python "$PYTORCH_XPU_PYTHON"; then exit 0; fi

declare -a candidates=()
[[ -n "${CONDA_PREFIX:-}" ]] && candidates+=("$CONDA_PREFIX/bin/python")
while IFS= read -r p; do candidates+=("$p"); done < <(compgen -G "$HOME/miniforge*/envs/*/bin/python" || true)
while IFS= read -r p; do candidates+=("$p"); done < <(compgen -G "$HOME/mambaforge*/envs/*/bin/python" || true)
while IFS= read -r p; do candidates+=("$p"); done < <(compgen -G "$HOME/*conda*/envs/*/bin/python" || true)
command -v python3 >/dev/null 2>&1 && candidates+=("$(command -v python3)")
command -v python  >/dev/null 2>&1 && candidates+=("$(command -v python)")

declare -A seen=()
for candidate in "${candidates[@]}"; do
  [[ -n "$candidate" ]] || continue
  [[ -n "${seen[$candidate]:-}" ]] && continue
  seen[$candidate]=1
  check_python "$candidate" && exit 0
done

printf 'No torch.xpu-capable Python interpreter found. Set PYTORCH_XPU_PYTHON.\n' >&2
exit 1
