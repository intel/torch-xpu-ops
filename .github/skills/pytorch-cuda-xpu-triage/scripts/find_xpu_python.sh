#!/usr/bin/env bash
# find_xpu_python.sh
# Resolve a torch.xpu-capable Python interpreter.
# Outputs the first qualified interpreter path and exits 0.
# If PYTORCH_XPU_PYTHON is set and capable, it is used directly.
# Otherwise, probes common conda/mamba/miniforge environments and PATH.
set -euo pipefail

check_python() {
  local python_path="$1"

  if [[ ! -x "$python_path" ]]; then
    return 1
  fi

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

if [[ -n "${PYTORCH_XPU_PYTHON:-}" ]] && check_python "$PYTORCH_XPU_PYTHON"; then
  exit 0
fi

declare -a candidates=()

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  candidates+=("$CONDA_PREFIX/bin/python")
fi

while IFS= read -r path; do
  candidates+=("$path")
done < <(
  compgen -G "$HOME/miniforge*/envs/*/bin/python" || true
)

while IFS= read -r path; do
  candidates+=("$path")
done < <(
  compgen -G "$HOME/mambaforge*/envs/*/bin/python" || true
)

while IFS= read -r path; do
  candidates+=("$path")
done < <(
  compgen -G "$HOME/*conda*/envs/*/bin/python" || true
)

if command -v python3 >/dev/null 2>&1; then
  candidates+=("$(command -v python3)")
fi

if command -v python >/dev/null 2>&1; then
  candidates+=("$(command -v python)")
fi

declare -A seen=()
for candidate in "${candidates[@]}"; do
  [[ -n "$candidate" ]] || continue
  if [[ -n "${seen[$candidate]:-}" ]]; then
    continue
  fi
  seen[$candidate]=1
  if check_python "$candidate"; then
    exit 0
  fi
done

printf 'No torch.xpu-capable Python interpreter found. Set PYTORCH_XPU_PYTHON.\n' >&2
exit 1
