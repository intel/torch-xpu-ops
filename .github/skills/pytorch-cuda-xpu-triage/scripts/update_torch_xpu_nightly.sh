#!/usr/bin/env bash
# update_torch_xpu_nightly.sh
# Install or upgrade torch/torchvision/torchaudio from the XPU nightly index.
# Usage: update_torch_xpu_nightly.sh [--dry-run] [python_path]
#   --dry-run    Print the pip command without executing it.
#   python_path  Target Python interpreter (overrides PYTORCH_XPU_PYTHON).
#
# Environment variables:
#   PYTORCH_XPU_PYTHON            Path to target interpreter (used when no positional arg given).
#   PYTORCH_XPU_NIGHTLY_INDEX_URL Override the pip index URL.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
target_python="${PYTORCH_XPU_PYTHON:-}"
nightly_index_url="${PYTORCH_XPU_NIGHTLY_INDEX_URL:-https://download.pytorch.org/whl/nightly/xpu}"
dry_run=0

usage() {
  cat <<'EOF'
Usage: update_torch_xpu_nightly.sh [--dry-run] [python_path]

Updates torch XPU nightly packages for the target interpreter.
Defaults to PYTORCH_XPU_PYTHON; falls back to find_xpu_python.sh auto-detection.
EOF
}

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*"
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
  esac
fi

if [[ $# -gt 0 ]]; then
  target_python="$1"
  shift
fi

if [[ $# -gt 0 ]]; then
  usage >&2
  exit 2
fi

# Fall back to auto-detection when no interpreter was provided.
if [[ -z "$target_python" ]]; then
  target_python="$("$script_dir/find_xpu_python.sh")"
fi

if [[ ! -x "$target_python" ]]; then
  printf 'Target Python is not executable: %s\n' "$target_python" >&2
  exit 1
fi

# Use XDG_RUNTIME_DIR or /tmp for the concurrency lock — portable across
# different checkout locations.
lock_dir="${XDG_RUNTIME_DIR:-/tmp}"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$lock_dir/update-torch-xpu-nightly.lock"
  if ! flock -n 9; then
    log "Another nightly update is already running; exiting."
    exit 0
  fi
fi

packages=(torch torchvision torchaudio)
install_cmd=(
  "$target_python" -m pip install
  --upgrade
  --pre
  --index-url "$nightly_index_url"
  "${packages[@]}"
)

log "Target Python: $target_python"
log "Index URL: $nightly_index_url"
log "Current package versions:"
"$target_python" -m pip list | grep -E '^(pip|torch|torchaudio|torchvision)[[:space:]]' || true

if [[ "$dry_run" -eq 1 ]]; then
  printf 'Dry-run command:'
  printf ' %q' "${install_cmd[@]}"
  printf '\n'
  exit 0
fi

log "Starting nightly package update"
"${install_cmd[@]}"
log "Updated package versions:"
"$target_python" -m pip list | grep -E '^(pip|torch|torchaudio|torchvision)[[:space:]]' || true
