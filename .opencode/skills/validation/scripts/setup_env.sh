#!/usr/bin/env bash
# Usage: bash setup_env.sh [build_type]
#   build_type: "source" (default) - build pytorch from local source
#               "nightly"          - install pytorch nightly wheels

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Walk up from scripts/ -> inductor repo root (6 levels)
INDUCTOR_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../" && pwd)"
ENV_NAME="$(basename "${INDUCTOR_ROOT}")"
WORKDIR="${INDUCTOR_ROOT}"
CONDA_BASE="$HOME/miniforge3"

export http_proxy="http://proxy-dmz.intel.com:912"
export https_proxy="http://proxy-dmz.intel.com:912"
export HTTP_PROXY="http://proxy-dmz.intel.com:912"
export HTTPS_PROXY="http://proxy-dmz.intel.com:912"
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"

# Allow override via first argument: "source" or "nightly"
BUILD_TYPE="${1:-source}"

# ── Create / recreate conda environment ──────────────────────────────
echo "=== Creating conda environment: ${ENV_NAME} ==="
"${CONDA_BASE}/bin/conda" remove --name "${ENV_NAME}" --all --yes 2>/dev/null || true
"${CONDA_BASE}/bin/conda" create --name "${ENV_NAME}" python=3.10 -y

# shellcheck disable=SC1091
source "${CONDA_BASE}/bin/activate" "${ENV_NAME}"

# ── Install build & CI dependencies ──────────────────────────────────
echo "=== Installing dependencies ==="
pip install --root-user-action=ignore -r "${WORKDIR}/.ci/docker/requirements-ci.txt" 2>/dev/null || \
  echo "[warn] requirements-ci.txt install had issues, continuing..."

pip install --root-user-action=ignore cmake ninja pybind11
pip install --root-user-action=ignore -r "${WORKDIR}/requirements.txt"
pip install --root-user-action=ignore openpyxl
pip install --root-user-action=ignore pytest-timeout

pip uninstall torch -y 2>/dev/null || true
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/xpu

# Install matching triton
TMPDIR=$(mktemp -d)
pip download --no-deps --index-url https://download.pytorch.org/whl/nightly/xpu \
  --pre pytorch-triton-xpu --dest "${TMPDIR}" 2>/dev/null
pip install --root-user-action=ignore "${TMPDIR}"/pytorch_triton_xpu-*.whl 2>/dev/null || true
rm -rf "${TMPDIR}"

# ── Log versions ──────────────────────────────────────────────────────
echo "=== Environment created: ${ENV_NAME} ==="
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('XPU available:', getattr(torch, 'xpu', None) and torch.xpu.is_available())" 2>/dev/null || \
  python -c "import torch; print('PyTorch:', torch.__version__)"
pip list 2>/dev/null | grep -i torch
