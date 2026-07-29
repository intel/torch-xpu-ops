#!/usr/bin/env bash
# Usage: bash setup_env.sh [build_type] [env_name] [pytorch_folder] [torch_version]
#   build_type:     "nightly" (default) - install pytorch XPU wheels
#                   "source"            - build pytorch from local source (TBD,
#                                         not supported yet)
#   env_name:       conda env name to create (default: basename of the repo root).
#                   Pass "pytorch_opencode_env" to match the validation skills'
#                   canonical local XPU env.
#   pytorch_folder: local pytorch/pytorch checkout to prepare
#                   (default: $HOME/daisy_pytorch). If it does not exist it is
#                   cloned from pytorch/pytorch main. After the torch wheel is
#                   installed the checkout is synced to the wheel's source
#                   commit, and intel/torch-xpu-ops is checked out under
#                   <pytorch_folder>/third_party at the commit pinned in
#                   third_party/xpu.txt.
#   torch_version:  optional torch wheel to install. Empty (default) installs
#                   the latest nightly XPU wheel from the nightly index. When a
#                   version is given, the index is chosen by its shape:
#                   a release version (e.g. "2.13.0+xpu") is pulled from the
#                   test index (https://download.pytorch.org/whl/test/xpu);
#                   a date/commit nightly version (e.g. "2.9.0.dev20250601+xpu")
#                   is pulled from the nightly index
#                   (https://download.pytorch.org/whl/nightly/xpu).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Walk up from scripts/ -> inductor repo root (6 levels)
CUR_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../" && pwd)"
ENV_NAME="${2:-$(basename "${CUR_ROOT}")}"
PYTORCH_FOLDER="${3:-$HOME/daisy_pytorch}"
TORCH_VERSION="${4:-}"
CONDA_BASE="$HOME/miniforge3"

export http_proxy="http://proxy-dmz.intel.com:912"
export https_proxy="http://proxy-dmz.intel.com:912"
export HTTP_PROXY="http://proxy-dmz.intel.com:912"
export HTTPS_PROXY="http://proxy-dmz.intel.com:912"
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"

# Allow override via first argument: "nightly" (default) or "source"
BUILD_TYPE="${1:-nightly}"
if [ "${BUILD_TYPE}" = "source" ]; then
  echo "[error] build_type 'source' is not supported yet (TBD). Use 'nightly'."
  exit 1
elif [ "${BUILD_TYPE}" != "nightly" ]; then
  echo "[error] unknown build_type '${BUILD_TYPE}'. Use 'nightly'."
  exit 1
fi

# ── Create / recreate conda environment ──────────────────────────────
echo "=== Creating conda environment: ${ENV_NAME} ==="
"${CONDA_BASE}/bin/conda" remove --name "${ENV_NAME}" --all --yes 2>/dev/null || true
"${CONDA_BASE}/bin/conda" create --name "${ENV_NAME}" python=3.10 -y

# shellcheck disable=SC1091
source "${CONDA_BASE}/bin/activate" "${ENV_NAME}"

# --- Install tooling + torch wheel ---
echo "=== Installing base tooling ==="
pip install --root-user-action=ignore cmake ninja pybind11
pip install --root-user-action=ignore openpyxl
pip install --root-user-action=ignore pytest-timeout
pip install --root-user-action=ignore junitparser
"${CONDA_BASE}/bin/conda" install --name "${ENV_NAME}" -c conda-forge gh -y 2>/dev/null || \
  pip install --root-user-action=ignore gh 2>/dev/null || \
  echo "[warn] gh install skipped; install GitHub CLI manually if needed"

pip uninstall torch -y 2>/dev/null || true
NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly/xpu"
TEST_INDEX="https://download.pytorch.org/whl/test/xpu"
if [ -z "${TORCH_VERSION}" ]; then
  # No version pinned: install the latest nightly.
  TORCH_INDEX="${NIGHTLY_INDEX}"
  echo "=== Installing latest nightly torch from ${TORCH_INDEX} ==="
  pip install --root-user-action=ignore --pre torch --index-url "${TORCH_INDEX}"
elif [[ "${TORCH_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([+][^.]+)?$ ]]; then
  # Release-style version (e.g. 2.13.0+xpu): use the test channel.
  TORCH_INDEX="${TEST_INDEX}"
  echo "=== Installing release torch==${TORCH_VERSION} from ${TORCH_INDEX} ==="
  pip install --root-user-action=ignore "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
else
  # Date/commit (nightly-style) version: use the nightly channel.
  TORCH_INDEX="${NIGHTLY_INDEX}"
  echo "=== Installing nightly torch==${TORCH_VERSION} from ${TORCH_INDEX} ==="
  pip install --root-user-action=ignore --pre "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
fi

# Install matching triton from the same index the torch wheel came from.
TMPDIR=$(mktemp -d)
pip download --no-deps --index-url "${TORCH_INDEX}" \
  --pre pytorch-triton-xpu --dest "${TMPDIR}" 2>/dev/null
pip install --root-user-action=ignore "${TMPDIR}"/pytorch_triton_xpu-*.whl 2>/dev/null || true
rm -rf "${TMPDIR}"

# --- Prepare local pytorch checkout and pin torch-xpu-ops ---
# Clone pytorch/pytorch if the target folder is absent, then sync the checkout
# to the installed wheel's source commit and pin third_party/torch-xpu-ops.
# This happens before installing pytorch requirements so the deps match the
# exact source tree we test against.
echo "=== Preparing pytorch checkout: ${PYTORCH_FOLDER} ==="
if [ ! -d "${PYTORCH_FOLDER}/.git" ]; then
  echo "=== Cloning pytorch/pytorch main into ${PYTORCH_FOLDER} ==="
  git clone https://github.com/pytorch/pytorch.git "${PYTORCH_FOLDER}"
fi

# Sync the checkout to the exact commit the installed torch wheel was built from.
WHEEL_COMMIT="$(python -c 'import torch; print(torch.version.git_version or "")' 2>/dev/null || true)"
if [ -n "${WHEEL_COMMIT}" ]; then
  echo "=== Syncing ${PYTORCH_FOLDER} to torch wheel commit ${WHEEL_COMMIT} ==="
  if git -C "${PYTORCH_FOLDER}" checkout --detach "${WHEEL_COMMIT}" 2>/dev/null; then
    echo "=== pytorch synced to ${WHEEL_COMMIT} ==="
  elif git -C "${PYTORCH_FOLDER}" fetch origin "${WHEEL_COMMIT}" 2>/dev/null &&
       git -C "${PYTORCH_FOLDER}" checkout --detach FETCH_HEAD 2>/dev/null; then
    echo "=== pytorch synced to ${WHEEL_COMMIT} ==="
  else
    echo "[warn] could not check out wheel commit ${WHEEL_COMMIT} (nightly commit may be gone); leaving pytorch on its current branch"
  fi
else
  echo "[warn] installed torch exposes no git_version; skipping pytorch commit sync"
fi

# Check out intel/torch-xpu-ops under third_party at the pinned commit.
XPU_PIN_FILE="${PYTORCH_FOLDER}/third_party/xpu.txt"
XPU_OPS_DIR="${PYTORCH_FOLDER}/third_party/torch-xpu-ops"
if [ -f "${XPU_PIN_FILE}" ]; then
  XPU_COMMIT="$(tr -d '[:space:]' < "${XPU_PIN_FILE}")"
  echo "=== Pinning torch-xpu-ops to ${XPU_COMMIT} (from third_party/xpu.txt) ==="
  if [ ! -d "${XPU_OPS_DIR}/.git" ]; then
    git clone https://github.com/intel/torch-xpu-ops.git "${XPU_OPS_DIR}"
  fi
  if git -C "${XPU_OPS_DIR}" checkout --detach "${XPU_COMMIT}" 2>/dev/null; then
    echo "=== torch-xpu-ops checked out at ${XPU_COMMIT} ==="
  elif git -C "${XPU_OPS_DIR}" fetch origin "${XPU_COMMIT}" 2>/dev/null &&
       git -C "${XPU_OPS_DIR}" checkout --detach FETCH_HEAD 2>/dev/null; then
    echo "=== torch-xpu-ops checked out at ${XPU_COMMIT} ==="
  else
    echo "[warn] could not check out torch-xpu-ops commit ${XPU_COMMIT}; leaving as-is"
  fi
else
  echo "[warn] ${XPU_PIN_FILE} not found; skipping torch-xpu-ops pin"
fi

# --- Install pytorch CI + build dependencies (from the synced checkout) ---
echo "=== Installing pytorch dependencies from ${PYTORCH_FOLDER} ==="
if [ -f "${PYTORCH_FOLDER}/.ci/docker/requirements-ci.txt" ]; then
  pip install --root-user-action=ignore -r "${PYTORCH_FOLDER}/.ci/docker/requirements-ci.txt" 2>/dev/null || \
    echo "[warn] requirements-ci.txt install had issues, continuing..."
fi
if [ -f "${PYTORCH_FOLDER}/requirements.txt" ]; then
  pip install --root-user-action=ignore -r "${PYTORCH_FOLDER}/requirements.txt"
fi

# ── Log versions ──────────────────────────────────────────────────────
echo "=== Environment created: ${ENV_NAME} ==="
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('XPU available:', getattr(torch, 'xpu', None) and torch.xpu.is_available())" 2>/dev/null || \
  python -c "import torch; print('PyTorch:', torch.__version__)"
pip list 2>/dev/null | grep -i torch
