#!/bin/bash
# Build PyTorch XPU wheel. Assumes PyTorch source is already prepared
# (via prepare_pytorch.py) at ${WORKSPACE}/pytorch.
#
# Usage:
#   ./build.sh [--WORKSPACE=<path>] [--USE_DPCLANG=yes]
set -xe

USE_DPCLANG=${USE_DPCLANG:-"no"}
for var; do
    # shellcheck disable=SC2086
    eval "export $(echo ${var@Q} |sed "s/^'-*//g;s/=/='/")"
done

WORKSPACE=$(realpath "${WORKSPACE:-/tmp}")
cd "${WORKSPACE}/pytorch"

# Build using PyTorch's upstream build pipeline
export GPU_ARCH_TYPE=xpu
export DESIRED_CUDA=xpu

if [ "${USE_DPCLANG}" == "yes" ]; then
    # dpclang: open-source SYCL compiler, skip upstream env setup (no oneAPI)
    export XPU_SYCL_COMPILER=dpclang
    export USE_KINETO=0
    export USE_ONEMKL_XPU=0
    export USE_STATIC_MKL=1
    export TH_BINARY_BUILD=1
    export USE_CUDA=0
    export INSTALL_TEST=0
    python -m pip install -r requirements.txt
    python -m pip install mkl-static==2026.0.0 mkl-include==2026.0.0
    python -m pip install build auditwheel==6.4.2
    WERROR=1 python -m build --wheel --no-isolation --outdir dist/
else
    # Normal XPU: use PyTorch's upstream build scripts
    if [ "${XPU_ONEAPI_PATH}" == "" ]; then
        REQS=$(python -c "
import sys; sys.path.insert(0, '.github/scripts')
from generate_binary_build_matrix import PYTORCH_EXTRA_INSTALL_REQUIREMENTS
print(PYTORCH_EXTRA_INSTALL_REQUIREMENTS['xpu'])
")
        export PYTORCH_EXTRA_INSTALL_REQUIREMENTS="${REQS}"
    fi

    # Step 1: Environment setup (sources oneAPI, sets XPU build flags)
    ENV_FILE=$(mktemp)
    trap 'rm -f "$ENV_FILE"' EXIT
    python .ci/manywheel/build_env_setup.py --env-out "$ENV_FILE"
    # shellcheck source=/dev/null
    source "$ENV_FILE"

    # Step 2: Install build dependencies
    python .ci/manywheel/build_install_deps.py "${WORKSPACE}/pytorch"

    # Step 3: Build wheel
    export WERROR=1
    RAW_WHEEL_DIR=$(mktemp -d)
    python .ci/manywheel/build_wheel.py "$RAW_WHEEL_DIR"

    # Step 4: Repair wheel (RPATH patching + platform retagging)
    mkdir -p dist/
    python .ci/manywheel/repair_wheel.py "$RAW_WHEEL_DIR" dist/
fi

# Post Build: Install and verify
python -m pip install --force-reinstall dist/torch*.whl

cd "${WORKSPACE}"
python "${WORKSPACE}/pytorch/torch/utils/collect_env.py"
python -c "import torch; print(torch.__config__.show())"
python -c "import torch; print(torch.__config__.parallel_info())"
xpu_is_compiled="$(python -c 'import torch; print(torch.xpu._is_compiled())')"

# Save wheel
if [ "${xpu_is_compiled,,}" == "true" ];then
    rm -rf "${WORKSPACE}"/torch-*.whl
    cp "${WORKSPACE}/pytorch/dist"/torch-*.whl "${WORKSPACE}"
else
    echo "Build got failed!"
    exit 1
fi
