#!/bin/bash
# Usage:
#   ./build.sh --WORKSPACE=<path/to/dir> \
#       --PYTORCH_REPO=<pytorch repo url> --PYTORCH_VERSION=<pytorch branch or commit> \
#       --TORCH_XPU_OPS_REPO=<torch-xpu-ops repo url> \
#       --TORCH_XPU_OPS_VERSION=<torch-xpu-ops branch, commit or pinned(use pytorch pinned commit)>
set -xe
export GIT_PAGER=cat

# Init params
WORKSPACE=$(realpath ${WORKSPACE:-"/tmp"})
PYTORCH_REPO=${PYTORCH_REPO:-"https://github.com/intel-innersource/frameworks.ai.pytorch.private-gpu.git"}
PYTORCH_VERSION=${PYTORCH_VERSION:-"master_next"}
TORCH_XPU_OPS_REPO=${TORCH_XPU_OPS_REPO:-"https://github.com/intel-innersource/frameworks.ai.pytorch.torch-xpu-ops.git"}
TORCH_XPU_OPS_VERSION=${TORCH_XPU_OPS_VERSION:-"main"}
for var; do
    eval "export $(echo ${var@Q} |sed "s/^'-*//g;s/=/='/")"
done

# Set pytorch
rm -rf ${WORKSPACE}/pytorch
git clone ${PYTORCH_REPO} ${WORKSPACE}/pytorch
cd ${WORKSPACE}/pytorch
git checkout ${PYTORCH_VERSION}
git remote -v && git branch && git show -s
git rev-parse HEAD > ${WORKSPACE}/pytorch.commit

# Set torch-xpu-ops
if [ "${TORCH_XPU_OPS_VERSION,,}" == "pinned" ];then
    TORCH_XPU_OPS_REPO="https://github.com/intel/torch-xpu-ops.git"
    TORCH_XPU_OPS_VERSION="$(cat ${WORKSPACE}/pytorch/third_party/xpu.txt)"
fi
if [ "${TORCH_XPU_OPS_VERSION,,}" != "cicd" ];then
    rm -rf ${WORKSPACE}/torch-xpu-ops
    git clone ${TORCH_XPU_OPS_REPO} ${WORKSPACE}/torch-xpu-ops
    cd ${WORKSPACE}/torch-xpu-ops
    git checkout ${TORCH_XPU_OPS_VERSION}
fi
cd ${WORKSPACE}/torch-xpu-ops
git remote -v && git branch && git show -s
cd ${WORKSPACE}/pytorch
rm -rf third_party/torch-xpu-ops
cp -r ${WORKSPACE}/torch-xpu-ops third_party/torch-xpu-ops

# Pre Build
cd ${WORKSPACE}/pytorch
python -m pip install requests
python third_party/torch-xpu-ops/.github/scripts/apply_torch_pr.py
git submodule sync && git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pip install mkl-static mkl-include
export USE_STATIC_MKL=1
export USE_XCCL=0
export USE_ONEMKL_XPU=0

if [ "${PYTORCH_VERSION}" == "CRI_master_next" ];then
    echo -e "\n==================== Build for CRI ===================="
    export USE_KINETO=OFF
    export TORCH_XPU_ARCH_LIST=cri
fi

# Build
sed -i "s/checkout --quiet \${TORCH_XPU_OPS_COMMIT}/log -n 1/g" caffe2/CMakeLists.txt
if [ -n "${ONEDNN_COMMIT}" ];then
    sed -i "s/GIT_TAG prv-gpu/GIT_TAG ${ONEDNN_COMMIT}/g" cmake/Modules/FindMKLDNN.cmake
fi
git diff
python setup.py bdist_wheel

# Post Build
python -m pip install --force-reinstall dist/torch*.whl

# Verify
cd ${WORKSPACE}
python ${WORKSPACE}/pytorch/torch/utils/collect_env.py
python -c "import torch; print(torch.__config__.show())"
python -c "import torch; print(torch.__config__.parallel_info())"
xpu_is_compiled="$(python -c 'import torch; print(torch.xpu._is_compiled())')"

# Save wheel
if [ "${xpu_is_compiled,,}" == "true" ];then
    cp ${WORKSPACE}/pytorch/dist/torch*.whl ${WORKSPACE}
else
    echo "Build got failed!"
    exit 1
fi
