#!/bin/bash
set -xe
export GIT_PAGER=cat

# Init params
WORKSPACE=$(realpath ${WORKSPACE:-"/tmp"})
PYTORCH_REPO=${PYTORCH_REPO:-"https://github.com/pytorch/pytorch.git"}
PYTORCH_VERSION=${PYTORCH_VERSION:-"main"}
TORCH_XPU_OPS_REPO=${TORCH_XPU_OPS_REPO:-"https://github.com/intel/torch-xpu-ops.git"}
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
if [ "${TORCH_XPU_OPS_VERSION,,}" == "pytorch-pinned" ];then
    TORCH_XPU_OPS_REPO="https://github.com/intel/torch-xpu-ops.git"
    TORCH_XPU_OPS_VERSION="$(cat ${WORKSPACE}/pytorch/third_party/xpu.txt)"
fi
if [ "${TORCH_XPU_OPS_VERSION,,}" != "ci-nightly" ];then
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
sed -i "s/checkout --quiet \${TORCH_XPU_OPS_COMMIT}/log -n 1/g" caffe2/CMakeLists.txt

# Pre Build
cd ${WORKSPACE}/pytorch
python -m pip install requests
python third_party/torch-xpu-ops/.github/scripts/apply_torch_pr.py
git diff && git submodule sync && git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pip install mkl-static mkl-include
# python -m pip install -U cmake==3.31.6
export USE_ONEMKL=1
export USE_XCCL=1

# Build
WERROR=1 python setup.py bdist_wheel

# Post Build
python -m pip install patchelf
rm -rf ./tmp
bash third_party/torch-xpu-ops/.github/scripts/rpath.sh ${WORKSPACE}/pytorch/dist/torch*.whl
python -m pip install --force-reinstall tmp/torch*.whl

# Verify
cd ${WORKSPACE}
python ${WORKSPACE}/pytorch/torch/utils/collect_env.py
python -c "import torch; print(torch.__config__.show())"
python -c "import torch; print(torch.__config__.parallel_info())"
xpu_is_compiled="$(python -c 'import torch; print(torch.xpu._is_compiled())')"

# Save wheel
if [ "${xpu_is_compiled,,}" == "true" ];then
    cp ${WORKSPACE}/pytorch/tmp/torch*.whl ${WORKSPACE}
else
    echo "Build got failed!"
    exit 1
fi
