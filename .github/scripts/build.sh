#!/bin/bash
set -xe
export GIT_PAGER=cat

# Init
GITHUB_WORKSPACE=${GITHUB_WORKSPACE:-"/tmp"}
XPU_TORCH=${XPU_TORCH:-"https://github.com/pytorch/pytorch.git@main"}
KEEP_TORCH_XPU_OPS=${KEEP_TORCH_XPU_OPS:-"https://github.com/intel/torch-xpu-ops.git@main"}
XPU_DRIVER=${XPU_DRIVER:-"lts"}

# Set pytorch
rm -rf ${GITHUB_WORKSPACE}/pytorch
git clone ${XPU_TORCH/@*} ${GITHUB_WORKSPACE}/pytorch
cd ${GITHUB_WORKSPACE}/pytorch
git checkout ${XPU_TORCH/*@}
git remote -v && git branch && git show -s

# Set torch-xpu-ops
if [ "${KEEP_TORCH_XPU_OPS,,}" == "true" ];then
    KEEP_TORCH_XPU_OPS="https://github.com/intel/torch-xpu-ops.git@$(cat third_party/xpu.txt)"
fi
if [ "${KEEP_TORCH_XPU_OPS,,}" != "false" ];then
    rm -rf ${GITHUB_WORKSPACE}/torch-xpu-ops
    git clone ${KEEP_TORCH_XPU_OPS/@*} ${GITHUB_WORKSPACE}/torch-xpu-ops
    cd ${GITHUB_WORKSPACE}/torch-xpu-ops
    git checkout ${KEEP_TORCH_XPU_OPS/*@}
fi
cd ${GITHUB_WORKSPACE}/torch-xpu-ops
git remote -v && git branch && git show -s
cd ${GITHUB_WORKSPACE}/pytorch
rm -rf third_party/torch-xpu-ops
cp -r ${GITHUB_WORKSPACE}/torch-xpu-ops third_party/torch-xpu-ops
sed -i "s/checkout --quiet \${TORCH_XPU_OPS_COMMIT}/log -n 1/g" caffe2/CMakeLists.txt

# oneAPI DLE
source third_party/torch-xpu-ops/.github/scripts/env.sh
icpx --version

# Pre Build
cd ${GITHUB_WORKSPACE}/pytorch
python -m pip install requests
python third_party/torch-xpu-ops/.github/scripts/apply_torch_pr.py
git remote -v && git branch && git show -s && git diff
git submodule sync && git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pip install mkl-static mkl-include
# python -m pip install -U cmake==3.31.6
export USE_ONEMKL=1
export USE_XCCL=1
if [ "${XPU_DRIVER}" == "lts" ]; then
    export TORCH_XPU_ARCH_LIST='pvc'
fi
# gcc 11
source /opt/rh/gcc-toolset-11/enable

# Build
WERROR=1 python setup.py bdist_wheel

# Post Build
python -m pip install patchelf
rm -rf ./tmp
bash third_party/torch-xpu-ops/.github/scripts/rpath.sh ${GITHUB_WORKSPACE}/pytorch/dist/torch*.whl
python -m pip install --force-reinstall tmp/torch*.whl

# Verify
cd ${GITHUB_WORKSPACE}
python ${GITHUB_WORKSPACE}/pytorch/torch/utils/collect_env.py
python -c "import torch; print(torch.__config__.show())"
python -c "import torch; print(torch.__config__.parallel_info())"
xpu_is_compiled="$(python -c 'import torch; print(torch.xpu._is_compiled())')"

# Save wheel
if [ "${xpu_is_compiled,,}" == "true" ];then
    cp ${GITHUB_WORKSPACE}/pytorch/tmp/torch*.whl ${GITHUB_WORKSPACE}
else
    echo "Build got failed!"
    exit 1
fi
