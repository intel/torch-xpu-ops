#!/bin/bash
set -xe
export GIT_PAGER=cat

# Init
GITHUB_WORKSPACE=${GITHUB_WORKSPACE:-"/tmp"}
XPU_TORCH=${XPU_TORCH:-"https://github.com/pytorch/pytorch.git@main"}
KEEP_TORCH_XPU_OPS=${KEEP_TORCH_XPU_OPS:-"https://github.com/intel/torch-xpu-ops.git@main"}
XPU_DRIVER=${XPU_DRIVER:-"lts"}
XPU_CONDA_ENV=${XPU_CONDA_ENV:-"xpu-build"}
XPU_PYTHON=${XPU_PYTHON:-"3.10"}

# Conda env
. "$(dirname ${CONDA_EXE})/../etc/profile.d/conda.sh"
conda clean -ay
conda create python=${XPU_PYTHON} -y -n ${XPU_CONDA_ENV}
conda activate ${XPU_CONDA_ENV}
conda info -e
gcc -v
python -V
python -m pip config set global.progress-bar off
python -m pip install -U pip wheel setuptools

# Set pytorch
rm -rf ${GITHUB_WORKSPACE}/xpu-pytorch
git clone ${XPU_TORCH/@*} ${GITHUB_WORKSPACE}/xpu-pytorch
cd ${GITHUB_WORKSPACE}/xpu-pytorch
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
cd ${GITHUB_WORKSPACE}/xpu-pytorch
rm -rf third_party/torch-xpu-ops
cp -r ${GITHUB_WORKSPACE}/torch-xpu-ops third_party/torch-xpu-ops
sed -i "s/checkout --quiet \${TORCH_XPU_OPS_COMMIT}/log -n 1/g" caffe2/CMakeLists.txt

# oneAPI DLE
source ${GITHUB_WORKSPACE}/torch-xpu-ops/.github/scripts/env.sh
sycl-ls
icpx --version

# Pre Build
cd ${GITHUB_WORKSPACE}/xpu-pytorch
python -m pip install requests
python ${GITHUB_WORKSPACE}/torch-xpu-ops/.github/scripts/apply_torch_pr.py
git remote -v && git branch && git show -s && git diff
git submodule sync && git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pip install mkl-static mkl-include
python -m pip install -U cmake==3.31.6
export USE_ONEMKL=1
export USE_XCCL=1
if [ "${XPU_DRIVER}" == "lts" ]; then
    export TORCH_XPU_ARCH_LIST='pvc'
fi

# Build
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
WERROR=1 python setup.py bdist_wheel

# Post Build
conda install patchelf zip -y
bash ${GITHUB_WORKSPACE}/torch-xpu-ops/.github/scripts/rpath.sh ${GITHUB_WORKSPACE}/xpu-pytorch/dist/torch*.whl
python -m pip install --force-reinstall tmp/torch*.whl

# Verify
cd ${GITHUB_WORKSPACE}
python ${GITHUB_WORKSPACE}/xpu-pytorch/torch/utils/collect_env.py
conda list
python -c "import torch; print(torch.__config__.show())"
python -c "import torch; print(torch.__config__.parallel_info())"
xpu_is_compiled="$(python -c 'import torch; print(torch.xpu._is_compiled())')"

# Save wheel
if [ "${xpu_is_compiled,,}" == "true" ];then
    cp ${GITHUB_WORKSPACE}/xpu-pytorch/tmp/torch*.whl ${GITHUB_WORKSPACE}
else
    echo "Build got failed!"
    exit 1
fi
