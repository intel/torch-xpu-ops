#!/bin/bash
# Usage:
#   ./build.sh --WORKSPACE=<path/to/dir> \
#       --PYTORCH_REPO=<pytorch repo url> --PYTORCH_COMMIT=<pytorch branch or commit> \
#       --TORCH_XPU_OPS_REPO=<torch-xpu-ops repo url> \
#       --TORCH_XPU_OPS_COMMIT=<torch-xpu-ops branch, commit or pinned(use pytorch pinned commit)>
set -xe
export GIT_PAGER=cat

# Init params
WORKSPACE=$(realpath ${WORKSPACE:-"/tmp"})
PYTORCH_REPO=${PYTORCH_REPO:-"https://github.com/pytorch/pytorch.git"}
PYTORCH_COMMIT=${PYTORCH_COMMIT:-"main"}
TORCH_XPU_OPS_REPO=${TORCH_XPU_OPS_REPO:-"https://github.com/intel/torch-xpu-ops.git"}
TORCH_XPU_OPS_COMMIT=${TORCH_XPU_OPS_COMMIT:-"main"}
for var; do
    eval "export $(echo ${var@Q} |sed "s/^'-*//g;s/=/='/")"
done

# Set pytorch
rm -rf ${WORKSPACE}/pytorch
git clone https://github.com/daisyden/pytorch.git ${WORKSPACE}/pytorch
cd ${WORKSPACE}/pytorch
git checkout ${PYTORCH_COMMIT}
git remote -v && git branch && git show -s
git rev-parse HEAD > ${WORKSPACE}/pytorch.commit

# Set torch-xpu-ops
if [ "${TORCH_XPU_OPS_COMMIT,,}" == "pinned" ];then
    TORCH_XPU_OPS_REPO="https://github.com/intel/torch-xpu-ops.git"
    TORCH_XPU_OPS_COMMIT="$(cat ${WORKSPACE}/pytorch/third_party/xpu.txt)"
fi
rm -rf third_party/torch-xpu-ops
if [ "${GITHUB_EVENT_NAME}" == "pull_request" ];then
    cp -r ${WORKSPACE}/torch-xpu-ops third_party/torch-xpu-ops
    cd third_party/torch-xpu-ops
else
    git clone ${TORCH_XPU_OPS_REPO} third_party/torch-xpu-ops
    cd third_party/torch-xpu-ops
    git checkout ${TORCH_XPU_OPS_COMMIT}
fi
git remote -v && git branch && git show -s

# Pre Build
cd ${WORKSPACE}/pytorch
python -m pip install requests
git submodule sync && git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pip install mkl-static mkl-include
export USE_STATIC_MKL=1
if [ "${XPU_ONEAPI_PATH}" == "" ];then
    export PYTORCH_EXTRA_INSTALL_REQUIREMENTS=" \
        intel-cmplr-lib-rt==2025.3.2 | \
        intel-cmplr-lib-ur==2025.3.2 | \
        intel-cmplr-lic-rt==2025.3.2 | \
        intel-sycl-rt==2025.3.2 | \
        oneccl-devel==2021.17.2 | \
        oneccl==2021.17.2 | \
        impi-rt==2021.17.2 | \
        onemkl-sycl-blas==2025.3.1 | \
        onemkl-sycl-dft==2025.3.1 | \
        onemkl-sycl-lapack==2025.3.1 | \
        onemkl-sycl-rng==2025.3.1 | \
        onemkl-sycl-sparse==2025.3.1 | \
        dpcpp-cpp-rt==2025.3.2 | \
        intel-opencl-rt==2025.3.2 | \
        mkl==2025.3.1 | \
        intel-openmp==2025.3.2 | \
        tbb==2022.3.1 | \
        tcmlib==1.4.1 | \
        umf==1.0.3 | \
        intel-pti==0.16.0
    "
fi

# Build
sed -i "s/checkout --quiet \${TORCH_XPU_OPS_COMMIT}/log -n 1/g" caffe2/CMakeLists.txt
git diff
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
