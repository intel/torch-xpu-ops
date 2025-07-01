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
export USE_STATIC_MKL=1
export USE_XCCL=0
if [[ "${BUILD_ONEAPI_NAME}"  == *"2025.0.2.9_offline"* ]];then
python -m pip install mkl-static==2025.0.1 mkl-include==2025.0.1
export PYTORCH_EXTRA_INSTALL_REQUIREMENTS=" \
    intel-cmplr-lib-rt==2025.0.5 |\
    intel-cmplr-lib-ur==2025.0.5 |\
    intel-cmplr-lic-rt==2025.0.5 |\
    intel-sycl-rt==2025.0.5 |\
    impi-devel==2021.14.2 |\
    mkl-devel==2025.0.1 |\
    mkl==2025.0.1 | \
    onemkl-sycl-dft==2025.0.1 |\
    tcmlib==1.2.0 | umf==0.9.1 | intel-pti==0.10.2 | \
    onemkl-sycl-blas==2025.0.1 | \
    onemkl-sycl-dft==2025.0.1 | \
    onemkl-sycl-lapack==2025.0.1 | \
    onemkl-sycl-rng==2025.0.1 | \
    onemkl-sycl-sparse==2025.0.1 \
"
elif [[ "${BUILD_ONEAPI_NAME}"  == *"2025.1.0.581_offline"* ]];then
python -m pip install mkl-static==2025.1.0 mkl-include==2025.1.0
export PYTORCH_EXTRA_INSTALL_REQUIREMENTS=" \
    intel-cmplr-lib-rt==2025.1.0 | \
    intel-cmplr-lib-ur==2025.1.0 | \
    intel-cmplr-lic-rt==2025.1.0 | \
    intel-sycl-rt==2025.1.0 | \
    impi-rt==2021.15.0; platform_system == 'Linux' and platform_machine == 'x86_64' | \
    onemkl-sycl-blas==2025.1.0 | \
    onemkl-sycl-dft==2025.1.0 | \
    onemkl-sycl-lapack==2025.1.0 | \
    onemkl-sycl-rng==2025.1.0 | \
    onemkl-sycl-sparse==2025.1.0 | \
    dpcpp-cpp-rt==2025.1.0 | \
    intel-opencl-rt==2025.1.0 | \
    mkl==2025.1.0 | \
    intel-openmp==2025.1.0 | \
    tbb==2022.1.0 | \
    tcmlib==1.3.0 | \
    umf==0.10.0 | \
    intel-pti==0.11.0
"
elif [[ "${BUILD_ONEAPI_NAME}"  == *"2025.1.3.7_offline"* ]];then
python -m pip install mkl-static==2025.1.1 mkl-include==2025.1.1
export PYTORCH_EXTRA_INSTALL_REQUIREMENTS=" \
    intel-cmplr-lib-rt==2025.1.1 | \
    intel-cmplr-lib-ur==2025.1.1 | \
    intel-cmplr-lic-rt==2025.1.1 | \
    intel-sycl-rt==2025.1.1 | \
    oneccl-devel==2021.15.2; platform_system == 'Linux' and platform_machine == 'x86_64' | \
    oneccl==2021.15.2; platform_system == 'Linux' and platform_machine == 'x86_64' | \
    impi-rt==2021.15.0; platform_system == 'Linux' and platform_machine == 'x86_64' | \
    onemkl-sycl-blas==2025.1.0 | \
    onemkl-sycl-dft==2025.1.0 | \
    onemkl-sycl-lapack==2025.1.0 | \
    onemkl-sycl-rng==2025.1.0 | \
    onemkl-sycl-sparse==2025.1.0 | \
    dpcpp-cpp-rt==2025.1.1 | \
    intel-opencl-rt==2025.1.1 | \
    mkl==2025.1.0 | \
    intel-openmp==2025.1.1 | \
    tbb==2022.1.0 | \
    tcmlib==1.3.0 | \
    umf==0.10.0 | \
    intel-pti==0.12.3
"
else
python -m pip install mkl-static mkl-include
export PYTORCH_EXTRA_INSTALL_REQUIREMENTS=" \
    intel-cmplr-lib-rt | \
    intel-cmplr-lib-ur | \
    intel-cmplr-lic-rt | \
    intel-sycl-rt | \
    oneccl-devel | \
    oneccl | \
    impi-rt| \
    onemkl-sycl-blas | \
    onemkl-sycl-dft | \
    onemkl-sycl-lapack | \
    onemkl-sycl-rng | \
    onemkl-sycl-sparse | \
    dpcpp-cpp-rt | \
    intel-opencl-rt | \
    mkl | \
    intel-openmp | \
    tbb | \
    tcmlib | \
    umf | \
    intel-pti
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
