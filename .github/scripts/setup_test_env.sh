#!/bin/bash

set -xe
export GIT_PAGER=cat

# Init params
WORKSPACE=$(realpath ${WORKSPACE:-"/tmp"})
CONDA_ENV=${CONDA_ENV:-"xpu-test"}
PYTHON_VERSION=${PYTHON_VERSION:-"3.10"}
PYTORCH_REPO=${PYTORCH_REPO:-"https://github.com/pytorch/pytorch.git"}
PYTORCH_VERSION=${PYTORCH_VERSION:-"main"}
for var; do
    eval "export $(echo ${var@Q} |sed "s/^'-*//g;s/=/='/")"
done

# Python env via conda
conda create python=${PYTHON_VERSION} -y -n ${CONDA_ENV}
conda activate ${CONDA_ENV}
conda info -e
which python && python -V && conda list
python -m pip install requests pandas scipy psutil

# Prepare pytorch
if [ "${PYTORCH_VERSION}" == "release" ];then
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
elif [ "${PYTORCH_VERSION}" == "test" ];then
    python -m pip install torch torchvision torchaudio --pre --index-url https://download.pytorch.org/whl/test/xpu
elif [ "${PYTORCH_VERSION}" == "nightly" ];then
    python -m pip install torch torchvision torchaudio --pre --index-url https://download.pytorch.org/whl/nightly/xpu
else
    python -m pip install ${WORKSPACE}/torch*.whl
fi
TORCH_COMMIT="$(python -c 'import torch; print(torch.version.git_version)')"
rm -rf ./pytorch
git clone ${PYTORCH_REPO} pytorch
cd pytorch
git checkout ${TORCH_COMMIT}
git remote -v && git branch && git show -s

# Prepare torch-xpu-ops
rm -rf third_party/torch-xpu-ops
if [ "${PYTORCH_VERSION}" != "main" ];then
    TORCH_XPU_OPS_COMMIT=$(cat third_party/xpu.txt)
    git clone https://github.com/intel/torch-xpu-ops.git third_party/torch-xpu-ops
    cd third_party/torch-xpu-ops
    git checkout ${TORCH_XPU_OPS_COMMIT}
else
    cp -r ${WORKSPACE}/torch-xpu-ops third_party/torch-xpu-ops
    cd third_party/torch-xpu-ops
fi
git remote -v && git branch && git show -s
cd ../..
if [ "${PYTORCH_VERSION}" == "cicd" ];then
    python third_party/torch-xpu-ops/.github/scripts/apply_torch_pr.py -e https://github.com/pytorch/pytorch/pull/152940
else
    python third_party/torch-xpu-ops/.github/scripts/apply_torch_pr.py
fi

# Install torchvision torchaudio if source build
if [ "${PYTORCH_VERSION}" == "main" ];then
    TORCHVISION_COMMIT=$(cat .github/ci_commit_pins/vision.txt)
    TORCHAUDIO_COMMIT=$(cat .github/ci_commit_pins/audio.txt)
    rm -rf xpu-vision && git git clone https://github.com/pytorch/vision.git xpu-vision
    cd xpu-vision && python setup.py install && cd ..
    rm -rf xpu-audio && git git clone https://github.com/pytorch/audio.git xpu-audio
    cd xpu-audio && python setup.py install && cd ..
fi
python -m pip install -r .ci/docker/requirements-ci.txt

# Collect env infos
cd ..
python -c "import torch; print(torch.__config__.show())"
python -c "import torch; print(torch.__config__.parallel_info())"
python -c "import torch; print(torch.__config__.torch.xpu.device_count())"
python -c "import triton; print(triton.__version__)"
python pytorch/torch/utils/collect_env.py

# Clean cache
rm -rf /tmp/torchinductor_*
rm -rf ~/.triton/
rm -rf /tmp/tmp*
