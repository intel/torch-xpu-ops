#!/bin/bash
set -xe

input_type="$1"
action_type="$2"

# Activate oneAPI DLE
if [[ "${input_type}" == *"oneapi"* ]];then
    ONEAPI_ROOT="/opt/intel/oneapi"
    source "${ONEAPI_ROOT}/compiler/latest/env/vars.sh"
    source "${ONEAPI_ROOT}/umf/latest/env/vars.sh"
    source "${ONEAPI_ROOT}/pti/latest/env/vars.sh"
    source "${ONEAPI_ROOT}/ccl/latest/env/vars.sh"
    source "${ONEAPI_ROOT}/mpi/latest/env/vars.sh"
    source "${ONEAPI_ROOT}/mkl/latest/env/vars.sh"
    sycl-ls
    icpx --version
fi

# Conda Env
if [[ "${input_type}" == *"conda"* ]];then
    conda_config_file="/opt/conda/etc/profile.d/conda.sh"
    if $(which conda > /dev/null 2>&1) ;then
        conda_config_file="$(realpath "$(dirname ${CONDA_EXE})/../etc/profile.d/conda.sh")"
    fi
    if [ ! -e "${conda_config_file}" ];then
        rm -rf /opt/conda "${HOME}/.conda" "${HOME}/.condarc"
        curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
        bash "Miniforge3-$(uname)-$(uname -m).sh" -b -f -p "/opt/conda"
    fi
    . "${conda_config_file}"
    if [[ "${action_type}" == *"create"* ]];then
        conda remove --all -y -n "xpu_op_${ZE_AFFINITY_MASK}" || rm -rf $(dirname ${CONDA_EXE})/../envs/xpu_op_${ZE_AFFINITY_MASK}
        conda create python=${{ inputs.python }} cmake ninja -n "xpu_op_${ZE_AFFINITY_MASK}" -y
    fi
    conda activate "xpu_op_${ZE_AFFINITY_MASK}"
    conda info -e
fi

# Prepare PyTorch
if [[ "${input_type}" == *"pytorch"* ]];then
    if [[ "${{ inputs.pytorch }}" != *"nightly_wheel"* ]]; then
      pip install --force-reinstall ${GITHUB_WORKSPACE}/torch*.whl
    else
      pip install torch torchvision torchaudio --pre --index-url https://download.pytorch.org/whl/nightly/xpu
    fi
    TORCH_COMMIT_ID=$(python -c 'import torch; print(torch.version.git_version)')
    rm -rf ${GITHUB_WORKSPACE}/pytorch
    git clone https://github.com/pytorch/pytorch ${GITHUB_WORKSPACE}/pytorch
    cd ${GITHUB_WORKSPACE}/pytorch
    git checkout ${TORCH_COMMIT_ID}
    pip install requests
    python third_party/torch-xpu-ops/.github/scripts/apply_torch_pr.py
    git status && git show -s
    pip install -r .ci/docker/requirements-ci.txt
    # Torch-xpu-ops
    rm -rf third_party/torch-xpu-ops
    if [[ "${{ inputs.pytorch }}" != *"nightly_wheel"* ]]; then
      cp -r ${GITHUB_WORKSPACE}/torch-xpu-ops third_party/torch-xpu-ops
    else
      TORCH_XPU_OPS_COMMIT=$(<third_party/xpu.txt)
      git clone https://github.com/intel/torch-xpu-ops.git third_party/torch-xpu-ops
      cd third_party/torch-xpu-ops
      git checkout ${TORCH_XPU_OPS_COMMIT}
      git status && git show -s
      cd ../../
    fi
    # Triton
    TRITON_REPO="https://github.com/intel/intel-xpu-backend-for-triton"
    if [ -z "${{ inputs.triton }}" ]; then
      TRITON_COMMIT_ID="$(<.ci/docker/ci_commit_pins/triton-xpu.txt)"
    else
      TRITON_COMMIT_ID="${{ inputs.triton }}"
    fi
    echo ${TRITON_REPO}@${TRITON_COMMIT_ID}
    if [ "${{ inputs.pytorch }}" != "nightly_wheel" ] || [ ! -z "${{ inputs.triton }}" ]; then
      pip install --force-reinstall "git+${TRITON_REPO}@${TRITON_COMMIT_ID}#subdirectory=python"
    fi
fi
