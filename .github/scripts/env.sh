#!/bin/bash

input_type="$1"
action_type="$2"
conda_env="${3:-"xpu_op_${ZE_AFFINITY_MASK:-"0"}"}"

# Prapre oneAPI DLE
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

# Prapre Python Env
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
    conda deactivate && conda deactivate
    if [[ "${action_type}" == *"create"* ]];then
        conda remove --all -y -n "${conda_env}" || rm -rf $(dirname ${CONDA_EXE})/../envs/${conda_env}
        conda create python=${INPUTS_PYTHON:-"3.10"} cmake ninja -n "${conda_env}" -y
    fi
    conda activate "${conda_env}"
    conda info -e
elif [[ "${input_type}" == *"venv"* ]];then
    if [[ "${action_type}" == *"create"* ]];then
        rm -rf ${HOME}/xpu-venv/${conda_env}
        eval "/usr/bin/python${INPUTS_PYTHON:-"3.10"} -m venv ${HOME}/xpu-venv/${conda_env}"
    fi
    source ${HOME}/xpu-venv/${conda_env}/bin/activate
fi

# Prepare PyTorch
if [[ "${input_type}" == *"pytorch"* ]];then
    if [[ "${INPUTS_PYTORCH}" != *"nightly_wheel"* ]]; then
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
    if [[ "${INPUTS_PYTORCH}" != *"nightly_wheel"* ]]; then
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
    TRITON_COMMIT_ID="${INPUTS_TRITON:-"$(<.ci/docker/ci_commit_pins/triton-xpu.txt)"}"
    echo ${TRITON_REPO}@${TRITON_COMMIT_ID}
    if [ "${INPUTS_PYTORCH}" != "nightly_wheel" ] || [ ! -z "${INPUTS_TRITON}" ]; then
        pip install --force-reinstall "git+${TRITON_REPO}@${TRITON_COMMIT_ID}#subdirectory=python"
    fi
fi
