#!/bin/bash
set -xe
export GIT_PAGER=cat

# Init params
WORKSPACE=$(realpath ${WORKSPACE:-"/tmp"})
PYTORCH_COMMIT=${PYTORCH_COMMIT:-"main"}
TORCH_XPU_OPS_COMMIT=${TORCH_XPU_OPS_COMMIT:-"main"}
PREPARE_ENV=${PREPARE_ENV:-"no"}
for var; do
    eval "export $(echo ${var@Q} |sed "s/^'-*//g;s/=/='/")"
done

if [ "${PYTORCH_COMMIT}" == "search" ];then
    PYTORCH_COMMIT="$(git rev-parse HEAD)"
fi
if [ "${TORCH_XPU_OPS_COMMIT}" == "search" ];then
    TORCH_XPU_OPS_COMMIT="$(git rev-parse HEAD)"
fi

# Clean WORKSPACE
mkdir -p ${WORKSPACE}
rm -rf "${WORKSPACE:?}/"* || sudo rm -rf "${WORKSPACE:?}/"*

# Build pytorch
pip uninstall -y torch torchvision torchaudio pytorch-triton-xpu triton
source $(dirname $(realpath $0))/env.sh 2> /dev/null
build_status="$($(dirname $(realpath $0))/build.sh \
    --WORKSPACE="${WORKSPACE}" \
    --PYTORCH_COMMIT="${PYTORCH_COMMIT}" \
    --TORCH_XPU_OPS_COMMIT="${TORCH_XPU_OPS_COMMIT}" \
    > ${GITHUB_WORKSPACE}/gs-logs/build-${PYTORCH_COMMIT}-${TORCH_XPU_OPS_COMMIT}.log 2>&1 && echo $? || echo $?)"
if [ ${build_status} -ne 0 ];then
    tail -n 100 ${GITHUB_WORKSPACE}/gs-logs/build-${PYTORCH_COMMIT}-${TORCH_XPU_OPS_COMMIT}.log
    echo "Build got failed"
    exit 1
fi
pip list |grep torch

# Install torchvision, torchaudio and triton
cd ${WORKSPACE}/pytorch
rm -rf torch
TORCHVISION_COMMIT_ID="$(cat .github/ci_commit_pins/vision.txt)"
TORCHAUDIO_COMMIT_ID="$(cat .github/ci_commit_pins/audio.txt)"
TRITON_COMMIT_ID="$(cat .ci/docker/ci_commit_pins/triton-xpu.txt)"
git clone https://github.com/pytorch/vision gs-vision
git clone https://github.com/pytorch/audio gs-audio
cd gs-vision && git checkout ${TORCHVISION_COMMIT_ID} && python setup.py install && cd ..
cd gs-audio && git checkout ${TORCHAUDIO_COMMIT_ID} && python setup.py install && cd ..
cd .. && pip install git+https://github.com/intel/intel-xpu-backend-for-triton@${TRITON_COMMIT_ID} || \
                pip install git+https://github.com/intel/intel-xpu-backend-for-triton@${TRITON_COMMIT_ID}#subdirectory=python

if [ "${PREPARE_ENV}" == "yes" ];then
    # deps
    if [[ "${SEARCH_CASE}" == *"benchmarks/dynamo/huggingface.py"* ]];then
        pip install transformers==4.55.2
    elif [[ "${SEARCH_CASE}" == *"benchmarks/dynamo/timm_models.py"* ]];then
        pip install timm==1.0.19
    elif [[ "${SEARCH_CASE}" == *"benchmarks/dynamo/torchbench.py"* ]];then
        model_name="$(echo ${SEARCH_CASE} |sed 's+.*\--only *++;s/ .*//')"
        git clone https://github.com/pytorch/benchmark gs-benchmark
        cd gs-benchmark
        pip install -r requirements.txt
        export PYTHONPATH=${PWD}:${PYTHONPATH}
        python install.py ${model_name}
        pip uninstall -y pynvml
        # for dlrm
        pip install pyre-extensions
        curl -fsSL https://raw.githubusercontent.com/facebookresearch/dlrm/refs/heads/torchrec-dlrm/requirements.txt |xargs pip install
        # for soft_actor_critic, temp fix
        pip install git+https://github.com/nocoding03/gym@fix-np
        cd ..
    else
        pip install -r ./.ci/docker/requirements-ci.txt
    fi
fi

# Test
pip install -U numpy==1.26.4
test_result=1
if [ "${SEARCH_CHECK}" == "accuracy" ];then
    cd ${WORKSPACE}/pytorch
    rm -rf torch
    test_status="$(eval "${SEARCH_CASE} --output=${WORKSPACE}/tmp.csv" \
        > ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_COMMIT}-${TORCH_XPU_OPS_COMMIT}.log 2>&1 && echo $? || echo $?)"
    if [ ${test_status} -eq 0 ];then
        acc_result=$(tail -n 1 ${WORKSPACE}/tmp.csv |awk -F, '{print $4}')
        if [[ "${acc_result}" == "pass"* ]];then
            test_result=0
        fi
    fi
elif [ "${SEARCH_CHECK}" == "performance" ];then
    cd ${WORKSPACE}/pytorch
    rm -rf torch
    test_status="$(eval "${SEARCH_CASE} --output=${WORKSPACE}/tmp.csv" \
        > ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_COMMIT}-${TORCH_XPU_OPS_COMMIT}.log 2>&1 && echo $? || echo $?)"
    if [ ${test_status} -eq 0 ];then
        perf_result=$(tail -n 1 ${WORKSPACE}/tmp.csv |awk -F, '{print $5}')
        test_result=$(echo "${perf_result},${SEARCH_GOOD_VALUE:-"0.00001"},${SEARCH_CRITERIA}" |awk -F, '{
            if ($1/$2 > (1 - $3)){
                print "0";
            }else{
                print "1";
            }
        }')
    fi
elif [ "${SEARCH_CHECK}" == "op_regressions" ];then
    cd ${WORKSPACE}/pytorch/third_party/torch-xpu-ops/test/regressions
    test_status="$(eval "${SEARCH_CASE}" \
        > ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_COMMIT}-${TORCH_XPU_OPS_COMMIT}.log 2>&1 && echo $? || echo $?)"
    if [ ${test_status} -eq 0 ];then
        test_result=0
    fi
elif [ "${SEARCH_CHECK}" == "op_extended" ];then
    cd ${WORKSPACE}/pytorch/third_party/torch-xpu-ops/test/xpu/extended
    test_status="$(eval "${SEARCH_CASE}" \
        > ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_COMMIT}-${TORCH_XPU_OPS_COMMIT}.log 2>&1 && echo $? || echo $?)"
    if [ ${test_status} -eq 0 ];then
        test_result=0
    fi
elif [ "${SEARCH_CHECK}" == "ut_xpu" ];then
    cd ${WORKSPACE}/pytorch/third_party/torch-xpu-ops/test/xpu
    test_status="$(eval "${SEARCH_CASE}" \
        > ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_COMMIT}-${TORCH_XPU_OPS_COMMIT}.log 2>&1 && echo $? || echo $?)"
    if [ ${test_status} -eq 0 ];then
        test_result=0
    fi
else
    cd ${WORKSPACE}/pytorch
    test_status="$(eval "${SEARCH_CASE}" \
        > ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_COMMIT}-${TORCH_XPU_OPS_COMMIT}.log 2>&1 && echo $? || echo $?)"
    if [ ${test_status} -eq 0 ];then
        test_result=0
    fi
fi

# Test result
cat ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_COMMIT}-${TORCH_XPU_OPS_COMMIT}.log
echo "${test_result},${acc_result},${perf_result},${PYTORCH_COMMIT},${TORCH_XPU_OPS_COMMIT}" |\
    tee -a ${GITHUB_WORKSPACE}/gs-logs/summary.csv |tee -a ${WORKSPACE}/result.csv
exit ${test_result}
