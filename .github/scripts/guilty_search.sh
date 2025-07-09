#!/bin/bash
set -x
set +e
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

# Clean WORKSPACE
mkdir -p ${WORKSPACE}
rm -rf ${WORKSPACE}/* || sudo rm -rf ${WORKSPACE}/*

# Build pytorch
pip uninstall -y torch
source $(dirname $(realpath $0))/env.sh
$(dirname $(realpath $0))/build.sh \
    --WORKSPACE="${WORKSPACE}" \
    --PYTORCH_REPO="${PYTORCH_REPO}" \
    --PYTORCH_VERSION="${PYTORCH_VERSION}" \
    --TORCH_XPU_OPS_REPO="${TORCH_XPU_OPS_REPO}" \
    --TORCH_XPU_OPS_VERSION="${TORCH_XPU_OPS_VERSION}" \
    > ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log 2>&1
pip list |grep torch

# Test
test_result=1
if [ "${SEARCH_CHECK}" == "accuracy" ];then
    cd ${WORKSPACE}/pytorch
    eval "${SEARCH_CASE} --output=${WORKSPACE}/tmp.csv" \
        > ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log 2>&1
    if [ $? -eq 0 ];then
        acc_result=$(tail -n 1 ${WORKSPACE}/tmp.csv |awk -F, '{print $4}')
        if [[ "${acc_result}" == "pass"* ]];then
            test_result=0
        fi
    fi
elif [ "${SEARCH_CHECK}" == "performance" ];then
    cd ${WORKSPACE}/pytorch
    eval "${SEARCH_CASE} --output=${WORKSPACE}/tmp.csv" \
        > ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log 2>&1
    if [ $? -eq 0 ];then
        perf_result=$(tail -n 1 ${WORKSPACE}/tmp.csv |awk -F, '{print $5}')
        test_result=$(echo "${perf_result},${SEARCH_GOOD_VALUE},${SEARCH_CRITERIA}" |awk -F, '{
            if ($1/$2 > (1 - $3)){
                print "0";
            }else{
                print "1";
            }
        }')
    fi
elif [ "${SEARCH_CHECK}" == "ut_regressions" ];then
    cd ${WORKSPACE}/pytorch/third_party/torch-xpu-ops/test/regressions
    eval "${SEARCH_CASE}" > ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log 2>&1
    if [ $? -eq 0 ];then
        test_result=0
    fi
elif [ "${SEARCH_CHECK}" == "ut_extended" ];then
    cd ${WORKSPACE}/pytorch/third_party/torch-xpu-ops/test/xpu/extended
    eval "${SEARCH_CASE}" > ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log 2>&1
    if [ $? -eq 0 ];then
        test_result=0
    fi
elif [ "${SEARCH_CHECK}" == "ut_xpu" ];then
    cd ${WORKSPACE}/pytorch/third_party/torch-xpu-ops/test/xpu
    eval "${SEARCH_CASE}" > ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log 2>&1
    if [ $? -eq 0 ];then
        test_result=0
    fi
else
    eval "${SEARCH_CASE}" > ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log 2>&1
    if [ $? -eq 0 ];then
        test_result=0
    fi
fi

# Test result
echo "${test_result},${acc_result},${perf_result},${PYTORCH_VERSION},${TORCH_XPU_OPS_VERSION}" |\
    tee ${GITHUB_WORKSPACE}/gs-logs/summary.csv |tee ${WORKSPACE}/result.csv
exit ${test_result}
