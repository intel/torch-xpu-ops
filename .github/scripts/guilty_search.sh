#!/bin/bash
set -xe
export GIT_PAGER=cat

# Init params
WORKSPACE=$(realpath ${WORKSPACE:-"/tmp"})
PYTORCH_VERSION=${PYTORCH_VERSION:-"main"}
TORCH_XPU_OPS_VERSION=${TORCH_XPU_OPS_VERSION:-"main"}
for var; do
    eval "export $(echo ${var@Q} |sed "s/^'-*//g;s/=/='/")"
done

# Clean WORKSPACE
mkdir -p ${WORKSPACE}
rm -rf ${WORKSPACE}/* || sudo rm -rf ${WORKSPACE}/*

# Build pytorch
pip uninstall -y torch
source $(dirname $(realpath $0))/env.sh 2> /dev/null
build_status="$($(dirname $(realpath $0))/build.sh \
    --WORKSPACE="${WORKSPACE}" \
    --PYTORCH_VERSION="${PYTORCH_VERSION}" \
    --TORCH_XPU_OPS_VERSION="${TORCH_XPU_OPS_VERSION}" \
    > ${GITHUB_WORKSPACE}/gs-logs/build-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log 2>&1 && echo $? || echo $?)"
if [ ${build_status} -ne 0 ];then
    tail -n 100 ${GITHUB_WORKSPACE}/gs-logs/build-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log
    echo "Build got failed"
    exit 1
fi
pip list |grep torch

# Test
test_result=1
if [ "${SEARCH_CHECK}" == "accuracy" ];then
    cd ${WORKSPACE}/pytorch
    test_status="$(eval "${SEARCH_CASE} --output=${WORKSPACE}/tmp.csv" \
        > ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log 2>&1 && echo $? || echo $?)"
    if [ ${test_status} -eq 0 ];then
        acc_result=$(tail -n 1 ${WORKSPACE}/tmp.csv |awk -F, '{print $4}')
        if [[ "${acc_result}" == "pass"* ]];then
            test_result=0
        fi
    else
        tail -n 100 ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log
    fi
elif [ "${SEARCH_CHECK}" == "performance" ];then
    cd ${WORKSPACE}/pytorch
    test_status="$(eval "${SEARCH_CASE} --output=${WORKSPACE}/tmp.csv" \
        > ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log 2>&1 && echo $? || echo $?)"
    if [ ${test_status} -eq 0 ];then
        perf_result=$(tail -n 1 ${WORKSPACE}/tmp.csv |awk -F, '{print $5}')
        test_result=$(echo "${perf_result},${SEARCH_GOOD_VALUE:-"0.00001"},${SEARCH_CRITERIA}" |awk -F, '{
            if ($1/$2 > (1 - $3)){
                print "0";
            }else{
                print "1";
            }
        }')
    else
        tail -n 100 ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log
    fi
elif [ "${SEARCH_CHECK}" == "ut_regressions" ];then
    cd ${WORKSPACE}/pytorch/third_party/torch-xpu-ops/test/regressions
    test_status="$(eval "${SEARCH_CASE}" \
        > ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log 2>&1 && echo $? || echo $?)"
    if [ ${test_status} -eq 0 ];then
        test_result=0
    else
        tail -n 100 ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log
    fi
elif [ "${SEARCH_CHECK}" == "ut_extended" ];then
    cd ${WORKSPACE}/pytorch/third_party/torch-xpu-ops/test/xpu/extended
    test_status="$(eval "${SEARCH_CASE}" \
        > ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log 2>&1 && echo $? || echo $?)"
    if [ ${test_status} -eq 0 ];then
        test_result=0
    else
        tail -n 100 ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log
    fi
elif [ "${SEARCH_CHECK}" == "ut_xpu" ];then
    cd ${WORKSPACE}/pytorch/third_party/torch-xpu-ops/test/xpu
    test_status="$(eval "${SEARCH_CASE}" \
        > ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log 2>&1 && echo $? || echo $?)"
    if [ ${test_status} -eq 0 ];then
        test_result=0
    else
        tail -n 100 ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log
    fi
else
    test_status="$(eval "${SEARCH_CASE}" \
        > ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log 2>&1 && echo $? || echo $?)"
    if [ ${test_status} -eq 0 ];then
        test_result=0
    else
        tail -n 100 ${GITHUB_WORKSPACE}/gs-logs/test-${PYTORCH_VERSION}-${TORCH_XPU_OPS_VERSION}.log
    fi
fi

# Test result
echo "${test_result},${acc_result},${perf_result},${PYTORCH_VERSION},${TORCH_XPU_OPS_VERSION}" |\
    tee ${GITHUB_WORKSPACE}/gs-logs/summary.csv |tee ${WORKSPACE}/result.csv
exit ${test_result}
