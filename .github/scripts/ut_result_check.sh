#!/bin/bash
ut_suite="${1:-op_regression}"   # op_regression / op_extended / op_ut / torch_xpu

if [[ "${ut_suite}" == 'op_regression' || "${ut_suite}" == 'op_regression_dev1' || "${ut_suite}" == 'op_extended' ]]; then
    grep -E "^FAILED|have failures" "${ut_suite}"_test.log | awk '{print $2}' > ./"${ut_suite}"_failed.log
    grep "PASSED" "${ut_suite}"_test.log | awk '{print $1}' > ./"${ut_suite}"_passed.log
    num_failed=$(wc -l < "./${ut_suite}_failed.log")
    num_passed=$(wc -l < "./${ut_suite}_passed.log")
    echo -e "========================================================================="
    echo -e "Show Failed cases in ${ut_suite}"
    echo -e "========================================================================="
    cat "./${ut_suite}_failed.log"
    if [[ $num_failed -gt 0 ]] || [[ $num_passed -le 0 ]]; then
      echo -e "[ERROR] UT ${ut_suite} test Fail"
      exit 1
    else
      echo -e "[PASS] UT ${ut_suite} test Pass"
    fi
fi
if [[ "${ut_suite}" == 'op_ut' ]]; then
    grep -E "^FAILED|have failures" op_ut_with_skip_test.log | awk '{print $2}' > ./"${ut_suite}"_with_skip_test_failed.log
    grep -E "^FAILED|have failures" op_ut_with_only_test.log | awk '{print $2}' > ./"${ut_suite}"_with_only_test_failed.log
    num_failed_with_skip=$(wc -l < "./${ut_suite}_with_skip_test_failed.log")
    num_failed_with_only=$(wc -l < "./${ut_suite}_with_only_test_failed.log")
    echo -e "========================================================================="
    echo -e "Show Failed cases in ${ut_suite} with skip"
    echo -e "========================================================================="
    cat "./${ut_suite}_with_skip_test_failed.log"
    echo -e "========================================================================="
    echo -e "Show Failed cases in ${ut_suite} with only"
    echo -e "========================================================================="
    cat "./${ut_suite}_with_only_test_failed.log"
    ((num_failed=num_failed_with_skip+num_failed_with_only))
    grep "PASSED" op_ut_with_skip_test.log | awk '{print $1}' > ./"${ut_suite}"_with_skip_test_passed.log
    grep "PASSED" op_ut_with_only_test.log | awk '{print $1}' > ./"${ut_suite}"_with_only_test_passed.log
    num_passed_with_skip=$(wc -l < "./${ut_suite}_with_skip_test_passed.log")
    num_passed_with_only=$(wc -l < "./${ut_suite}_with_only_test_passed.log")
    ((num_passed=num_passed_with_skip+num_passed_with_only))
    if [[ $num_failed -gt 0 ]] || [[ $num_passed -le 0 ]]; then
      echo -e "[ERROR] UT ${ut_suite} test Fail"
      exit 1
    else
      echo -e "[PASS] UT ${ut_suite} test Pass"
    fi
fi
if [[ "${ut_suite}" == 'torch_xpu' ]]; then
    echo "Pytorch XPU binary UT checking"
    cd ../../pytorch || exit
    for xpu_case in build/bin/*{xpu,sycl}*; do
      if [[ "$xpu_case" != *"*"* && "$xpu_case" != *.so && "$xpu_case" != *.a ]]; then
        case_name=$(basename "$xpu_case")
        cd ../ut_log/torch_xpu || exit
        grep -E "FAILED|have failures" binary_ut_"${ut_suite}"_"${case_name}"_test.log | awk '{print $2}' > ./binary_ut_"${ut_suite}"_"${case_name}"_failed.log
        wc -l < "./binary_ut_${ut_suite}_${case_name}_failed.log" | tee -a ./binary_ut_"${ut_suite}"_failed_summary.log
        grep -E "PASSED|Pass" binary_ut_"${ut_suite}"_"${case_name}"_test.log | awk '{print $2}' > ./binary_ut_"${ut_suite}"_"${case_name}"_passed.log
        wc -l < "./binary_ut_${ut_suite}_${case_name}_passed.log" | tee -a ./binary_ut_"${ut_suite}"_passed_summary.log
        cd - || exit
      fi
    done
    echo -e "========================================================================="
    echo -e "Show Failed cases in ${ut_suite}"
    echo -e "========================================================================="
    cd ../ut_log/torch_xpu || exit
    cat "./binary_ut_${ut_suite}_${case_name}_failed.log"
    num_failed_binary_ut=$(awk '{sum += $1};END {print sum}' binary_ut_"${ut_suite}"_failed_summary.log)
    num_passed_binary_ut=$(awk '{sum += $1};END {print sum}' binary_ut_"${ut_suite}"_passed_summary.log)
    ((num_failed=num_failed_binary_ut))
    if [[ $num_failed -gt 0 ]] || [[ $num_passed_binary_ut -le 0 ]]; then
      echo -e "[ERROR] UT ${ut_suite} test Fail"
      exit 1
    else
      echo -e "[PASS] UT ${ut_suite} test Pass"
    fi
fi
if [[ "${ut_suite}" == 'xpu_distributed' ]]; then
    grep -E "^FAILED|have failures" xpu_distributed_test.log | awk '{print $2}' > ./"${ut_suite}"_xpu_distributed_test_failed.log
    num_failed_xpu_distributed=$(wc -l < "./${ut_suite}_xpu_distributed_test_failed.log")
    echo -e "========================================================================="
    echo -e "Show Failed cases in ${ut_suite} xpu distributed"
    echo -e "========================================================================="
    cat "./${ut_suite}_xpu_distributed_test_failed.log"
    ((num_failed=num_failed_xpu_distributed))
    if [[ $num_failed -gt 0 ]]; then
      echo -e "[ERROR] UT ${ut_suite} test Fail"
      exit 1
    else
      echo -e "[PASS] UT ${ut_suite} test Pass"
    fi
fi
