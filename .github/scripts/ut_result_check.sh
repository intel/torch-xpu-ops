ut_suite=${1:-op_regression}   # op_regression / op_extended / op_ut / torch_xpu

if [[ ${ut_suite} == 'op_regression' || ${ut_suite} == 'op_regression_dev1' || ${ut_suite} == 'op_extended' ]]; then
    grep "^FAILED" ${ut_suite}_test.log | awk '{print $2}' > ./${ut_suite}_failed.log
    grep "PASSED" ${ut_suite}_test.log | awk '{print $1}' > ./${ut_suite}_passed.log
    num_failed=$(cat ./${ut_suite}_failed.log | wc -l)
    num_passed=$(cat ./${ut_suite}_passed.log | wc -l)
    echo -e "========================================================================="
    echo -e "Show Failed cases in ${ut_suite}"
    echo -e "========================================================================="
    cat ./${ut_suite}_failed.log
    if [[ $num_failed -gt 0 ]] || [[ $num_passed -le 0 ]]; then
      echo -e "[ERROR] UT ${ut_suite} test Fail"
      exit 1
    else
      echo -e "[PASS] UT ${ut_suite} test Pass"
    fi
fi
if [[ ${ut_suite} == 'op_ut' ]]; then
    grep "^FAILED" op_ut_with_skip_test.log | awk '{print $2}' > ./${ut_suite}_with_skip_test_failed.log
    grep "^FAILED" op_ut_with_only_test.log | awk '{print $2}' > ./${ut_suite}_with_only_test_failed.log
    num_failed_with_skip=$(cat ./${ut_suite}_with_skip_test_failed.log | wc -l)
    num_failed_with_only=$(cat ./${ut_suite}_with_only_test_failed.log | wc -l)
    echo -e "========================================================================="
    echo -e "Show Failed cases in ${ut_suite} with skip"
    echo -e "========================================================================="
    cat ./${ut_suite}_with_skip_test_failed.log
    echo -e "========================================================================="
    echo -e "Show Failed cases in ${ut_suite} with only"
    echo -e "========================================================================="
    cat ./${ut_suite}_with_only_test_failed.log
    let num_failed=num_failed_with_skip+num_failed_with_only
    grep "PASSED" op_ut_with_skip_test.log | awk '{print $1}' > ./${ut_suite}_with_skip_test_passed.log
    grep "PASSED" op_ut_with_only_test.log | awk '{print $1}' > ./${ut_suite}_with_only_test_passed.log
    num_passed_with_skip=$(cat ./${ut_suite}_with_skip_test_passed.log | wc -l)
    num_passed_with_only=$(cat ./${ut_suite}_with_only_test_passed.log | wc -l)
    let num_passed=num_passed_with_skip+num_passed_with_only
    if [[ $num_failed -gt 0 ]] || [[ $num_passed -le 0 ]]; then
      echo -e "[ERROR] UT ${ut_suite} test Fail"
      exit 1
    else
      echo -e "[PASS] UT ${ut_suite} test Pass"
    fi
fi
if [[ ${ut_suite} == 'torch_xpu' ]]; then
    echo "Pytorch XPU binary UT checking"
    cd ../../pytorch
    TEST_REPORTS_DIR=$(pwd)/test/test-reports
    for xpu_case in build/bin/*{xpu,sycl}*; do
      if [[ "$xpu_case" != *"*"* && "$xpu_case" != *.so && "$xpu_case" != *.a ]]; then
        case_name=$(basename "$xpu_case")
        cd ../ut_log/torch_xpu
        grep -E "FAILED" binary_ut_${ut_suite}_${case_name}_test.log | awk '{print $2}' > ./binary_ut_${ut_suite}_${case_name}_failed.log
        echo $(cat ./binary_ut_${ut_suite}_${case_name}_failed.log | wc -l) | tee -a ./binary_ut_${ut_suite}_failed_summary.log
        grep -E "PASSED|Pass" binary_ut_${ut_suite}_${case_name}_test.log | awk '{print $2}' > ./binary_ut_${ut_suite}_${case_name}_passed.log
        echo $(cat ./binary_ut_${ut_suite}_${case_name}_passed.log | wc -l) | tee -a ./binary_ut_${ut_suite}_passed_summary.log
        cd -
      fi
    done
    echo -e "========================================================================="
    echo -e "Show Failed cases in ${ut_suite}"
    echo -e "========================================================================="
    cd ../ut_log/torch_xpu
    cat ./binary_ut_${ut_suite}_${case_name}_failed.log
    num_failed_binary_ut=$(awk '{sum += $1};END {print sum}' binary_ut_${ut_suite}_failed_summary.log)
    num_passed_binary_ut=$(awk '{sum += $1};END {print sum}' binary_ut_${ut_suite}_passed_summary.log)
    let num_failed=num_failed_binary_ut
    if [[ $num_failed -gt 0 ]] || [[ $num_passed_binary_ut -le 0 ]]; then
      echo -e "[ERROR] UT ${ut_suite} test Fail"
      exit 1
    else
      echo -e "[PASS] UT ${ut_suite} test Pass"
    fi
fi
