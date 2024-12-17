transformers_test=${1:-backbone}
let expected_fail_number=0

grep "^FAILED" summary_short.txt | awk '{print $2}' > ./${transformers_test}_failed.log
grep "PASSED" summary_short.txt | awk '{print $2}' > ./${transformers_test}_passed.log
grep "SKIPPED" summary_short.txt | awk -F "] " '{print $2}' > ./${transformers_test}_skipped.log
num_failed=$(cat ./${transformers_test}_failed.log | wc -l)
num_passed=$(cat ./${transformers_test}_passed.log | wc -l)
num_skipped=$(cat ./${transformers_test}_skipped.log | wc -l)
num_errors=$(grep "errors" stats.txt | awk -F " " '{print $10}')

echo -e "========================================================================="
echo -e "Show results in ${transformers_test}"
echo -e "========================================================================="
echo -e "Pass: $num_passed"
echo -e "Fail: $num_failed"
echo -e "Skip: $num_skipped"
echo -e "Error: $num_errors"
echo -e "========================================================================="
printf "%-10s %-4s %-4s %-4s\n" Testgroup Passed Failed Skipped
printf "%-10s %-4s %-4s %-4s\n" ${transformers_test} $num_passed $num_failed $num_skipped
echo -e "========================================================================="
echo -e "========================================================================="

case ${transformers_test} in 
    tests_py)
    let expected_fail_number=8
    ;;
    tests_benchmark)
    let expected_fail_number=0
    ;;
    tests_generation)
    let expected_fail_number=18
    ;;
    tests_models)
    let expected_fail_number=407
    ;;
    tests_pipelines)
    let expected_fail_number=9
    ;;
    tests_trainer)
    let expected_fail_number=3
    ;;
    tests_utils)
    let expected_fail_number=1
    ;;
    backbone)
    let expected_fail_number=0
    ;;
esac

if [[ "$num_failed" -gt "$expected_fail_number" ]] || [[ "$num_passed -le 0" ]] || [[ "$num_errors -ne 0" ]]; then
    echo -e "[FAIL] ${transformers_test} test Fail"
    exit 1
else
    echo -e "[PASS] ${transformers_test} test Pass"
fi
