#!/bin/bash

# Wrapper script to run all analyses and show statistics using total_errors count

echo "Running all profile analysis checks..."
echo "======================================"

# Arrays to track results
declare -a PASSED_TESTS
declare -a FAILED_TESTS
declare -A ERROR_COUNTS
declare -A LOG_FILES

# Function to extract error count from analysis output
extract_error_count() {
    local output="$1"
    echo "$output" | grep -E "Total errors:? [0-9]+" | tail -1 | grep -oE '[0-9]+' | head -1
}

# Function to run test and track result
run_test() {
    local test_name=$1
    local log_file=$2
    local analysis_type=$3
    
    echo
    echo "Running: $test_name ($log_file)"
    echo "--------------------------------"

    local output
    output=$(bash profile_ut_result_summary.sh "$log_file" "$analysis_type" 2>&1)
    local error_count

    error_count=$(extract_error_count "$output")

    if [ -z "$error_count" ]; then
        error_count=1
        echo "⚠️  Could not extract error count, assuming test failed"
    fi

    echo "$output"

    ERROR_COUNTS["$test_name"]=$error_count
    LOG_FILES["$test_name"]=$log_file
    
    if [ "$error_count" -eq 0 ]; then
        PASSED_TESTS+=("$test_name")
        echo "✅ $test_name: PASSED (0 errors)"
    else
        FAILED_TESTS+=("$test_name")
        echo "❌ $test_name: FAILED ($error_count errors)"
    fi
}

# Run all tests
run_test "correlation_id_mixed" "correlation_id_mixed.log" 1
run_test "reproducer_missing_gpu_kernel_time" "reproducer.missing.gpu.kernel.time.log" 2
run_test "time_precision" "time_precision_in_profile.log" 3
run_test "partial_runtime_ops" "profile_partial_runtime_ops.log" 4
run_test "triton_xpu_ops" "triton_xpu_ops_time.log" 5
run_test "profiling_fp32_train_resnet50" "profiling.fp32.train.pt" 6

# Display detailed summary
echo
echo "======================================"
echo "           DETAILED TEST SUMMARY"
echo "======================================"

echo "✅ PASSED TESTS: ${#PASSED_TESTS[@]}"
for test in "${PASSED_TESTS[@]}"; do
    printf "  - %-40s: 0 errors\n" "$test"
done

echo
echo "❌ FAILED TESTS: ${#FAILED_TESTS[@]}"
for test in "${FAILED_TESTS[@]}"; do
    printf "  - %-40s: %d errors (File: %s)\n" "$test" "${ERROR_COUNTS[$test]}" "${LOG_FILES[$test]}"
done

echo
echo "--------------------------------------"
echo "STATISTICS:"
echo "  Total tests run:    $(( ${#PASSED_TESTS[@]} + ${#FAILED_TESTS[@]} ))"
echo "  Tests passed:       ${#PASSED_TESTS[@]}"
echo "  Tests failed:       ${#FAILED_TESTS[@]}"
echo "  Total errors found: $(printf "%s\n" "${ERROR_COUNTS[@]}" | awk '{sum+=$1} END {print sum}')"

# Final result
if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo
    echo "🎉 ALL TESTS PASSED! No errors found in any analysis."
    exit 0
else
    echo
    echo "⚠️  SOME TESTS FAILED! Please check the following analyses:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "    - $test (${ERROR_COUNTS[$test]} errors) - File: ${LOG_FILES[$test]}"
    done
    exit 1
fi