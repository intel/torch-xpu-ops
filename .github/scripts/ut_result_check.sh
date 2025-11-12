#!/bin/bash
# Test Suite Runner for Intel Torch-XPU-Ops
# Usage: ./script.sh <test_suite>
# Available suites: op_regression, op_extended, op_ut, test_xpu, xpu_distributed, skipped_ut
ut_suite="${1:-op_regression}"  # Default to op_regression if no suite specified
# Expected test case counts for each test suite category
# Used to detect significant test case reductions (>5%)
declare -A EXPECTED_CASES=(
    ["op_extended"]=5349
    ["op_regression"]=244
    ["op_regression_dev1"]=1
    ["op_transformers"]=237
    ["op_ut"]=120408
    ["test_xpu"]=69
)
# Tests that are known to randomly pass and should be ignored when detecting new passes
# These are typically flaky tests that don't indicate real improvements
IGNORE_TESTS=(
    "test_parity__foreach_div_fastpath_inplace_xpu_complex128"
    "test_parity__foreach_div_fastpath_outplace_xpu_complex128"
    "test_parity__foreach_addcdiv_fastpath_inplace_xpu_complex128"
    "test_parity__foreach_addcdiv_fastpath_outplace_xpu_complex128"
    "test_python_ref__refs_log2_xpu_complex128"
    "_jiterator_"  # Pattern to match all jiterator tests
)
# Find new failed test cases that are not in the known issues list
# Args: UT_results_file, known_issues_file, [output_file]
check_new_failed() {
    local ut_file="$1" known_file="$2" output_file="${3:-${ut_file%.*}_filtered.log}"
    if [[ $# -lt 2 ]]; then
        echo "❌ Need 2 files to compare" >&2
        return 1
    fi
    # Handle Windows line endings (CRLF -> LF)
    if grep -q $'\r' "$ut_file"; then
        sed -i 's/\r$//' "$ut_file"
    fi
    # Remove known issues from current failures to find new failures
    grep -vFxf "$known_file" "$ut_file" > "$output_file"
    echo -e "\\n📊 New Failed Cases Summary:"
    if [[ -s "$output_file" ]]; then
        echo -e "❌ New failures found:\\n"
        cat "$output_file"
    else
        echo -e "✅ No new failed cases"
    fi
}
# Find known issues that are now passing (regression fixes)
# Args: passed_tests_file, known_issues_file, [output_file]
check_passed_known_issues() {
    local passed_file="$1" known_file="$2" output_file="${3:-${passed_file%.*}_passed_known.log}"
    if [[ $# -lt 2 ]]; then
        echo "❌ Need 2 files to compare" >&2
        return 1
    fi
    # Handle Windows line endings
    if grep -q $'\r' "$passed_file"; then
        sed -i 's/\r$//' "$passed_file"
    fi
    # Find known issues that are now passing (intersection of passed tests and known issues)
    grep -Fxf "$passed_file" "$known_file" > "$output_file"
    echo -e "\\n📊 New Passing Known Issues:"
    if [[ -s "$output_file" ]]; then
        local count
        count=$(wc -l < "$output_file")
        cat "$output_file"
        echo -e "✅ ${count} known issues are now passing!"
    else
        echo -e "ℹ️  No known issues are now passing"
    fi
    rm -f "$output_file"  # Clean up temporary file
}
# Verify test case counts haven't dropped significantly (>5% reduction)
# Args: category_log_file
check_test_cases() {
    local log_file="$1"
    if [[ ! -f "$log_file" ]]; then
        echo "❌ File not found: ${log_file}" >&2
        return 1
    fi
    local all_pass="true" current_category=""
    # Parse category log file to extract test counts per category
    while IFS= read -r line; do
        if [[ "$line" =~ ^Category:\ ([^[:space:]]+) ]]; then
            current_category="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ Test\ cases:\ ([0-9]+) ]] && [[ -n "$current_category" ]]; then
            local actual="${BASH_REMATCH[1]}" expected="${EXPECTED_CASES[$current_category]}"
            if [[ -n "$expected" ]]; then
                # Calculate 95% threshold for acceptable reduction
                local threshold reduction
                threshold=$(echo "$expected * 0.95" | bc -l | awk '{print int($1+0.5)}')
                reduction=$(echo "scale=2; ($actual/$expected - 1) * 100" | bc -l)
                echo "📈 ${current_category}:"
                echo "   Expected: ${expected}, Actual: ${actual}"
                echo "   Threshold: ${threshold}, Reduction: ${reduction}%"
                if [[ "$actual" -lt "$threshold" ]]; then
                    echo "   Status: ❌ Abnormal (>5% reduction)"
                    all_pass="false"
                else
                    echo "   Status: ✅ Normal"
                fi
                echo "----------------------------------------"
            fi
            current_category=""
        fi
    done < "$log_file"
    echo "$all_pass"  # Return overall status
}
# Clean test case files: extract test names, sort, and remove duplicates
clean_file() {
    local file="$1"
    if [[ ! -s "$file" ]]; then
        return 0
    fi
    awk '{for(i=1;i<=NF;i++) if($i ~ /::.*::/) print $i}' "$file" | sort -u > "${file}.tmp"
    mv "${file}.tmp" "$file"
}
# Special check for skipped UT suite - detect newly passing tests that were previously skipped
check_skipped_ut() {
    echo "🔍 Checking for newly passed tests in skipped UT..."
    local test_file="skipped_ut_with_skip_test.log"
    local known_file="known-passed-issue.cases"
    local new_file="new-passed-issue.cases"
    local result_file="new-passed-this-run.cases"
    # Fetch known passing tests from GitHub issue tracking known passes
    if gh --repo intel/torch-xpu-ops issue view "${NEW_PASSED_ISSUE:-2333}" --json body -q .body 2>/dev/null | grep "::.*::" > "$known_file"; then
        echo "✅ Fetched known tests from GitHub"
    else
        echo "⚠️  Using empty known tests file"
        : > "$known_file"
    fi
    if [[ ! -f "$test_file" ]]; then
        echo "❌ Test log not found: ${test_file}" >&2
        return 1
    fi
    # Extract current passing tests from test log
    if ! grep "PASSED" "$test_file" | grep "::.*::" > "$new_file"; then
        : > "$new_file"
    fi
    clean_file "$known_file"
    clean_file "$new_file"
    # Find tests that are passing now but weren't in known passes, excluding ignored tests
    comm -13 "$known_file" "$new_file" | grep -vFf <(printf '%s\n' "${IGNORE_TESTS[@]}") > "$result_file"
    local new_count
    new_count=$(wc -l < "$result_file" 2>/dev/null || echo 0)
    if [[ "$new_count" -gt 0 ]]; then
        echo "❌ ${new_count} NEW PASSING TESTS:"
        cat "$result_file"
        echo "Please review these tests!"
        return 1
    else
        echo "✅ No new passing tests found"
        # Update GitHub issue with current passing tests for future reference
        gh --repo intel/torch-xpu-ops issue edit "${NEW_PASSED_ISSUE:-2333}" --body-file "$new_file"
    fi
}
# Main test runner for standard test suites (op_regression, op_extended, etc.)
run_main_tests() {
    local suite="$1"
    echo "========================================================================="
    echo "Running tests for: ${suite}"
    echo "========================================================================="
    # Display failed test cases
    echo "📋 Failed Cases:"
    if [[ -f "failures_${suite}.log" ]]; then
        cat "failures_${suite}.log"
    else
        echo "✅ No failed cases"
    fi
    # Check test case counts for significant reductions
    echo -e "\\n📊 Test Case Counts:"
    local all_pass
    all_pass=$(check_test_cases "category_${suite}.log")
    # Identify and display tests that are filtered out (known issues)
    echo -e "\\n🔍 Filtered Cases:"
    if [[ -f "failures_${suite}.log" ]]; then
        local filtered_count
        grep -noFf "Known_issue.log" "failures_${suite}.log" > "failures_${suite}_removed.log"
        filtered_count=$(wc -l < "failures_${suite}_removed.log")
        if [[ "$filtered_count" -gt 0 ]]; then
            echo "⏩ Skipping ${filtered_count} known issues:"
            awk -F':' '{printf "   Line %3d: %s\n", $1, $2}' "failures_${suite}_removed.log"
        else
            echo "✅ No skipped cases"
        fi
    fi
    # Check for known issues that are now passing (regression fixes)
    echo -e "\\n✅ Passing Known Issues:"
    check_passed_known_issues "passed_${suite}.log" "Known_issue.log"
    # Check for new failures not in known issues
    echo -e "\\n❌ New Failures:"
    if [[ -f "failures_${suite}.log" ]]; then
        check_new_failed "failures_${suite}.log" "Known_issue.log"
    fi
    # Calculate final statistics
    local failed_count=0 passed_count=0
    if [[ -f "failures_${suite}_filtered.log" ]]; then
        failed_count=$(wc -l < "failures_${suite}_filtered.log")
    fi
    if [[ -f "passed_${suite}.log" ]]; then
        passed_count=$(wc -l < "passed_${suite}.log")
    fi
    # Final test result determination
    echo -e "\\n📈 Final Summary:"
    echo "   Failed: ${failed_count}, Passed: ${passed_count}, All counts normal: ${all_pass}"
    if [[ "$failed_count" -gt 0 ]] || [[ "$passed_count" -le 0 ]] || [[ "$all_pass" == "false" ]]; then
        echo "❌ TEST FAILED: ${suite}"
        exit 1
    else
        echo "✅ TEST PASSED: ${suite}"
    fi
}
# Special runner for distributed test suite (different log format)
run_distributed_tests() {
    local suite="$1"
    echo "========================================================================="
    echo "Running distributed tests for: ${suite}"
    echo "========================================================================="
    # Process distributed test logs (different format than main tests)
    grep -E "^FAILED" "${suite}_test.log" | awk '{print $3 "\n" $2}' | grep -v '^[^.d]\+$' > "${suite}_failed.log"
    grep "PASSED" "${suite}_test.log" | awk '{print $1}' > "${suite}_passed.log"
    echo "📋 Failed Cases:"
    cat "${suite}_failed.log"
    # Identify filtered tests (known issues in distributed tests)
    echo -e "\\n🔍 Filtered Cases:"
    local filtered_count
    grep -noFf "Known_issue.log" "${suite}_failed.log" > "${suite}_removed.log"
    filtered_count=$(wc -l < "${suite}_removed.log")
    if [[ "$filtered_count" -gt 0 ]]; then
        echo "⏩ Skipping ${filtered_count} known issues"
        awk -F':' '{printf "   Line %3d: %s\n", $1, $2}' "${suite}_removed.log"
    else
        echo "✅ No skipped cases"
    fi
    # Run standard checks for distributed tests
    check_passed_known_issues "${suite}_passed.log" "Known_issue.log"
    check_new_failed "${suite}_failed.log" "Known_issue.log"
    # Calculate final statistics for distributed tests
    local failed_count=0 passed_count=0
    if [[ -f "${suite}_failed_filtered.log" ]]; then
        failed_count=$(wc -l < "${suite}_failed_filtered.log")
    fi
    passed_count=$(wc -l < "${suite}_passed.log")
    # Final result determination for distributed tests
    if [[ "$failed_count" -gt 0 ]] || [[ "$passed_count" -eq 0 ]]; then
        echo "❌ TEST FAILED: ${suite}"
        exit 1
    else
        echo "✅ TEST PASSED: ${suite}"
    fi
}
# Main dispatcher - route to appropriate test runner based on suite type
case "$ut_suite" in
    op_regression|op_regression_dev1|op_extended|op_transformers|op_ut|test_xpu)
        run_main_tests "$ut_suite"
        ;;
    xpu_distributed)
        run_distributed_tests "$ut_suite"
        ;;
    skipped_ut)
        check_skipped_ut
        ;;
    *)
        echo "❌ Unknown test suite: ${ut_suite}" >&2
        echo "💡 Available: op_regression, op_extended, op_ut, test_xpu, xpu_distributed, skipped_ut" >&2
        exit 1
        ;;
esac
