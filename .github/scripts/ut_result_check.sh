#!/bin/bash
ut_suite="${1:-op_regression}"   # op_regression / op_extended / op_ut / torch_xpu

# usage
# compare_and_filter_logs <UT'log> <Known_issue.log> [output.log]
all_pass=""

compare_and_filter_logs() {
    local file_UT="$1"
    local file_known_issue="$2"
    local output_file="${3:-${file_UT%.*}_filtered.log}"
    local filtered_content="${file_UT%.*}_removed.log"

    if [[ $# -lt 2 ]]; then
        echo "[ERROR] Need 2 files to compare"
        return 1
    fi

    # Check whether UT's failed log contains the case of the known issue'log
    echo "Checking whether $file_UT contains $file_known_issue"
    if grep -qxFf "$file_known_issue" "$file_UT"; then
        echo "$file_UT contains $file_known_issue"
    else
        echo "$file_UT does not contain $file_known_issue"
    fi

    # Filter the same content from file_UT as file_known_issue
    echo "Filtering $file_known_issue for $file_UT"
    grep -vFxf "$file_known_issue" "$file_UT" > "$output_file"

    # Keep the filtered UT cases
    grep -noFf "$file_known_issue" "$file_UT" > "$filtered_content"
    echo "Filtered cases file: $filtered_content"

    if [[ -s "$filtered_content" ]]; then
        echo -e "\n\033[1;31m[These failed cases are in skip list, will filter]\033[0m"
        awk -F':' '{
            line_number = $1
            $1 = ""
            gsub(/^ /, "", $0)
            printf "\033[33m%3d\033[0m: %s\n", line_number, $0
        }' "$filtered_content"
    else
        echo -e "\n\033[1;32mNo Skipped Cases\033[0m"
    fi

    echo -e "\n\033[1;31m[New failed cases Summary]\033[0m"
    if [[ -z "$(tr -d ' \t\n\r\f' < "$output_file" 2>/dev/null)" ]]; then
        echo -e "\033[1;32mNo new failed cases found\033[0m"
    else
        echo -e "\n\033[1;31mNew failed cases, not in known issues\033[0m"
        cat "$output_file"
    fi
}

check_passed_known_issues() {
    local file_passed_UT="$1"
    local file_known_issue="$2"
    local output_file="${3:-${file_passed_UT%.*}_passed_known_issues.log}"
    if [[ $# -lt 2 ]]; then
        echo "[ERROR] Need 2 files to compare"
        return 1
    fi
    echo "Checking for known issues that are now passing in $file_passed_UT"
    grep -Fxf "$file_passed_UT" "$file_known_issue" > "$output_file"
    echo -e "\n\033[1;32m[New passed cases Summary]\033[0m"
    if [[ -s "$output_file" ]]; then
        cat "$output_file"
        echo -e "\n\033[1;32mTotal: $(wc -l < "$output_file") known issues are now passing\033[0m"
    else
        echo -e "\033[1;33mNo known issues are now passing\033[0m"
    fi
}

check_test_cases() {
    local log_file="$1"
    declare -A expected_cases=(
        ["op_extended"]=5349
        ["op_regression"]=244
        ["op_regression_dev1"]=1
        ["op_transformers"]=237
        ["op_ut"]=120408
    )

    if [[ ! -f "$log_file" ]]; then
        echo "False"
        echo "[ERROR] Need test file $log_file" >&2
        return 1
    fi

    all_pass="true"
    local current_category=""

    while IFS= read -r line; do
        if [[ $line =~ ^Category:\ ([^[:space:]]+) ]]; then
            current_category="${BASH_REMATCH[1]}"
        elif [[ $line =~ Test\ cases:\ ([0-9]+) ]] && [[ -n "$current_category" ]]; then
            actual_cases="${BASH_REMATCH[1]}"
            expected_cases_value="${expected_cases[$current_category]}"
            
            if [[ -n "$expected_cases_value" ]]; then
                threshold=$(echo "$expected_cases_value * 0.95" | bc -l | awk '{print int($1+0.5)}')

                echo "Category: $current_category"
                echo "Expected number: $expected_cases_value"
                echo "Current number: $actual_cases"
                echo "Threshold(95%): $threshold"

                if [[ "$actual_cases" -lt "$threshold" ]]; then
                    echo "  Status: ❌ Abnormal (reduction exceeds 5%)"
                    all_pass="false"
                else
                    reduction=$(echo "scale=2; ($actual_cases/$expected_cases_value - 1) * 100" | bc -l)
                    echo "  Status: ✅ Normal (reduction ${reduction}%)"
                fi
                echo "----------------------------------------"
            fi
            current_category=""
        fi
    done < "$log_file"
}


if [[ "${ut_suite}" == 'op_regression' || "${ut_suite}" == 'op_regression_dev1' || "${ut_suite}" == 'op_extended' || "${ut_suite}" == 'op_transformers' || "${ut_suite}" == 'op_ut' ]]; then
    echo -e "========================================================================="
    echo -e "Show Failed cases in ${ut_suite}"
    echo -e "========================================================================="
    cat "./failures_${ut_suite}.log"
    echo -e "========================================================================="
    echo -e "Checking Failed cases in ${ut_suite}"
    echo -e "========================================================================="
    compare_and_filter_logs failures_${ut_suite}.log Known_issue.log
    echo -e "========================================================================="
    echo -e "Checking New passed cases in Known issue list for ${ut_suite}"
    echo -e "========================================================================="
    check_passed_known_issues passed_${ut_suite}.log Known_issue.log
    echo -e "========================================================================="
    echo -e "Checking Test case number for ${ut_suite}"
    echo -e "========================================================================="
    check_test_cases category_${ut_suite}.log
    if [[ -f "failures_${ut_suite}_filtered.log" ]]; then
      num_failed=$(wc -l < "./failures_${ut_suite}_filtered.log")
    else
      num_failed=$(wc -l < "./failures_${ut_suite}.log")
    fi
    num_passed=$(wc -l < "./passed_${ut_suite}.log")
    echo -e "========================================================================="
    echo -e "Provide the reproduce command for ${ut_suite}"
    echo -e "========================================================================="
    if [[ $num_failed -gt 0 ]]; then
      echo -e "Need reproduce command"
      if [[ -f "reproduce.log" ]]; then
        cat "./reproduce.log"
      fi
    else
      echo -e "Not need reproduce command"
    fi
    if [[ $num_failed -gt 0 ]] || [[ $num_passed -le 0 ]] || [[ "$all_pass" == 'false' ]]; then
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
        grep -E "FAILED" binary_ut_"${ut_suite}"_"${case_name}"_test.log | awk '{print $2}' > ./binary_ut_"${ut_suite}"_"${case_name}"_failed.log
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
    grep -E "^FAILED" xpu_distributed_test.log | awk '{print $2}' > ./"${ut_suite}"_xpu_distributed_test_failed.log
    grep "PASSED" xpu_distributed_test.log | awk '{print $1}' > ./"${ut_suite}"_xpu_distributed_test_passed.log
    echo -e "========================================================================="
    echo -e "Show Failed cases in ${ut_suite} xpu distributed"
    echo -e "========================================================================="
    cat "./${ut_suite}_xpu_distributed_test_failed.log"
    echo -e "========================================================================="
    echo -e "Checking Failed cases in ${ut_suite} xpu distributed"
    echo -e "========================================================================="
    compare_and_filter_logs "${ut_suite}"_xpu_distributed_test_failed.log Known_issue.log
    echo -e "========================================================================="
    echo -e "Checking New passed cases in Known issue list for ${ut_suite}"
    echo -e "========================================================================="
    check_passed_known_issues "${ut_suite}"_xpu_distributed_test_passed.log Known_issue.log
    if [[ -f "${ut_suite}_xpu_distributed_test_failed_filtered.log" ]]; then
      num_failed_xpu_distributed=$(wc -l < "./${ut_suite}_xpu_distributed_test_failed_filtered.log")
    else
      num_failed_xpu_distributed=$(wc -l < "./${ut_suite}_xpu_distributed_test_failed.log")
    fi
    ((num_failed=num_failed_xpu_distributed))
    if [[ $num_failed -gt 0 ]]; then
      echo -e "[ERROR] UT ${ut_suite} test Fail"
      exit 1
    else
      echo -e "[PASS] UT ${ut_suite} test Pass"
    fi
fi
