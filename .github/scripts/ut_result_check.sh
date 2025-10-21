#!/bin/bash
ut_suite="${1:-op_regression}"   # op_regression / op_extended / op_ut / torch_xpu

# usage
# check_new_failed <UT'log> <Known_issue.log> [output.log]
all_pass=""

check_new_failed() {
    local file_UT="$1"
    local file_known_issue="$2"
    local output_file="${3:-${file_UT%.*}_filtered.log}"

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
    if grep -q $'\r' "$file_UT"; then
        echo "Detected log from windows"
        sed -i 's/\r$//' "$file_UT"
    fi
    grep -vFxf "$file_known_issue" "$file_UT" > "$output_file"

    echo -e "\n\033[1;31m[New failed cases Summary]\033[0m"
    if [[ -z "$(tr -d ' \t\n\r\f' < "$output_file" 2>/dev/null)" ]]; then
        echo -e "\033[1;32mNo new failed cases found\033[0m"
    else
        echo -e "\n\033[1;31mNew failed cases, not in known issues\033[0m"
        cat "$output_file"
    fi
}

check_filtered_logs() {
  local file_UT="$1"
  local file_known_issue="$2"
  local filtered_content="${file_UT%.*}_removed.log"
  # Keep the filtered UT cases
  grep -noFf "$file_known_issue" "$file_UT" > "$filtered_content"
  echo "Filtered cases file: $filtered_content"
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
    if grep -q $'\r' "$file_passed_UT"; then
        echo "Detected log from windows"
        sed -i 's/\r$//' "$file_passed_UT"
    fi
    grep -Fxf "$file_passed_UT" "$file_known_issue" > "$output_file"
    echo -e "\n\033[1;32m[New passed cases Summary]\033[0m"
    if [[ -s "$output_file" ]]; then
        cat "$output_file"
        echo -e "\n\033[1;32mTotal: $(wc -l < "$output_file") known issues are now passing\033[0m"
    else
        echo -e "\033[1;33mNo known issues are now passing\033[0m"
    fi

    rm -f ${output_file}
}

check_test_cases() {
    local log_file="$1"
    declare -A expected_cases=(
        ["op_extended"]=5349
        ["op_regression"]=244
        ["op_regression_dev1"]=1
        ["op_transformers"]=237
        ["op_ut"]=120408
        ["test_xpu"]=69
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


if [[ "${ut_suite}" == 'op_regression' || "${ut_suite}" == 'op_regression_dev1' || "${ut_suite}" == 'op_extended' || "${ut_suite}" == 'op_transformers' || "${ut_suite}" == 'op_ut' || "${ut_suite}" == 'test_xpu' ]]; then
    echo -e "========================================================================="
    echo -e "Show Failed cases in ${ut_suite}"
    echo -e "========================================================================="
    if [[ -f "failures_${ut_suite}.log" ]]; then
      cat "./failures_${ut_suite}.log"
    else
      echo -e "\033[1;32mNo failed cases\033[0m"
    fi
    echo -e "========================================================================="
    echo -e "Checking Test case number for ${ut_suite}"
    echo -e "========================================================================="
    check_test_cases category_${ut_suite}.log
    echo -e "========================================================================="
    echo -e "Checking Filtered cases for ${ut_suite}"
    echo -e "========================================================================="
    if [[ -f "failures_${ut_suite}.log" ]]; then
      check_filtered_logs failures_${ut_suite}.log Known_issue.log
      num_filtered=$(wc -l < "./failures_${ut_suite}_removed.log")
      if [[ $num_filtered -gt 0 ]]; then
          echo -e "\n\033[1;31m[These failed cases are in skip list, will filter]\033[0m"
          awk -F':' '{
              line_number = $1
              $1 = ""
              gsub(/^ /, "", $0)
              printf "\033[33m%3d\033[0m: %s\n", line_number, $0
          }' "failures_${ut_suite}_removed.log"
      else
          echo -e "\n\033[1;32mNo Skipped Cases\033[0m"
      fi
    else
      echo -e "\033[1;32mNo need to check filtered cases\033[0m"
    fi
    echo -e "========================================================================="
    echo -e "Checking New passed cases in Known issue list for ${ut_suite}"
    echo -e "========================================================================="
    check_passed_known_issues passed_${ut_suite}.log Known_issue.log
    echo -e "========================================================================="
    echo -e "Checking New Failed cases in ${ut_suite}"
    echo -e "========================================================================="
    if [[ -f "failures_${ut_suite}.log" ]]; then
      check_new_failed failures_${ut_suite}.log Known_issue.log
    else
      echo -e "\033[1;32mNo need to check failed cases\033[0m"
    fi

    if [[ -f "failures_${ut_suite}_filtered.log" ]]; then
      num_failed=$(wc -l < "./failures_${ut_suite}_filtered.log")
    elif [[ -f "failures_${ut_suite}.log" ]]; then
      num_failed=$(wc -l < "./failures_${ut_suite}.log")
    else
      num_failed=0
    fi
    num_passed=$(wc -l < "./passed_${ut_suite}.log")
    echo -e "========================================================================="
    echo -e "Provide the reproduce command for ${ut_suite}"
    echo -e "========================================================================="
    if [[ $num_failed -gt 0 ]]; then
      echo -e "Need reproduce command"
      if [[ -f "reproduce_${ut_suite}.log" ]]; then
        cat "./reproduce_${ut_suite}.log"
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

if [[ "${ut_suite}" == 'xpu_distributed' ]]; then
    grep -E "^FAILED" xpu_distributed_test.log | awk '{print $3}' > ./"${ut_suite}"_xpu_distributed_test_failed.log
    grep -E "^FAILED" xpu_distributed_test.log | awk '{print $2}' >> ./"${ut_suite}"_xpu_distributed_test_failed.log
    sed -i '/^[^.d]\+/d' ./"${ut_suite}"_xpu_distributed_test_failed.log
    grep "PASSED" xpu_distributed_test.log | awk '{print $1}' > ./"${ut_suite}"_xpu_distributed_test_passed.log
    echo -e "========================================================================="
    echo -e "Show Failed cases in ${ut_suite} xpu distributed"
    echo -e "========================================================================="
    cat "./${ut_suite}_xpu_distributed_test_failed.log"
    echo -e "========================================================================="
    echo -e "Checking Filtered cases for ${ut_suite} xpu distributed"
    echo -e "========================================================================="
    check_filtered_logs "${ut_suite}"_xpu_distributed_test_failed.log Known_issue.log
    num_filtered_xpu_distributed=$(wc -l < "./${ut_suite}_xpu_distributed_test_failed_removed.log")
    if [[ $num_filtered_xpu_distributed -gt 0 ]]; then
        echo -e "\n\033[1;31m[These failed cases are in skip list, will filter]\033[0m"
        awk -F':' '{
            line_number = $1
            $1 = ""
            gsub(/^ /, "", $0)
            printf "\033[33m%3d\033[0m: %s\n", line_number, $0
        }' "${ut_suite}_xpu_distributed_test_failed_removed.log"
    else
        echo -e "\n\033[1;32mNo Skipped Cases\033[0m"
    fi
    echo -e "========================================================================="
    echo -e "Checking New passed cases in Known issue list for ${ut_suite}"
    echo -e "========================================================================="
    check_passed_known_issues "${ut_suite}"_xpu_distributed_test_passed.log Known_issue.log
    echo -e "========================================================================="
    echo -e "Checking Failed cases in ${ut_suite} xpu distributed"
    echo -e "========================================================================="
    check_new_failed "${ut_suite}"_xpu_distributed_test_failed.log Known_issue.log
    if [[ -f "${ut_suite}_xpu_distributed_test_failed_filtered.log" ]]; then
      num_failed_xpu_distributed=$(wc -l < "./${ut_suite}_xpu_distributed_test_failed_filtered.log")
    else
      num_failed_xpu_distributed=$(wc -l < "./${ut_suite}_xpu_distributed_test_failed.log")
    fi
    ((num_failed=num_failed_xpu_distributed))
    num_passed=$(wc -l < "./${ut_suite}_xpu_distributed_test_passed.log")
    if [[ $num_failed -gt 0 ]] || [[ $num_passed -eq 0 ]]; then
      echo -e "[ERROR] UT ${ut_suite} test Fail"
      exit 1
    else
      echo -e "[PASS] UT ${ut_suite} test Pass"
    fi
fi

if [[ "${ut_suite}" == 'skipped_ut' ]]; then
  random_cases=(
    "test_parity__foreach_div_fastpath_inplace_xpu_complex128"
    "test_parity__foreach_div_fastpath_outplace_xpu_complex128"
    "test_parity__foreach_addcdiv_fastpath_inplace_xpu_complex128"
    "test_parity__foreach_addcdiv_fastpath_outplace_xpu_complex128"
    "test_python_ref__refs_log2_xpu_complex128"
  )
  grep "PASSED" skipped_ut_with_skip_test.log | grep -vFf <(printf '%s\n' "${random_cases[@]}") > ./skipped_ut_with_skip_test_passed.log
  num_passed=$(wc -l < "./skipped_ut_with_skip_test_passed.log")
  if [ ${num_passed} -gt 0 ];then
    echo -e "========================================================================="
    echo -e "Checking New passed cases in Skip list for ${ut_suite}"
    echo -e "========================================================================="
    cat ./skipped_ut_with_skip_test_passed.log
    echo -e "[Warning] Has ${num_passed} new pass in ${ut_suite}"
    exit 1
  fi
fi
