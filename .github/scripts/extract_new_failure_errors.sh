#!/bin/bash
# Extract error messages/tracebacks for new failures from pytest log files
# Usage: bash extract_new_failure_errors.sh <new_failures_file> <test_log_dir> <output_file>
#
# Arguments:
#   new_failures_file: File with new failure test names (one per line, format: file::class::test)
#   test_log_dir: Directory containing *_test.log files
#   output_file: Output file to write extracted errors

set -euo pipefail

new_failures_file="${1:?Usage: $0 <new_failures_file> <test_log_dir> <output_file>}"
test_log_dir="${2:?Usage: $0 <new_failures_file> <test_log_dir> <output_file>}"
output_file="${3:?Usage: $0 <new_failures_file> <test_log_dir> <output_file>}"

if [[ ! -s "$new_failures_file" ]]; then
    echo "No new failures to extract." > "$output_file"
    exit 0
fi

: > "$output_file"

# Find all test log files
log_files=$(find "$test_log_dir" -name "*_test.log" -o -name "*_test_error.log" 2>/dev/null | sort)

if [[ -z "$log_files" ]]; then
    echo "Warning: No test log files found in $test_log_dir" >> "$output_file"
    exit 0
fi

failure_count=0
while IFS= read -r test_name; do
    [[ -z "$test_name" ]] && continue
    failure_count=$((failure_count + 1))

    {
        echo "========================================"
        echo "FAILURE #${failure_count}: ${test_name}"
        echo "========================================"
    } >> "$output_file"

    # Extract the short test name (last part after ::) for searching
    short_name=$(echo "$test_name" | awk -F'::' '{print $NF}')
    found=false

    for log_file in $log_files; do
        # Search for FAILED line or error traceback related to this test
        # Pytest outputs: FAILED test_file::TestClass::test_name - reason
        # and also has a section starting with "_ _ _" or "ERRORS" or "FAILURES"
        # with the full traceback

        # Try to extract the FAILED section from pytest short summary
        if grep -Fq -- "$short_name" "$log_file" 2>/dev/null; then
            # Extract traceback block: look for the test name in FAILURES section
            # Pytest format: ___ test_name ___  followed by traceback until next ___ or ====
            error_block=$(awk -v test="$short_name" '
                BEGIN { printing=0; buffer="" }
                /^_{3,}.*_{3,}$/ || /^={3,}.*={3,}$/ {
                    if (printing) {
                        printing=0
                        print buffer
                        buffer=""
                    }
                    header = $0
                    gsub(/^[_=[:space:]]+/, "", header)
                    gsub(/[_=[:space:]]+$/, "", header)
                    start = length(header) - length(test) + 1
                    if (start > 0 && substr(header, start) == test) {
                        if (start == 1) {
                            delim_ok = 1
                        } else {
                            prev = substr(header, start - 1, 1)
                            delim_ok = (prev == ":" || prev == "." || prev == "/" || prev == " " || prev == "]")
                        }
                    } else {
                        delim_ok = 0
                    }
                    if (delim_ok) {
                        printing=1
                        buffer=$0 "\n"
                        next
                    }
                }
                printing {
                    buffer=buffer $0 "\n"
                }
            ' "$log_file" 2>/dev/null)

            if [[ -n "$error_block" ]]; then
                echo "$error_block" >> "$output_file"
                found=true
                break
            fi

            # Fallback: grab the FAILED line with reason
            failed_line=$(awk -v test="$short_name" '
                BEGIN { count=0 }
                {
                    failed_pos = index($0, "FAILED")
                    test_pos = index($0, test)
                    if (failed_pos > 0 && test_pos > failed_pos) {
                        print
                        count++
                        if (count == 5) {
                            exit
                        }
                    }
                }
            ' "$log_file" 2>/dev/null || true)
            if [[ -n "$failed_line" ]]; then
                echo "$failed_line" >> "$output_file"
                found=true
                break
            fi
        fi
    done

    if [[ "$found" == "false" ]]; then
        echo "(No traceback found in test logs for this failure)" >> "$output_file"
    fi
    echo "" >> "$output_file"

done < "$new_failures_file"

{
    echo "========================================"
    echo "Total new failures: ${failure_count}"
    echo "========================================"
} >> "$output_file"
