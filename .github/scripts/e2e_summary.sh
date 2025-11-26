#!/usr/bin/env bash

set -euo pipefail

# Script: test_results_processor.sh
# Description: Process accuracy and performance test results for XPU operations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_NAME

# Constants
readonly RED='🔴'
readonly GREEN='🟢'
readonly BLUE='🔵'
readonly YELLOW='🟡'

# Global state
accuracy_regression=0
performance_regression=0

main() {
    if [[ $# -ne 2 ]]; then
        echo "Usage: $0 <results_dir> <reference_dir>" >&2
        exit 1
    fi

    local results_dir="$1"
    local reference_dir="$2"

    validate_directories "$results_dir" "$reference_dir"
    cleanup_temp_files

    echo "Processing test results..."
    echo "Results: $results_dir, Reference: $reference_dir"

    process_accuracy "$results_dir"
    process_performance "$results_dir" "$reference_dir"
    generate_report

    echo "Processing completed"
}

validate_directories() {
    for dir in "$1" "$2"; do
        if [[ ! -d "$dir" ]]; then
            echo "Error: Directory not found: $dir" >&2
            exit 1
        fi
    done
}

cleanup_temp_files() {
    rm -rf /tmp/tmp-*.txt /tmp/tmp-*.json
    rm -rf accuracy.*.html performance.*.html
}

# Accuracy Processing
process_accuracy() {
    local results_dir="$1"

    if ! find "$results_dir" -name "*_xpu_accuracy.csv" -quit; then
        return
    fi

    echo "Processing accuracy results..."

    # Get known issues
    python "$SCRIPT_DIR/../scripts/get_issue.py" \
        --repo_owner intel \
        --repo_name torch-xpu-ops \
        --labels "module: infra" E2E Accuracy skipped \
        --output /tmp/tmp-known-issue.json

    generate_accuracy_summary "$results_dir"
    generate_accuracy_details "$results_dir"
}

generate_accuracy_summary() {
    local results_dir="$1"
    local check_file="$SCRIPT_DIR/../ci_expected_accuracy/check_expected.py"

    cat > accuracy.summary.html << EOF

#### accuracy

| Category | Total | Passed | Pass Rate | Failed | Xfailed | Timeout | New Passed | New Enabled | Not Run |
|----------|-------|--------|-----------|--------|---------|---------|------------|-------------|---------|
EOF

    while IFS= read -r csv_file; do
        process_csv_file "$check_file" "$csv_file"
    done < <(find "$results_dir" -name "*_xpu_accuracy.csv" | sort)
    echo -e "\n\n" >> accuracy.summary.html
}

process_csv_file() {
    local check_file="$1" csv_file="$2"
    local category suite mode dtype

    category=$(basename "$csv_file" | sed 's/inductor_//;s/_xpu_accuracy.*//')
    suite=$(echo "$csv_file" | sed 's/.*inductor_//;s/_.*//;s/timm/timm_models/')
    mode=$(echo "$csv_file" | sed 's/_xpu_accuracy.*//;s/.*_//')
    dtype=$(echo "$csv_file" | sed -E 's/.*inductor_[a-z]*_//;s/models_//;s/_infer.*|_train.*//')

    local tmp_file="/tmp/tmp-${suite}-${mode}-${dtype}.txt"

    python "$check_file" \
        --suite "$suite" \
        --mode "$mode" \
        --dtype "$dtype" \
        --issue_file /tmp/tmp-known-issue.json \
        --csv_file "$csv_file" > "$tmp_file"

    local result
    result=$(parse_test_results "$tmp_file")
    echo "| $category | $result |" >> accuracy.summary.html
}

parse_test_results() {
    local tmp_file="$1"
    sed 's/, /,/g' "$tmp_file" | awk '
    BEGIN {
        total = passed = pass_rate = failed = xfail = timeout = 0
        new_passed = new_enabled = not_run = 0
    }
    /Total models:/ { total = $3 }
    /Passed models:/ { passed = $3 }
    /Pass rate:/ { pass_rate = $3 }
    /Real failed models:/ { failed = format_count($4, "🔴") }
    /Expected failed models:/ { xfail = format_count($4, "🔵") }
    /Warning timeout models:/ { timeout = format_count($4, "🟡") }
    /Failed to passed models:/ { new_passed = format_count($5, "🟢") }
    /Not run.in models:/ { not_run = $4 }
    /New models:/ { new_enabled = format_count($3, "🔵") }

    function format_count(count, icon) {
        return count > 0 ? icon count : count
    }

    END {
        printf "%s | %s | %s | %s | %s | %s | %s | %s | %s",
            total, passed, pass_rate, failed, xfail, timeout, new_passed, new_enabled, not_run
    }'
}

generate_accuracy_details() {
    local results_dir="$1"

    # Create table headers
    cat > accuracy.details.html << EOF

#### accuracy

<table>
<thead>
    <tr>
        <th rowspan="2">Suite</th>
        <th rowspan="2">Model</th>
        <th colspan="5">Training</th>
        <th colspan="5">Inference</th>
    </tr>
    <tr>
        <th>float32</th><th>bfloat16</th><th>float16</th><th>amp_bf16</th><th>amp_fp16</th>
        <th>float32</th><th>bfloat16</th><th>float16</th><th>amp_bf16</th><th>amp_fp16</th>
    </tr>
</thead>
<tbody>
EOF

    cp accuracy.details.html accuracy.regression.html

    # Process all test suites
    while IFS= read -r suite; do
        process_suite "$results_dir" "$suite"
    done < <(find "$results_dir" -name "*_xpu_accuracy.csv" | \
        sed 's/.*inductor_//;s/_[abf].*//' | sort | uniq)

    echo -e "</tbody></table>\n\n" >> accuracy.details.html
    echo -e "</tbody></table>\n\n" >> accuracy.regression.html

    # Clear regression file if no issues
    if [[ $accuracy_regression -eq 0 ]]; then
        rm -f accuracy.regression.html
    fi
}

process_suite() {
    local results_dir="$1" suite="$2"

    while IFS= read -r model; do
        process_model "$results_dir" "$suite" "$model"
    done < <(get_models_for_suite "$results_dir" "$suite")
}

get_models_for_suite() {
    local results_dir="$1" suite="$2"
    find "$results_dir" -name "*${suite}*_xpu_accuracy.csv" -exec cat {} \; | \
        grep "^xpu," | cut -d, -f2 | sort | uniq
}

process_model() {
    local results_dir="$1" suite="$2" model="$3"
    local -A results=()

    # Collect results for all data types and modes
    for dtype in float32 bfloat16 float16 amp_bf16 amp_fp16; do
        for mode in training inference; do
            local key="${mode}_${dtype}"
            results[$key]=$(get_model_result "$results_dir" "$suite" "$model" "$dtype" "$mode")
        done
    done

    local row
    row=$(generate_html_row "$suite" "$model" "${results[@]}")

    if [[ "$row" =~ ${RED}|${GREEN}|${YELLOW} ]]; then
        echo "$row" | tee -a accuracy.details.html >> accuracy.regression.html
        accuracy_regression=1
        echo "acc 1" >> /tmp/tmp-acc-result.txt
    else
        echo "$row" >> accuracy.details.html
        echo "acc 0" >> /tmp/tmp-acc-result.txt
    fi
}

get_model_result() {
    local results_dir="$1" suite="$2" model="$3" dtype="$4" mode="$5"
    local tmp_file="/tmp/tmp-${suite}-${mode}-${dtype}.txt"
    local color="black"

    if [[ -f "$tmp_file" ]] && grep -q -w "$model" "$tmp_file"; then
        color=$(determine_color "$tmp_file" "$model")
    fi

    local value
    value=$(find "$results_dir" -name "*${suite}_${dtype}_${mode}_xpu_accuracy.csv" -type f | \
        head -1 | xargs grep -h ",${model}," 2>/dev/null | cut -d, -f4 | head -1)

    if [[ "$color" != "black" ]]; then
        echo "${color}${value}"
    else
        echo "${value}"
    fi
}

determine_color() {
    local tmp_file="$1" model="$2"
    grep -w "$model" "$tmp_file" | awk '
        /Real failed models:/ { print "🔴"; exit }
        /Expected failed models:|New models:/ { print "🔵"; exit }
        /Warning timeout models:/ { print "🟡"; exit }
        /Failed to passed models:/ { print "🟢"; exit }
        { print "black" }
    ' | head -1
}

generate_html_row() {
    local suite="$1" model="$2"
    shift 2
    local results=("$@")

    cat << EOF
<tr>
    <td>$suite</td>
    <td>$model</td>
    <td>${results[0]}</td><td>${results[1]}</td><td>${results[2]}</td><td>${results[3]}</td><td>${results[4]}</td>
    <td>${results[5]}</td><td>${results[6]}</td><td>${results[7]}</td><td>${results[8]}</td><td>${results[9]}</td>
</tr>
EOF
}

# Performance Processing
process_performance() {
    local results_dir="$1" reference_dir="$2"

    if ! find "$results_dir" -name "*_xpu_performance.csv" -quit; then
        return
    fi

    echo "Processing performance results..."

    local perf_args=("--target" "$results_dir" "--baseline" "$reference_dir")
    if [[ "${GITHUB_EVENT_NAME:-}" == "pull_request" ]]; then
        perf_args+=("--pr")
    fi

    python "$SCRIPT_DIR/perf_comparison.py" "${perf_args[@]}"

    if [[ -f "performance.regression.html" ]]; then
        performance_regression=1
    fi

    update_best_performance "$results_dir" "$reference_dir"
}

update_best_performance() {
    local results_dir="$1" reference_dir="$2"
    local best_file="$results_dir/best.csv"
    local output_file="${GITHUB_OUTPUT:-/dev/null}"

    cp "$reference_dir/best.csv" "$best_file" 2>/dev/null || true

    python "$SCRIPT_DIR/calculate_best_perf.py" \
        --new "$results_dir" \
        --best "$best_file" \
        --device PVC1100 \
        --os "${OS_PRETTY_NAME:-}" \
        --driver "${DRIVER_VERSION:-}" \
        --oneapi "${BUNDLE_VERSION:-}" \
        --gcc "${GCC_VERSION:-}" \
        --python "${python:-}" \
        --pytorch "${TORCH_BRANCH_ID:-}/${TORCH_COMMIT_ID:-}" \
        --torch-xpu-ops "${TORCH_XPU_OPS_COMMIT:-${GITHUB_SHA:-}}"

    echo "performance_regression=$performance_regression" >> "$output_file"
}

# Report Generation
generate_report() {
    local summary_file="e2e-test-result.html"

    {
        generate_header
        generate_highlights
        generate_summary
        generate_details
    } > "$summary_file"

    echo "Report generated: $summary_file"
}

generate_header() {
    cat << EOF

#### Note:
🔴: Failed cases needing investigation
🟢: New passed cases needing reference update
🔵: Expected failed or new enabled cases
🟡: Warning cases
Empty: Cases not run

EOF
}

generate_highlights() {
    if (( accuracy_regression + performance_regression > 0 )); then
        echo -e "### 🎯 Highlight regressions\n"
        [[ $accuracy_regression -gt 0 ]] && cat accuracy.regression.html
        [[ $performance_regression -gt 0 ]] && cat performance.regression.html
    else
        echo -e "### ✅ No regressions detected\n"
    fi
}

generate_summary() {
    echo "### 📊 Summary"
    echo
    cat accuracy.summary.html 2>/dev/null || echo "No accuracy data"
    echo
    cat performance.summary.html 2>/dev/null || echo "No performance data"
    echo
}

generate_details() {
    cat << EOF
### 📖 Details

<details>
<summary>View detailed result</summary>

EOF

    cat accuracy.details.html 2>/dev/null || echo "No accuracy details"
    cat performance.details.html 2>/dev/null || echo "No performance details"

    echo "</details>"
}

# Run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
