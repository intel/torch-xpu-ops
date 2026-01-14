#!/bin/bash
# Enhanced script for CPU/XPU/CUDA device Dynamo benchmark tests with regression checking

set -euo pipefail

# Default configuration
declare -A DEFAULTS=(
    ["SUITE"]="huggingface"     # huggingface / torchbench / timm_models
    ["DTYPE"]="float32"         # float32 / float16 / amp (amp_bf16) / amp_fp16
    ["MODE"]="inference"        # inference / training
    ["SCENARIO"]="accuracy"     # accuracy / performance
    ["DEVICE"]="xpu"            # xpu / cuda
    ["CARD"]="0"                # 0 / 1 / 2 / 3 ...
    ["SHAPE"]="static"          # static / dynamic
    ["NUM_SHARDS"]=""           # num test shards
    ["SHARD_ID"]=""             # shard id
    ["MODEL_ONLY"]=""           # GoogleFnet / T5Small / ...
    ["REFERENCE_DIR"]=""        # Reference result dir
    ["REGRESSION_CHECK"]="false" # Enable regression checking
    ["PERF_RERUN_THRESHOLD"]="0.9" # Threshold for performance rerun (0.9 = 90% of reference)
    ["PERF_RERUN_COUNT"]="3"    # Number of times to rerun performance tests
    ["SKIP_ACCURACY"]="false"   # Skip accuracy tests in rerun mode
)

# Function to display usage
show_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run inductor tests with flexible configuration options.

Options:
  -s, --suite SUITE          Test suite: huggingface, torchbench, timm_models (default: ${DEFAULTS[SUITE]})
  -d, --dtype DTYPE          Data type: float32, float16, amp, amp_bf16, amp_fp16 (default: ${DEFAULTS[DTYPE]})
  -m, --mode MODE            Mode: inference, training (default: ${DEFAULTS[MODE]})
  -c, --scenario SCENARIO    Scenario: accuracy, performance (default: ${DEFAULTS[SCENARIO]})
  -e, --device DEVICE        Device: xpu, cuda (default: ${DEFAULTS[DEVICE]})
  -a, --card CARD            Device card index (default: ${DEFAULTS[CARD]})
  -p, --shape SHAPE          Shape: static, dynamic (default: ${DEFAULTS[SHAPE]})
  -n, --num-shards NUM       Number of shards for parallel execution
  -i, --shard-id ID          Shard ID for this execution
  -o, --model-only MODEL     Run only specific model (supports -k pattern)
  -w, --workspace DIR        Workspace directory (default: current)
  -l, --log-dir DIR          Log directory (default: \$WORKSPACE/inductor_log/\$SUITE/\$DTYPE)
  -r, --reference-dir DIR    Reference result directory
  --regression-check         Enable regression checking against reference
  --perf-rerun-threshold VAL Performance rerun threshold (default: ${DEFAULTS[PERF_RERUN_THRESHOLD]})
  --perf-rerun-count NUM     Number of performance reruns (default: ${DEFAULTS[PERF_RERUN_COUNT]})
  --skip-accuracy            Skip accuracy tests in rerun mode
  -h, --help                 Show this help message

Examples:
  $(basename "$0") --suite huggingface --dtype amp_bf16 --mode inference
  $(basename "$0") -s torchbench -d float16 -m training -c performance
  $(basename "$0") --model-only "GoogleFnet" --device cuda --card 1
  $(basename "$0") --num-shards 4 --shard-id 2 --model-only "-k T5"
  $(basename "$0") --regression-check --reference-dir /path/to/reference
  $(basename "$0") --regression-check --perf-rerun-threshold 0.8 --perf-rerun-count 5

EOF
}

# Parse command-line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--suite)
                SUITE="$2"
                shift 2
                ;;
            -d|--dtype)
                DTYPE="$2"
                shift 2
                ;;
            -m|--mode)
                MODE="$2"
                shift 2
                ;;
            -c|--scenario)
                SCENARIO="$2"
                shift 2
                ;;
            -e|--device)
                DEVICE="$2"
                shift 2
                ;;
            -a|--card)
                CARD="$2"
                shift 2
                ;;
            -p|--shape)
                SHAPE="$2"
                shift 2
                ;;
            -n|--num-shards)
                NUM_SHARDS="$2"
                shift 2
                ;;
            -i|--shard-id)
                SHARD_ID="$2"
                shift 2
                ;;
            -o|--model-only)
                MODEL_ONLY="$2"
                shift 2
                ;;
            -w|--workspace)
                WORKSPACE="$2"
                shift 2
                ;;
            -l|--log-dir)
                LOG_DIR="$2"
                shift 2
                ;;
            -r|--reference-dir)
                REFERENCE_DIR="$2"
                shift 2
                ;;
            --regression-check)
                REGRESSION_CHECK="true"
                shift
                ;;
            --perf-rerun-threshold)
                PERF_RERUN_THRESHOLD="$2"
                shift 2
                ;;
            --perf-rerun-count)
                PERF_RERUN_COUNT="$2"
                shift 2
                ;;
            --skip-accuracy)
                SKIP_ACCURACY="true"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Initialize variables with defaults or empty values
for key in "${!DEFAULTS[@]}"; do
    declare "$key"="${DEFAULTS[$key]}"
done

# Parse command line arguments
parse_args "$@"

# Set defaults if not provided via command line
WORKSPACE="${WORKSPACE:-$(pwd)}"

# Configure log directory
LOG_DIR="${LOG_DIR:-${WORKSPACE}/inductor_log/${SUITE}/${DTYPE}}"

# Create log directory with verbose output
echo "Creating log directory: ${LOG_DIR}"
mkdir -p "${LOG_DIR}" || {
    echo "Error: Failed to create log directory ${LOG_DIR}"
    exit 1
}

# Generate log file name
LOG_NAME="inductor_${SUITE}_${DTYPE}_${MODE}_${DEVICE}_${SCENARIO}"

# Display configuration
print_configuration() {
    echo "========================================"
    echo "Test Configuration:"
    echo "========================================"
    echo "Suite:          ${SUITE}"
    echo "Data Type:      ${DTYPE}"
    echo "Mode:           ${MODE}"
    echo "Scenario:       ${SCENARIO}"
    echo "Device:         ${DEVICE}"
    echo "Card:           ${CARD}"
    echo "Shape:          ${SHAPE}"
    echo "Workspace:      ${WORKSPACE}"
    echo "Log Directory:  ${LOG_DIR}"
    echo "Log Name:       ${LOG_NAME}"
    echo "Regression Check: ${REGRESSION_CHECK}"
    echo "Perf Rerun Threshold: ${PERF_RERUN_THRESHOLD}"
    echo "Perf Rerun Count: ${PERF_RERUN_COUNT}"
    echo "Skip Accuracy: ${SKIP_ACCURACY}"

    if [[ -n "${NUM_SHARDS}" && -n "${SHARD_ID}" ]]; then
        echo "Shards:         ${SHARD_ID}/${NUM_SHARDS}"
    fi

    if [[ -n "${MODEL_ONLY}" ]]; then
        echo "Model Filter:   ${MODEL_ONLY}"
    fi

    if [[ -n "${REFERENCE_DIR}" ]]; then
        echo "Reference Dir:  ${REFERENCE_DIR}"
    fi
    echo "========================================"
}

print_configuration

# Validate regression check configuration
if [[ "${REGRESSION_CHECK}" == "true" ]]; then
    if [[ -z "${REFERENCE_DIR}" ]]; then
        echo "ERROR: Regression check enabled but no reference directory specified!"
        echo "Please provide --reference-dir or -r option"
        exit 1
    fi

    if [[ ! -d "${REFERENCE_DIR}" ]]; then
        echo "ERROR: Reference directory does not exist: ${REFERENCE_DIR}"
        exit 1
    fi

    echo "Regression checking enabled. Will compare against reference in: ${REFERENCE_DIR}"
fi

# Function to extract unique test cases from CSV files
extract_test_cases() {
    local dir=$1
    local output_file=$2

    echo "Extracting test cases from $dir..."

    # Find all CSV files and concatenate, removing header from all but first
    for csv in $(find $dir -type f -name "*.summary.csv"); do
        if [ -f "$csv" ]; then
            tail -n +2 "$csv" >> "$output_file"
        fi
    done

    if [ ! -f "$output_file" ] || [ ! -s "$output_file" ]; then
        echo "ERROR: No CSV files found in $dir"
        exit 1
    fi

    echo "Extracted $(wc -l < "$output_file") lines from $dir"
}

# Function to find performance regression test cases
find_perf_reg_cases() {
    local new_file=$1
    local ref_file=$2
    local output_file=$3

    echo "Finding matching test cases..."

    # Create a combined view with both results
    # Using awk to match on Suite,Dtype,Mode,Scenario,Model,BS
    awk -F',' '
    BEGIN {
        OFS=","
    }
    NR==FNR {
        # Read new results
        key = $1 "," $2 "," $3 "," $4 "," $5 "," $6
        new_eager = $8
        new_inductor = $9
        new_data[key] = new_eager "," new_inductor
        next
    }
    {
        # Read reference results
        key = $1 "," $2 "," $3 "," $4 "," $5 "," $6
        ref_eager = $8
        ref_inductor = $9

        if (key in new_data) {
            split(new_data[key], new_vals, ",")
            new_eager_val = new_vals[1]
            new_inductor_val = new_vals[2]

            # Calculate ratios
            if (new_eager_val + 0 != 0) {
                eager_ratio = ref_eager / new_eager_val
            } else {
                eager_ratio = 0
            }

            if (new_inductor_val + 0 != 0) {
                inductor_ratio = ref_inductor / new_inductor_val
            } else {
                inductor_ratio = 0
            }

            # Check for regression
            regression = "NO"
            regression_type = ""
            if (eager_ratio < 0.9) {
                regression = "YES"
                regression_type = "Eager"
            }
            if (inductor_ratio < 0.9) {
                regression = "YES"
                regression_type = "Inductor"
            }

            print $0, new_eager_val, new_inductor_val, eager_ratio, inductor_ratio, regression_type, regression
        }
    }' "$new_file" "$ref_file" > "$output_file"

    # Add header to the combined file
    header="Suite,Dtype,Mode,Scenario,Model,BS,Acc,Ref_Eager,Ref_Inductor,New_Eager,New_Inductor,Eager_Ratio,Inductor_Ratio,Regression_type,Regression"
    sed -i "1s/^/$header\n/" "$output_file"

    echo "Generated combined comparison file: $output_file"
}

# Function to run performance tests for specific models
run_performance_tests() {
    local models_file=$1
    local rerun_count=$2
    local base_log_dir="${LOG_DIR}/rerun"

    mkdir -p "${base_log_dir}"

    # Parse models that need rerun
    local models_to_rerun=()
    while IFS=',' read -r suite dtype mode scenario model batch_size accuracy ref_eager ref_inductor new_eager new_inductor eager_ratio inductor_ratio regression_type regression; do
        if [[ "$regression" == "YES" ]] && [[ "$scenario" == "performance" ]]; then
            models_to_rerun+=("$model,$batch_size,$regression_type")
        fi
    done < <(tail -n +2 "$models_file")  # Skip header

    if [[ ${#models_to_rerun[@]} -eq 0 ]]; then
        echo "No performance regression cases found for rerun."
        return 0
    fi

    # Remove duplicates
    IFS=$'\n' unique_models=($(sort -u <<<"${models_to_rerun[*]}"))
    unset IFS

    echo "Found ${#unique_models[@]} models with performance regression for rerun:"
    printf '%s\n' "${unique_models[@]}"

    # Run performance tests for each model multiple times
    for ((run=1; run<=rerun_count; run++)); do
        echo "=== Performance Rerun $run/$rerun_count ==="

        for var in "${unique_models[@]}"; do
            model="${var%%,*}"
            batch_size="${var%,*}"
            batch_size="${batch_size#*,}"
            regression_type="${var##*,}"
            echo "Rerunning performance test for model: $model with $model (run $run)"

            local run_log_dir="${base_log_dir}/run_${run}"
            mkdir -p "${run_log_dir}"

            # Build command for this specific model
            local rerun_cmd=(
                python benchmarks/dynamo/"${SUITE}".py
                "--performance"
                "--${REAL_DTYPE}"
                "-d" "${DEVICE}"
                "-n10"
                ${DTYPE_EXTRA}
                ${MODE_EXTRA}
                ${SHAPE_EXTRA}
                ${PARTITION_FLAGS}
                "--only" "${model}"
                "--batch_size" "${batch_size}"
                "--backend=inductor"
                "--cold-start-latency"
                "--timeout=3600"
                "--disable-cudagraphs"
                "--output=${run_log_dir}/rerun_${run}_${model//\//_}.csv"
            )

            # Execute the rerun
            echo "Running: ${rerun_cmd[@]}"
            ZE_AFFINITY_MASK="${CARD}" \
                "${rerun_cmd[@]}" 2>&1 | tee "${run_log_dir}/rerun_${run}_${model//\//_}.log"

            # Process the rerun results
            if [[ -f "${run_log_dir}/rerun_${run}_${model//\//_}.csv" ]]; then
                awk -F, -v suite="${SUITE}" -v dtype="${DTYPE}" -v mode="${MODE}" -v scenario="performance" -v regression_type="${regression_type}" '
                {
                    if ($1 != "dev") {
                        model = $2
                        batch_size = $3
                        inductor = $5
                        eager = $4 * $5
                        printf("%s,%s,%s,%s,%s,%s,-1,%s,%s,%s\n",
                            suite, dtype, mode, scenario, model, batch_size, eager, inductor, regression_type)
                    }
                }' "${run_log_dir}/rerun_${run}_${model//\//_}.csv" >> "${run_log_dir}/rerun_summary.csv"
            fi
        done
    done

    # Analyze rerun results
    analyze_rerun_results "${base_log_dir}" "${models_file}"
}

# Function to analyze rerun results
analyze_rerun_results() {
    local base_log_dir=$1
    local regression_file=$2
    local output_file="${LOG_DIR}/regression_analysis.csv"

    echo "=== Analyzing Rerun Results ==="

    # Collect all rerun data
    local all_rerun_data="${base_log_dir}/all_reruns.csv"
    echo "Suite,Dtype,Mode,Scenario,Model,BS,Eager,Inductor,Run" > "$all_rerun_data"

    for run_dir in "${base_log_dir}"/run_*; do
        if [[ -f "${run_dir}/rerun_summary.csv" ]]; then
            awk -F, -v run="${run_dir##*/run_}" '{print $0 "," run}' "${run_dir}/rerun_summary.csv" >> "$all_rerun_data"
        fi
    done

    # Calculate averages and compare with reference
    awk -F',' -v threshold="$PERF_RERUN_THRESHOLD" '
    BEGIN {
        OFS=","
        print "Model,BS,Ref_Latency,Avg,Min,Max,StdDev,Ratio,Status,Type"
    }
    NR==FNR && FNR>1 {
        # Read regression data to get reference values
        key = $5 "," $6  # Model,BS
        ref_eager[key] = $8
        ref_inductor[key] = $9
        next
    }
    NR!=FNR && FNR>1 {
        # Process rerun data
        key = $5 "," $6  # Model,BS
        if ($10 == "Eager") {
            run = $8
        }else {
            run = $9
        }
        type = $10

        if (!(key in data_count)) {
            data_count[key] = 0
            total_inductor[key] = 0
            min_inductor[key] = 999999
            max_inductor[key] = 0
            values[key] = ""  # Store values for stddev calculation
        }

        inductor_val = $8 + 0
        total_inductor[key] += inductor_val
        data_count[key]++

        if (inductor_val < min_inductor[key]) min_inductor[key] = inductor_val
        if (inductor_val > max_inductor[key]) max_inductor[key] = inductor_val

        # Store value for stddev calculation
        values[key] = values[key] (values[key] == "" ? "" : " ") inductor_val
    }
    END {
        for (key in data_count) {
            avg_inductor = total_inductor[key] / data_count[key]

            # Calculate standard deviation
            split(values[key], arr, " ")
            sum_sq = 0
            for (i in arr) {
                diff = arr[i] - avg_inductor
                sum_sq += diff * diff
            }
            stddev = sqrt(sum_sq / data_count[key])

            # Calculate ratio against reference
            ratio = 0
            if (type == "Eager") {
                ref_latency = ref_eager
            }else {
                ref_latency = ref_inductor
            }
            if (key in ref_latency && ref_latency[key] > 0) {
                ratio = avg_inductor / ref_latency[key]
            }

            # Determine status
            status = "PASS"
            if (ratio < threshold) {
                status = "REGRESSION"
            } else if (ratio >= 1.1) {
                status = "IMPROVEMENT"
            }

            print key, ref_latency[key], avg_inductor, min_inductor[key], max_inductor[key], stddev, ratio, status, type
        }
    }' "$regression_file" "$all_rerun_data" | sort > "$output_file"

    echo "Regression analysis saved to: $output_file"

    # Count results
    local total_count=0
    local pass_count=0
    local regress_count=0
    local improve_count=0

    while IFS=',' read -r model_bs ref avg min max stddev ratio status; do
        if [[ "$model_bs" == "Model,BS" ]]; then
            continue
        fi
        total_count=$((total_count + 1))
        case "$status" in
            "PASS") pass_count=$((pass_count + 1)) ;;
            "REGRESSION") regress_count=$((regress_count + 1)) ;;
            "IMPROVEMENT") improve_count=$((improve_count + 1)) ;;
        esac
    done < "$output_file"

    echo "=== Rerun Analysis Summary ==="
    echo "Total tests: $total_count"
    echo "Pass: $pass_count"
    echo "Regression: $regress_count"
    echo "Improvement: $improve_count"
    echo "=============================="
}

# Build ModelOnly extra parameter
MODEL_ONLY_EXTRA=""
if [[ -n "${MODEL_ONLY}" ]]; then
    echo "Testing model/pattern: ${MODEL_ONLY}"
    if [[ "${MODEL_ONLY}" == *"-k "* ]]; then
        MODEL_ONLY_EXTRA=" ${MODEL_ONLY} "
    else
        MODEL_ONLY_EXTRA="--only ${MODEL_ONLY}"
    fi
fi

# Build Mode extra parameter based on torch version
MODE_EXTRA=""
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "0.0.0")

# Compare versions
if printf "%s\n2.0.2\n%s" "${TORCH_VERSION}" | sort -nr | head -1 | grep -q "^2.0.2$"; then
    # Version <= 2.0.2
    MODE_EXTRA=""
else
    # Version >= 2.1.0
    MODE_EXTRA="--inference "
fi
# Override for training mode
if [[ "${MODE}" == "training" ]]; then
    echo "Testing with training mode."
    MODE_EXTRA="--training "
fi

# Build DTYPE extra parameters
REAL_DTYPE="${DTYPE}"
DTYPE_EXTRA=''
case "${DTYPE}" in
    amp_bf16)
        REAL_DTYPE="amp"
        DTYPE_EXTRA="--amp-dtype bfloat16 "
        ;;
    amp_fp16|amp)
        REAL_DTYPE="amp"
        DTYPE_EXTRA="--amp-dtype float16 "
        ;;
esac

# Build Shape extra parameter
SHAPE_EXTRA=""
if [[ "${SHAPE}" == "dynamic" ]]; then
    echo "Testing with dynamic shapes."
    SHAPE_EXTRA="--dynamic-shapes --dynamic-batch-only "
fi

# Build partition flags
PARTITION_FLAGS=""
if [[ -n "${NUM_SHARDS}" && -n "${SHARD_ID}" ]] && [[ "${NUM_SHARDS}" -gt 1 ]]; then
    if [[ "${SHARD_ID}" -ge "${NUM_SHARDS}" ]]; then
        echo "Error: Shard ID (${SHARD_ID}) must be less than total shards (${NUM_SHARDS})"
        exit 1
    fi
    PARTITION_FLAGS="--total-partitions ${NUM_SHARDS} --partition-id ${SHARD_ID} "
    echo "Running shard ${SHARD_ID} of ${NUM_SHARDS}"
fi

# Build full command
CMD=(
    python benchmarks/dynamo/"${SUITE}".py
    "--${SCENARIO}"
    "--${REAL_DTYPE}"
    "-d" "${DEVICE}"
    "-n10"
    ${DTYPE_EXTRA}
    ${MODE_EXTRA}
    ${SHAPE_EXTRA}
    ${PARTITION_FLAGS}
    ${MODEL_ONLY_EXTRA}
    "--backend=inductor"
    "--cold-start-latency"
    "--timeout=10800"
    "--disable-cudagraphs"
    "--output=${LOG_DIR}/${LOG_NAME}.csv"
)

echo "Command to execute:"
echo "${CMD[@]}"
echo

# Execute the command with environment variable
echo "Starting test execution..."
echo "Full logs will be saved to: ${LOG_DIR}/${LOG_NAME}_card${CARD}.log"
echo

# Set ulimit if needed
# ulimit -n 1048576

# Execute with tee for both file and console output
ZE_AFFINITY_MASK="${CARD}" \
    "${CMD[@]}" 2>&1 | tee -a "${LOG_DIR}/${LOG_NAME}_card${CARD}.log"

EXIT_CODE=${PIPESTATUS[0]}

echo
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "Test execution completed successfully."
else
    echo "Test execution failed with exit code: ${EXIT_CODE}"
fi

# Process results
echo "Processing results..."
RESULT_FILE="${LOG_DIR}/${LOG_NAME}.summary.csv"
echo "Suite,Dtype,Mode,Scenario,Model,BS,Accuracy,Eager,Inductor" > "${RESULT_FILE}"

if [[ -f "${LOG_DIR}/${LOG_NAME}.csv" ]]; then
    awk -F, -v suite="${SUITE}" -v dtype="${DTYPE}" -v mode="${MODE}" -v scenario="${SCENARIO}" '
    {
        if ($1 != "dev") {
            model = $2
            batch_size = $3

            if (scenario == "performance") {
                inductor = $5
                eager = $4 * $5
                result = "-1," eager "," inductor
            } else {
                result = $4 ",-1,-1"
            }

            printf("%s,%s,%s,%s,%s,%s,%s\n",
                suite, dtype, mode, scenario, model, batch_size, result)
        }
    }' "${LOG_DIR}/${LOG_NAME}.csv" | tee -a "${RESULT_FILE}"

    echo "Results saved to: ${RESULT_FILE}"
else
    echo "Warning: CSV result file not found: ${LOG_DIR}/${LOG_NAME}.csv"
fi

# Perform regression check if enabled
if [[ "${REGRESSION_CHECK}" == "true" ]]; then
    echo "=== Performing Regression Check ==="

    # Create temporary directory for regression analysis
    REGRESSION_TEMP_DIR="/tmp/regression_${SUITE}_${DTYPE}_${MODE}"
    mkdir -p "${REGRESSION_TEMP_DIR}"

    # Extract test cases from current run
    extract_test_cases "${LOG_DIR}" "${REGRESSION_TEMP_DIR}/new.summary.csv"

    # Extract test cases from reference
    extract_test_cases "${REFERENCE_DIR}" "${REGRESSION_TEMP_DIR}/reference.summary.csv"

    # Find performance regression cases
    find_perf_reg_cases \
        "${REGRESSION_TEMP_DIR}/new.summary.csv" \
        "${REGRESSION_TEMP_DIR}/reference.summary.csv" \
        "${REGRESSION_TEMP_DIR}/regression.summary.csv"

    # Display regression summary
    echo "=== Regression Summary ==="
    echo "Regression analysis saved to: ${REGRESSION_TEMP_DIR}/regression.summary.csv"

    # Count regressions
    reg_count=$(tail -n +2 "${REGRESSION_TEMP_DIR}/regression.summary.csv" | grep -c ",YES$" || true)
    total_count=$(tail -n +2 "${REGRESSION_TEMP_DIR}/regression.summary.csv" | wc -l || true)

    echo "Total comparable tests: ${total_count}"
    echo "Performance regressions detected: ${reg_count}"

    # Copy regression summary to log directory
    cp "${REGRESSION_TEMP_DIR}/regression.summary.csv" "${LOG_DIR}/regression_summary.csv"

    # Run performance rerun if regressions found and not skipping
    if [[ ${reg_count} -gt 0 ]] && [[ "${SKIP_ACCURACY}" != "true" ]]; then
        echo "=== Running Performance Rerun for Regressed Models ==="
        run_performance_tests \
            "${REGRESSION_TEMP_DIR}/regression.summary.csv" \
            "${PERF_RERUN_COUNT}"
    elif [[ "${SKIP_ACCURACY}" == "true" ]]; then
        echo "Skipping accuracy tests as requested."
    fi

    # Generate final report
    echo "=== Final Regression Report ==="
    echo "Reference Directory: ${REFERENCE_DIR}"
    echo "Current Run Directory: ${LOG_DIR}"
    echo "Regression Analysis: ${LOG_DIR}/regression_summary.csv"

    if [[ -f "${LOG_DIR}/regression_analysis.csv" ]]; then
        echo "Rerun Analysis: ${LOG_DIR}/regression_analysis.csv"
    fi

    # Exit with error if regressions found and strict mode
    if [[ ${reg_count} -gt 0 ]]; then
        echo "WARNING: Performance regressions detected!"
        # You might want to exit with non-zero code here if needed
        # exit 1
    else
        echo "SUCCESS: No performance regressions detected."
    fi
fi

echo "Script completed."
exit ${EXIT_CODE}
