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
    ["WORKSPACE"]=""            # Workspace directory
    ["LOG_DIR"]=""              # Log directory
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
if [[ -z "${WORKSPACE}" ]]; then
    WORKSPACE=$(pwd)
fi

# Configure log directory
if [[ -z "${LOG_DIR}" ]]; then
    LOG_DIR="${WORKSPACE}/inductor_log/${SUITE}/${DTYPE}"
fi

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
    local dir="$1"
    local output_file="$2"

    echo "Extracting test cases from $dir..."

    # Clear output file
    > "$output_file"

    # Find all CSV files and concatenate, removing header from all but first
    local first_file=true
    while IFS= read -r -d '' csv; do
        if [[ -f "$csv" ]]; then
            if [[ "$first_file" == true ]]; then
                # Copy entire first file including header
                cat "$csv" > "$output_file"
                first_file=false
            else
                # Append without header
                tail -n +2 "$csv" >> "$output_file"
            fi
        fi
    done < <(find "$dir" -type f -name "*.summary.csv" -print0)

    if [[ ! -s "$output_file" ]]; then
        echo "ERROR: No CSV files found in $dir or files are empty"
        return 1
    fi

    local line_count=$(wc -l < "$output_file")
    echo "Extracted $((line_count-1)) test cases from $dir"
}

# Function to find performance regression test cases
find_perf_reg_cases() {
    local new_file="$1"
    local ref_file="$2"
    local output_file="$3"

    echo "Finding matching test cases..."

    # Create temporary files for processing
    local temp_new="${output_file}.temp_new"
    local temp_ref="${output_file}.temp_ref"

    # Create key-value files with comma escaping
    awk -F',' 'NR>1 {
        gsub(/"/, "", $0)
        key = $1 "," $2 "," $3 "," $4 "," $5 "," $6
        print key "," $7 "," $8 "," $9
    }' "$new_file" | sort > "$temp_new"

    awk -F',' 'NR>1 {
        gsub(/"/, "", $0)
        key = $1 "," $2 "," $3 "," $4 "," $5 "," $6
        print key "," $7 "," $8 "," $9
    }' "$ref_file" | sort > "$temp_ref"

    # Join files and calculate ratios
    join -t',' -j1 "$temp_ref" "$temp_new" > "${output_file}.joined"

    # Process joined data
    echo "Suite,Dtype,Mode,Scenario,Model,BS,Acc,Ref_Eager,Ref_Inductor,New_Eager,New_Inductor,Eager_Ratio,Inductor_Ratio,Regression_type,Regression" > "$output_file"

    awk -F',' -v threshold="0.9" '
    {
        split($0, fields, ",")

        # Extract values
        key = fields[1]
        ref_acc = fields[7]
        ref_eager = fields[8]
        ref_inductor = fields[9]
        new_acc = fields[10]
        new_eager = fields[11]
        new_inductor = fields[12]

        # Calculate ratios (higher is better - new should be faster)
        if (new_eager > 0) {
            eager_ratio = ref_eager / new_eager
        } else {
            eager_ratio = 0
        }

        if (new_inductor > 0) {
            inductor_ratio = ref_inductor / new_inductor
        } else {
            inductor_ratio = 0
        }

        # Check for regression
        regression = "NO"
        regression_type = ""
        if (eager_ratio < threshold) {
            regression = "YES"
            regression_type = "Eager"
        }
        if (inductor_ratio < threshold) {
            regression = "YES"
            if (regression_type != "") {
                regression_type = regression_type "/"
            }
            regression_type = regression_type "Inductor"
        }

        # Split key back into components
        split(key, key_parts, ",")

        print key_parts[1] "," key_parts[2] "," key_parts[3] "," \
              key_parts[4] "," key_parts[5] "," key_parts[6] "," \
              ref_acc "," ref_eager "," ref_inductor "," \
              new_eager "," new_inductor "," \
              eager_ratio "," inductor_ratio "," \
              regression_type "," regression
    }' "${output_file}.joined" >> "$output_file"

    # Clean up temp files
    rm -f "$temp_new" "$temp_ref" "${output_file}.joined"

    echo "Generated combined comparison file: $output_file"
}

# Function to run performance tests for specific models
run_performance_tests() {
    local models_file="$1"
    local rerun_count="$2"
    local base_log_dir="${LOG_DIR}/rerun"

    mkdir -p "${base_log_dir}"

    # Parse models that need rerun
    local models_to_rerun=()
    while IFS=',' read -r suite dtype mode scenario model batch_size accuracy ref_eager ref_inductor new_eager new_inductor eager_ratio inductor_ratio regression_type regression; do
        # Skip header and non-regression cases
        [[ "$suite" == "Suite" ]] && continue
        [[ "$regression" != "YES" ]] && continue
        [[ "$scenario" != "performance" ]] && continue

        # Clean up model name (remove quotes if present)
        model=$(echo "$model" | sed 's/^"//;s/"$//')
        models_to_rerun+=("$model,$batch_size,$regression_type")
    done < "$models_file"

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

        local run_log_dir="${base_log_dir}/run_${run}"
        mkdir -p "${run_log_dir}"

        # Clear summary file for this run
        > "${run_log_dir}/rerun_summary.csv"

        for var in "${unique_models[@]}"; do
            IFS=',' read -r model batch_size regression_type <<< "$var"

            echo "Rerunning performance test for model: $model with BS=$batch_size (run $run)"

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
                "--output=${run_log_dir}/rerun_${run}_$(echo "${model}" | tr '/' '_').csv"
            )

            # Execute the rerun
            echo "Running: ${rerun_cmd[*]}"

            local log_file="${run_log_dir}/rerun_${run}_$(echo "${model}" | tr '/' '_').log"
            if ! ZE_AFFINITY_MASK="${CARD}" \
                "${rerun_cmd[@]}" 2>&1 | tee "$log_file"; then
                echo "Warning: Rerun failed for model $model"
                continue
            fi

            # Process the rerun results
            local csv_file="${run_log_dir}/rerun_${run}_$(echo "${model}" | tr '/' '_').csv"
            if [[ -f "$csv_file" ]]; then
                awk -F, -v suite="${SUITE}" -v dtype="${DTYPE}" -v mode="${MODE}" -v scenario="performance" \
                    -v model="$model" -v batch_size="$batch_size" -v regression_type="$regression_type" '
                $1 != "dev" && $2 == model && $3 == batch_size {
                    eager_time = $4
                    inductor_time = $5
                    accuracy = $6

                    # Calculate eager time (assuming column 4 is ratio, column 5 is inductor time)
                    if (eager_time ~ /[0-9]+\.[0-9]+/) {
                        eager_actual = eager_time * inductor_time
                    } else {
                        eager_actual = eager_time
                    }

                    printf("%s,%s,%s,%s,%s,%s,%s,%.4f,%.4f,%s\n",
                        suite, dtype, mode, scenario, model, batch_size, accuracy,
                        eager_actual, inductor_time, regression_type)
                }' "$csv_file" >> "${run_log_dir}/rerun_summary.csv"
            fi
        done
    done

    # Analyze rerun results
    analyze_rerun_results "${base_log_dir}" "${models_file}"
}

# Function to analyze rerun results
analyze_rerun_results() {
    local base_log_dir="$1"
    local regression_file="$2"
    local output_file="${LOG_DIR}/regression_analysis.csv"

    echo "=== Analyzing Rerun Results ==="

    # Collect all rerun data
    local all_rerun_data="${base_log_dir}/all_reruns.csv"
    echo "Suite,Dtype,Mode,Scenario,Model,BS,Accuracy,Eager,Inductor,Run,Type" > "$all_rerun_data"

    for run_dir in "${base_log_dir}"/run_*; do
        if [[ -d "$run_dir" ]] && [[ -f "${run_dir}/rerun_summary.csv" ]]; then
            local run_num="${run_dir##*/run_}"
            awk -F, -v run="$run_num" '{print $0 "," run}' "${run_dir}/rerun_summary.csv" >> "$all_rerun_data"
        fi
    done

    # Process reference data
    declare -A ref_eager
    declare -A ref_inductor
    declare -A ref_type

    while IFS=',' read -r suite dtype mode scenario model bs acc eager inductor type reg; do
        # Skip header
        [[ "$suite" == "Suite" ]] && continue
        [[ "$scenario" != "performance" ]] && continue

        key="${model},${bs}"
        ref_eager["$key"]="$eager"
        ref_inductor["$key"]="$inductor"
        ref_type["$key"]="$type"
    done < "$regression_file"

    # Calculate averages and compare with reference
    {
        echo "Model,BS,Type,Ref_Latency,Avg,Min,Max,StdDev,Ratio,Status"

        # Group by model,bs,type
        awk -F',' '
        NR>1 {
            key = $5 "," $6 "," $10  # Model,BS,Type
            inductor = $9 + 0

            # Initialize arrays
            if (!(key in count)) {
                count[key] = 0
                sum[key] = 0
                min[key] = 999999
                max[key] = 0
                values[key] = ""
            }

            count[key]++
            sum[key] += inductor
            if (inductor < min[key]) min[key] = inductor
            if (inductor > max[key]) max[key] = inductor
            values[key] = values[key] (values[key] == "" ? "" : " ") inductor
        }
        END {
            for (key in count) {
                avg = sum[key] / count[key]

                # Calculate standard deviation
                n = split(values[key], arr, " ")
                sum_sq = 0
                for (i=1; i<=n; i++) {
                    diff = arr[i] - avg
                    sum_sq += diff * diff
                }
                stddev = sqrt(sum_sq / n)

                print key "," avg "," min[key] "," max[key] "," stddev
            }
        }
        ' "$all_rerun_data" | while IFS=',' read -r model bs type avg min max stddev; do
            key="${model},${bs}"

            # Get reference value based on type
            if [[ "$type" == *"Eager"* ]] && [[ "$type" != *"Inductor"* ]]; then
                ref_latency="${ref_eager[$key]:-0}"
            elif [[ "$type" == *"Inductor"* ]]; then
                ref_latency="${ref_inductor[$key]:-0}"
            else
                ref_latency="0"
            fi

            # Calculate ratio
            ratio="0"
            if [[ "$ref_latency" != "0" ]] && [[ "$ref_latency" != "" ]]; then
                ratio=$(echo "$avg / $ref_latency" | bc -l 2>/dev/null || echo "0")
            fi

            # Determine status
            status="PASS"
            if [[ $(echo "$ratio < $PERF_RERUN_THRESHOLD" | bc -l 2>/dev/null) -eq 1 ]]; then
                status="REGRESSION"
            elif [[ $(echo "$ratio > 1.1" | bc -l 2>/dev/null) -eq 1 ]]; then
                status="IMPROVEMENT"
            fi

            echo "${model},${bs},${type},${ref_latency},${avg},${min},${max},${stddev},${ratio},${status}"
        done
    } > "$output_file"

    echo "Regression analysis saved to: $output_file"

    # Count results
    local total_count=0
    local pass_count=0
    local regress_count=0
    local improve_count=0

    while IFS=',' read -r model bs type ref avg min max stddev ratio status; do
        [[ "$model" == "Model" ]] && continue
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

# Build Mode extra parameter based on torch version
MODE_EXTRA=""
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "0.0.0")

# Compare versions
if printf "%s\n2.0.2\n%s" "${TORCH_VERSION}" "${TORCH_VERSION}" | sort -V | head -1 | grep -q "2.0.2"; then
    # Version <= 2.0.2
    MODE_EXTRA=""
else
    # Version >= 2.1.0
    MODE_EXTRA="--inference"
fi
# Override for training mode
if [[ "${MODE}" == "training" ]]; then
    echo "Testing with training mode."
    MODE_EXTRA="--training"
fi

# Build DTYPE extra parameters
REAL_DTYPE="${DTYPE}"
DTYPE_EXTRA=''
case "${DTYPE}" in
    amp_bf16)
        REAL_DTYPE="amp"
        DTYPE_EXTRA="--amp-dtype bfloat16"
        ;;
    amp_fp16|amp)
        REAL_DTYPE="amp"
        DTYPE_EXTRA="--amp-dtype float16"
        ;;
esac

# Build Shape extra parameter
SHAPE_EXTRA=""
if [[ "${SHAPE}" == "dynamic" ]]; then
    echo "Testing with dynamic shapes."
    SHAPE_EXTRA="--dynamic-shapes --dynamic-batch-only"
fi

# Build partition flags
PARTITION_FLAGS=""
if [[ -n "${NUM_SHARDS}" && -n "${SHARD_ID}" ]] && [[ "${NUM_SHARDS}" -gt 1 ]]; then
    if [[ "${SHARD_ID}" -ge "${NUM_SHARDS}" ]]; then
        echo "Error: Shard ID (${SHARD_ID}) must be less than total shards (${NUM_SHARDS})"
        exit 1
    fi
    PARTITION_FLAGS="--total-partitions ${NUM_SHARDS} --partition-id ${SHARD_ID}"
    echo "Running shard ${SHARD_ID} of ${NUM_SHARDS}"
fi

# Build ModelOnly extra parameter
MODEL_ONLY_EXTRA=()
if [[ -n "${MODEL_ONLY}" ]]; then
    echo "Testing model/pattern: ${MODEL_ONLY}"
    if [[ "${MODEL_ONLY}" == *"-k "* ]]; then
        # Split the -k pattern
        IFS=' ' read -ra PATTERN_PARTS <<< "${MODEL_ONLY}"
        MODEL_ONLY_EXTRA=("${PATTERN_PARTS[@]}")
    else
        MODEL_ONLY_EXTRA=("--only" "${MODEL_ONLY}")
    fi
fi

# Build full command
CMD=(
    python benchmarks/dynamo/"${SUITE}".py
    "--${SCENARIO}"
    "--${REAL_DTYPE}"
    "-d" "${DEVICE}"
    "-n10"
)

# Add optional parameters if they're not empty
[[ -n "${DTYPE_EXTRA}" ]] && CMD+=("${DTYPE_EXTRA}")
[[ -n "${MODE_EXTRA}" ]] && CMD+=("${MODE_EXTRA}")
[[ -n "${SHAPE_EXTRA}" ]] && CMD+=(${SHAPE_EXTRA})
[[ -n "${PARTITION_FLAGS}" ]] && CMD+=(${PARTITION_FLAGS})
[[ ${#MODEL_ONLY_EXTRA[@]} -gt 0 ]] && CMD+=("${MODEL_ONLY_EXTRA[@]}")

# Add fixed parameters
CMD+=(
    "--backend=inductor"
    "--cold-start-latency"
    "--timeout=10800"
    "--disable-cudagraphs"
    "--output=${LOG_DIR}/${LOG_NAME}.csv"
)

echo "Command to execute:"
printf "'%s' " "${CMD[@]}"
echo
echo

# Execute the command with environment variable
echo "Starting test execution..."
echo "Full logs will be saved to: ${LOG_DIR}/${LOG_NAME}_card${CARD}.log"
echo

# Set ulimit if needed
ulimit -n 1048576 2>/dev/null || true

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
    BEGIN {
        OFS=","
    }
    $1 != "dev" {
        model = $2
        batch_size = $3

        if (scenario == "performance") {
            # Column 4 is eager/inductor ratio, column 5 is inductor time
            inductor_time = $5 + 0
            ratio = $4 + 0

            if (ratio > 0) {
                eager_time = inductor_time * ratio
            } else {
                eager_time = inductor_time  # Fallback
            }

            accuracy = "-1"
            eager = sprintf("%.4f", eager_time)
            inductor = sprintf("%.4f", inductor_time)
        } else {
            # accuracy mode
            accuracy = $4
            eager = "-1"
            inductor = "-1"
        }

        # Clean model name (remove quotes)
        gsub(/"/, "", model)

        print suite, dtype, mode, scenario, model, batch_size, accuracy, eager, inductor
    }' "${LOG_DIR}/${LOG_NAME}.csv" | tee -a "${RESULT_FILE}"

    echo "Results saved to: ${RESULT_FILE}"
else
    echo "Warning: CSV result file not found: ${LOG_DIR}/${LOG_NAME}.csv"
fi

# Perform regression check if enabled
if [[ "${REGRESSION_CHECK}" == "true" ]]; then
    echo "=== Performing Regression Check ==="

    # Create temporary directory for regression analysis
    REGRESSION_TEMP_DIR="/tmp/regression_${SUITE}_${DTYPE}_${MODE}_${SCENARIO}_$$"
    mkdir -p "${REGRESSION_TEMP_DIR}"

    # Extract test cases from current run
    if ! extract_test_cases "${LOG_DIR}" "${REGRESSION_TEMP_DIR}/new.summary.csv"; then
        echo "Error: Failed to extract test cases from current run"
        exit 1
    fi

    # Extract test cases from reference
    if ! extract_test_cases "${REFERENCE_DIR}" "${REGRESSION_TEMP_DIR}/reference.summary.csv"; then
        echo "Error: Failed to extract test cases from reference directory"
        exit 1
    fi

    # Find performance regression cases
    find_perf_reg_cases \
        "${REGRESSION_TEMP_DIR}/new.summary.csv" \
        "${REGRESSION_TEMP_DIR}/reference.summary.csv" \
        "${REGRESSION_TEMP_DIR}/regression.summary.csv"

    # Display regression summary
    echo "=== Regression Summary ==="
    echo "Regression analysis saved to: ${REGRESSION_TEMP_DIR}/regression.summary.csv"

    # Count regressions
    local reg_count=0
    local total_count=0

    if [[ -f "${REGRESSION_TEMP_DIR}/regression.summary.csv" ]]; then
        reg_count=$(tail -n +2 "${REGRESSION_TEMP_DIR}/regression.summary.csv" | awk -F',' '{if ($15 == "YES") count++} END {print count+0}')
        total_count=$(tail -n +2 "${REGRESSION_TEMP_DIR}/regression.summary.csv" | wc -l)
    fi

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

    # Clean up temp directory
    rm -rf "${REGRESSION_TEMP_DIR}"

    # Exit with error if regressions found and strict mode
    if [[ ${reg_count} -gt 0 ]]; then
        echo "WARNING: Performance regressions detected!"
        # You might want to exit with non-zero code here if needed
        # exit 1
    else
        echo "SUCCESS: No performance regressions detected."
    fi
fi

echo "Script completed with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
