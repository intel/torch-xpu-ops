#!/bin/bash
# Enhanced script for CPU/XPU/CUDA device Dynamo benchmark tests with regression checking

set -euo pipefail

# Default configuration
declare -A DEFAULTS=(
    ["SUITE"]="huggingface"     # huggingface / torchbench / timm_models
    ["DTYPE"]="float32"         # float32 / float16 / amp (amp_bf16) / amp_fp16
    ["MODE"]="inference"        # inference / training
    ["SCENARIO"]="accuracy"     # accuracy / performance
    ["DEVICE"]="xpu"            # xpu / cuda / cpu
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
    ["BASELINE_TYPE"]="current" # current / nightly / test / release / custom
    ["BASELINE_DIR"]=""         # Custom baseline wheel directory
    ["TORCH_VERSION"]=""        # Specific torch version for baseline
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
  -e, --device DEVICE        Device: xpu, cuda, cpu (default: ${DEFAULTS[DEVICE]})
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
  --baseline-type TYPE       Baseline type: current, nightly, test, release, custom (default: ${DEFAULTS[BASELINE_TYPE]})
  --baseline-dir DIR         Custom baseline wheel directory
  --torch-version VER        Specific torch version for baseline
  -h, --help                 Show this help message

Examples:
  $(basename "$0") --suite huggingface --dtype amp_bf16 --mode inference
  $(basename "$0") -s torchbench -d float16 -m training -c performance
  $(basename "$0") --model-only "GoogleFnet" --device cuda --card 1
  $(basename "$0") --num-shards 4 --shard-id 2 --model-only "-k T5"
  $(basename "$0") --regression-check --reference-dir /path/to/reference
  $(basename "$0") --regression-check --perf-rerun-threshold 0.8 --perf-rerun-count 5
  $(basename "$0") --regression-check --baseline-type nightly --torch-version 2.4.0

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
            --baseline-type)
                BASELINE_TYPE="$2"
                shift 2
                ;;
            --baseline-dir)
                BASELINE_DIR="$2"
                shift 2
                ;;
            --torch-version)
                TORCH_VERSION="$2"
                shift 2
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

# Get current torch version if not specified
if [[ -z "${TORCH_VERSION}" ]]; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "2.0.0")
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
    echo "Baseline Type: ${BASELINE_TYPE}"
    echo "Torch Version: ${TORCH_VERSION}"

    if [[ -n "${NUM_SHARDS}" && -n "${SHARD_ID}" ]]; then
        echo "Shards:         ${SHARD_ID}/${NUM_SHARDS}"
    fi

    if [[ -n "${MODEL_ONLY}" ]]; then
        echo "Model Filter:   ${MODEL_ONLY}"
    fi

    if [[ -n "${REFERENCE_DIR}" ]]; then
        echo "Reference Dir:  ${REFERENCE_DIR}"
    fi

    if [[ -n "${BASELINE_DIR}" ]]; then
        echo "Baseline Dir:   ${BASELINE_DIR}"
    fi
    echo "========================================"
}

print_configuration

# Function to setup baseline environment
setup_baseline_environment() {
    local baseline_type="$1"
    local torch_version="$2"
    local baseline_dir="$3"

    echo "=== Setting up Baseline Environment ==="
    echo "Type: $baseline_type"
    echo "Torch Version: $torch_version"

    # Save current environment info
    python3 -m pip freeze > "${LOG_DIR}/current_environment.txt" 2>/dev/null || true

    # Uninstall current torch packages
    echo "Uninstalling current torch packages..."
    pip uninstall -y torch torchvision torchaudio triton triton-xpu 2>/dev/null || true

    # Install baseline based on type
    case "$baseline_type" in
        "current")
            echo "Using current environment (no change)"
            ;;
        "release")
            echo "Installing release wheel for $torch_version..."
            if [[ "${DEVICE}" == "xpu" ]]; then
                pip install torch==${torch_version} torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
            elif [[ "${DEVICE}" == "cuda" ]]; then
                pip install torch==${torch_version} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            else
                pip install torch==${torch_version} torchvision torchaudio
            fi
            ;;
        "test")
            echo "Installing test wheel for $torch_version..."
            if [[ "${DEVICE}" == "xpu" ]]; then
                pip install torch==${torch_version} torchvision torchaudio --index-url https://download.pytorch.org/whl/test/xpu
            elif [[ "${DEVICE}" == "cuda" ]]; then
                pip install torch==${torch_version} torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu118
            else
                pip install torch==${torch_version} torchvision torchaudio --index-url https://download.pytorch.org/whl/test
            fi
            ;;
        "nightly")
            echo "Installing nightly wheel for $torch_version..."
            if [[ "${DEVICE}" == "xpu" ]]; then
                pip install torch==${torch_version} torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
            elif [[ "${DEVICE}" == "cuda" ]]; then
                pip install torch==${torch_version} torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
            else
                pip install torch==${torch_version} torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly
            fi
            ;;
        "custom")
            if [[ -z "$baseline_dir" ]] || [[ ! -d "$baseline_dir" ]]; then
                echo "ERROR: Custom baseline directory not specified or doesn't exist: $baseline_dir"
                return 1
            fi
            echo "Installing from custom directory: $baseline_dir"

            # Find the latest wheel directory
            local latest_dir=""
            if [[ -d "${baseline_dir}/Torch-XPU-Wheel-" ]]; then
                latest_dir="$(find "${baseline_dir}" -type d -name "Torch-XPU-Wheel-*" 2>/dev/null | sort -V | tail -n 1)"
            elif [[ -d "${baseline_dir}/Torch-CUDA-Wheel-" ]]; then
                latest_dir="$(find "${baseline_dir}" -type d -name "Torch-CUDA-Wheel-*" 2>/dev/null | sort -V | tail -n 1)"
            fi

            if [[ -z "$latest_dir" ]] || [[ ! -d "$latest_dir" ]]; then
                echo "WARNING: Could not find wheel directory in $baseline_dir"
                echo "Looking for any .whl files..."
                local whl_files=$(find "$baseline_dir" -name "*.whl" 2>/dev/null | head -5)
                if [[ -n "$whl_files" ]]; then
                    echo "Found wheel files:"
                    echo "$whl_files"
                    pip install --force-reinstall $whl_files
                else
                    echo "ERROR: No wheel files found in $baseline_dir"
                    return 1
                fi
            else
                echo "Found wheel directory: $latest_dir"
                pip install --force-reinstall $(find "${latest_dir}" -name "*.whl" 2>/dev/null)
            fi
            ;;
        *)
            echo "ERROR: Unknown baseline type: $baseline_type"
            return 1
            ;;
    esac

    # Verify installation
    echo "=== Verification ==="
    python3 -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'Device available: {torch.xpu.is_available() if hasattr(torch, \"xpu\") else torch.cuda.is_available()}')" 2>/dev/null || echo "Failed to import torch"

    # Save baseline environment info
    python3 -m pip freeze > "${LOG_DIR}/baseline_environment.txt" 2>/dev/null || true

    echo "Baseline environment setup completed"
    return 0
}

# Function to restore original environment
restore_original_environment() {
    echo "=== Restoring Original Environment ==="

    # Note: In production, you might want to implement a more sophisticated
    # environment restoration mechanism using virtual environments

    echo "Original environment restored (note: manual pip install may be needed)"
}

# Function to compare target vs baseline results
compare_target_baseline() {
    local target_dir="$1"
    local baseline_dir="$2"
    local output_file="$3"

    echo "=== Comparing Target vs Baseline Results ==="

    # Find summary files
    local target_summary=$(find "$target_dir" -name "*summary.csv" | head -1)
    local baseline_summary=$(find "$baseline_dir" -name "*summary.csv" | head -1)

    if [[ -z "$target_summary" ]] || [[ -z "$baseline_summary" ]]; then
        echo "ERROR: Could not find summary files for comparison"
        return 1
    fi

    echo "Target summary: $target_summary"
    echo "Baseline summary: $baseline_summary"

    # Create comparison CSV
    {
        echo "Model,BS,Target_Avg,Target_Min,Target_Max,Baseline_Avg,Baseline_Min,Baseline_Max,Ratio,Status"

        # Parse and compare results
        awk -F',' '
        BEGIN {
            OFS=","
            # Read target data
            while ((getline < "'"$target_summary"'") > 0) {
                if (NR>1 && $1 != "Suite") {
                    key = $5 "," $6
                    target_avg[key] = $8
                    target_min[key] = $9
                    target_max[key] = $10
                }
            }
            close("'"$target_summary"'")

            # Read baseline data
            while ((getline < "'"$baseline_summary"'") > 0) {
                if (NR>1 && $1 != "Suite") {
                    key = $5 "," $6
                    baseline_avg[key] = $8
                    baseline_min[key] = $9
                    baseline_max[key] = $10
                }
            }
            close("'"$baseline_summary"'")

            # Compare
            for (key in target_avg) {
                if (key in baseline_avg) {
                    t_avg = target_avg[key] + 0
                    b_avg = baseline_avg[key] + 0

                    t_min = target_min[key] + 0
                    t_max = target_max[key] + 0
                    b_min = baseline_min[key] + 0
                    b_max = baseline_max[key] + 0

                    ratio = 0
                    status = "N/A"

                    if (b_avg > 0 && t_avg > 0) {
                        ratio = t_avg / b_avg
                        if (ratio > 1.1) {
                            status = "REGRESSION"
                        } else if (ratio < 0.9) {
                            status = "IMPROVEMENT"
                        } else {
                            status = "PASS"
                        }
                    }

                    split(key, parts, ",")
                    model = parts[1]
                    bs = parts[2]

                    print model, bs, t_avg, t_min, t_max, b_avg, b_min, b_max, ratio, status
                }
            }
        }
        '
    } > "$output_file"

    echo "Comparison saved to: $output_file"

    # Generate summary
    local total=0
    local pass=0
    local regress=0
    local improve=0

    while IFS=',' read -r model bs t_avg t_min t_max b_avg b_min b_max ratio status; do
        [[ "$model" == "Model" ]] && continue
        total=$((total + 1))
        case "$status" in
            "PASS") pass=$((pass + 1)) ;;
            "REGRESSION") regress=$((regress + 1)) ;;
            "IMPROVEMENT") improve=$((improve + 1)) ;;
        esac
    done < "$output_file"

    echo "=== Comparison Summary ==="
    echo "Total comparisons: $total"
    echo "Pass: $pass"
    echo "Regression (target slower): $regress"
    echo "Improvement (target faster): $improve"
}

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

    # Find all CSV files and process them
    local found_csvs=0
    while IFS= read -r -d '' csv; do
        if [[ -f "$csv" ]]; then
            found_csvs=1
            echo "Processing: $csv"

            # Extract data (skip header if output file already has content)
            if [[ ! -s "$output_file" ]]; then
                # First file, include header
                cat "$csv" >> "$output_file"
            else
                # Subsequent files, skip header
                tail -n +2 "$csv" >> "$output_file"
            fi
        fi
    done < <(find "$dir" -type f -name "*.summary.csv" -print0)

    if [[ $found_csvs -eq 0 ]]; then
        echo "ERROR: No CSV files found in $dir"
        return 1
    fi

    if [[ ! -s "$output_file" ]]; then
        echo "ERROR: CSV files found but no data extracted from $dir"
        return 1
    fi

    local line_count=$(wc -l < "$output_file")
    echo "Extracted $((line_count-1)) test cases from $dir"
    return 0
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

    # Clean and prepare new results
    awk -F',' '
    BEGIN {
        OFS=","
    }
    NR==1 {
        next  # Skip header
    }
    {
        # Clean fields
        gsub(/"/, "", $0)
        gsub(/^[ \t]+|[ \t]+$/, "", $0)
        print $0
    }' "$new_file" | sort > "$temp_new"

    # Clean and prepare reference results
    awk -F',' '
    BEGIN {
        OFS=","
    }
    NR==1 {
        next  # Skip header
    }
    {
        # Clean fields
        gsub(/"/, "", $0)
        gsub(/^[ \t]+|[ \t]+$/, "", $0)
        print $0
    }' "$ref_file" | sort > "$temp_ref"

    # Join files and calculate ratios
    {
        # Write header
        echo "Suite,Dtype,Mode,Scenario,Model,BS,Acc,Ref_Eager,Ref_Inductor,New_Eager,New_Inductor,Eager_Ratio,Inductor_Ratio,Regression_type,Regression"

        # Join and process
        join -t',' -j1 "$temp_ref" "$temp_new" | awk -F',' -v threshold="$PERF_RERUN_THRESHOLD" '
        BEGIN {
            OFS=","
        }
        {
            # Extract key parts
            suite = $1
            dtype = $2
            mode = $3
            scenario = $4
            model = $5
            bs = $6

            # Extract reference values
            ref_acc = $7
            ref_eager = $8
            ref_inductor = $9

            # Extract new values
            new_acc = $(NF - 2)
            new_eager = $(NF - 1)
            new_inductor = $(NF - 0)

            # Convert to numbers
            ref_eager_num = ref_eager + 0
            ref_inductor_num = ref_inductor + 0
            new_eager_num = new_eager + 0
            new_inductor_num = new_inductor + 0

            # Calculate ratios (higher is better - new should be faster)
            eager_ratio = 0
            inductor_ratio = 0

            if (new_eager_num > 0 && ref_eager_num > 0) {
                eager_ratio = ref_eager_num / new_eager_num
            }

            if (new_inductor_num > 0 && ref_inductor_num > 0) {
                inductor_ratio = ref_inductor_num / new_inductor_num
            }

            # Check for regression
            regression = "NO"
            regression_type = ""

            if (scenario == "performance") {
                if (eager_ratio < threshold && eager_ratio > 0) {
                    regression = "YES"
                    regression_type = "Eager"
                }
                if (inductor_ratio < threshold && inductor_ratio > 0) {
                    regression = "YES"
                    regression_type = "Inductor"
                }
            }

            print suite, dtype, mode, scenario, model, bs, ref_acc, \
                  ref_eager_num, ref_inductor_num, \
                  new_eager_num, new_inductor_num, \
                  eager_ratio, inductor_ratio, \
                  regression_type, regression
        }'
    } > "$output_file"

    # Clean up temp files
    rm -f "$temp_new" "$temp_ref"

    if [[ -s "$output_file" ]]; then
        echo "Generated combined comparison file: $output_file"
        echo "Total records: $(($(wc -l < "$output_file") - 1))"
    else
        echo "ERROR: Failed to generate comparison file"
        return 1
    fi
}

# Function to run performance tests for specific models
run_performance_tests() {
    local models_file="$1"
    local rerun_count="$2"
    local rerun_type="$3"
    local base_log_dir="${LOG_DIR}/${rerun_type}_rerun"

    mkdir -p "${base_log_dir}"

    # Parse models that need rerun
    local models_to_rerun=()
    while IFS=',' read -r suite dtype mode scenario model batch_size accuracy ref_eager ref_inductor new_eager new_inductor eager_ratio inductor_ratio regression_type regression; do
        # Skip header
        [[ "$suite" == "Suite" ]] && continue
        # Skip non-performance or non-regression cases
        [[ "$scenario" != "performance" ]] && continue
        [[ "$regression" != "YES" ]] && continue

        # Clean up model name
        model=$(echo "$model" | sed 's/^"//;s/"$//;s/^[[:space:]]*//;s/[[:space:]]*$//')
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
    printf '  - %s\n' "${unique_models[@]}"

    # Run performance tests for each model multiple times
    for ((run=1; run<=rerun_count; run++)); do
        echo "=== ${rerun_type^^} Performance Rerun $run/$rerun_count ==="

        local run_log_dir="${base_log_dir}/run_${run}"
        mkdir -p "${run_log_dir}"

        # Clear summary file for this run
        > "${run_log_dir}/rerun_summary.csv"

        for var in "${unique_models[@]}"; do
            IFS=',' read -r model batch_size regression_type <<< "$var"

            echo "Rerunning performance test for model: '$model' with BS=$batch_size (run $run)"

            # Build command for this specific model
            local rerun_cmd=()
            rerun_cmd+=("python" "benchmarks/dynamo/${SUITE}.py")
            rerun_cmd+=("--${SCENARIO}")
            rerun_cmd+=("--${REAL_DTYPE}")
            rerun_cmd+=("-d" "${DEVICE}")
            rerun_cmd+=("-n10")
            [[ -n "${DTYPE_EXTRA}" ]] && rerun_cmd+=("${DTYPE_EXTRA}")
            [[ -n "${MODE_EXTRA}" ]] && rerun_cmd+=("${MODE_EXTRA}")
            [[ -n "${SHAPE_EXTRA}" ]] && rerun_cmd+=(${SHAPE_EXTRA})
            [[ -n "${PARTITION_FLAGS}" ]] && rerun_cmd+=(${PARTITION_FLAGS})
            rerun_cmd+=("--only" "${model}")
            rerun_cmd+=("--batch_size" "${batch_size}")
            rerun_cmd+=("--backend=inductor")
            rerun_cmd+=("--cold-start-latency")
            rerun_cmd+=("--timeout=3600")
            rerun_cmd+=("--disable-cudagraphs")

            local safe_model_name=$(echo "${model}" | tr '/' '_' | tr ' ' '_' | tr -cd '[:alnum:]_')
            local output_csv="${run_log_dir}/rerun_${run}_${safe_model_name}.csv"
            rerun_cmd+=("--output=${output_csv}")

            # Execute the rerun
            echo "Running: ${rerun_cmd[*]}"

            local log_file="${run_log_dir}/rerun_${run}_${safe_model_name}.log"
            echo "Command: ${rerun_cmd[*]}" > "$log_file"
            echo "=========================================" >> "$log_file"

            if ! ZE_AFFINITY_MASK="${CARD}" \
                "${rerun_cmd[@]}" 2>&1 | tee -a "$log_file"; then
                echo "Warning: Rerun failed for model '$model'"
                continue
            fi

            # Process the rerun results
            if [[ -f "$output_csv" ]]; then
                awk -F, -v suite="${SUITE}" -v dtype="${DTYPE}" -v mode="${MODE}" -v scenario="${SCENARIO}" \
                    -v model="$model" -v batch_size="$batch_size" -v regression_type="$regression_type" '
                BEGIN {
                    OFS=","
                    found = 0
                }
                $1 != "dev" && $2 == model && $3 == batch_size {
                    # Parse the CSV columns
                    # Assuming columns: device, model, batch_size, accuracy, eager_time, inductor_time
                    ratio = $4 + 0
                    inductor_time = $5 + 0
                    eager_time = inductor_time * ratio
                    accuracy = "-1"

                    printf("%s,%s,%s,%s,%s,%s,%s,%.4f,%.4f,%s\n",
                        suite, dtype, mode, scenario, model, batch_size, accuracy,
                        eager_time, inductor_time, regression_type)
                    found = 1
                }
                END {
                    if (!found) {
                        printf("WARNING: No data found for model %s, batch_size %s\n", model, batch_size) > "/dev/stderr"
                    }
                }' "$output_csv" >> "${run_log_dir}/rerun_summary.csv"
            else
                echo "Warning: Output CSV not found: $output_csv"
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

    # Check if we have any rerun data
    if ! ls "${base_log_dir}"/run_*/rerun_summary.csv >/dev/null 2>&1; then
        echo "No rerun data found to analyze"
        return 1
    fi

    # Collect all rerun data
    local all_rerun_data="${base_log_dir}/all_reruns.csv"
    {
        echo "Suite,Dtype,Mode,Scenario,Model,BS,Accuracy,Eager,Inductor,Run,Type"
        for run_dir in "${base_log_dir}"/run_*/; do
            if [[ -f "${run_dir}/rerun_summary.csv" ]]; then
                local run_num="${run_dir%/}"
                run_num="${run_num##*/run_}"
                # Append run number to each line
                awk -F, -v run="$run_num" '{print $0 "," run}' "${run_dir}/rerun_summary.csv"
            fi
        done
    } > "$all_rerun_data"

    if [[ ! -s "$all_rerun_data" ]] || [[ $(wc -l < "$all_rerun_data") -le 1 ]]; then
        echo "No valid rerun data collected"
        return 1
    fi

    # Process reference data from regression file
    declare -A ref_eager
    declare -A ref_inductor
    declare -A ref_type

    while IFS=',' read -r suite dtype mode scenario model bs acc eager inductor ratio1 ratio2 type reg; do
        # Skip header
        [[ "$suite" == "Suite" ]] && continue
        [[ "$scenario" != "performance" ]] && continue

        # Clean model name
        model=$(echo "$model" | sed 's/^"//;s/"$//;s/^[[:space:]]*//;s/[[:space:]]*$//')
        key="${model},${bs}"

        ref_eager["$key"]="$eager"
        ref_inductor["$key"]="$inductor"
        ref_type["$key"]="$type"
    done < "$regression_file"

    # Calculate averages and compare with reference
    {
        echo "Model,BS,Type,Ref_Latency,Avg,Min,Max,StdDev,Ratio,Status"

        # Process rerun data
        awk -F',' '
        BEGIN {
            OFS=","
        }
        NR>1 {
            model = $5
            bs = $6
            type = $10
            run = $11
            inductor_latency = $9 + 0

            # Skip if no inductor latency
            if (inductor_latency <= 0) next

            # Create composite key
            key = model "," bs "," type

            # Store values for statistics
            count[key]++
            sum[key] += inductor_latency
            sum_sq[key] += inductor_latency * inductor_latency

            if (!(key in min) || inductor_latency < min[key]) {
                min[key] = inductor_latency
            }
            if (!(key in max) || inductor_latency > max[key]) {
                max[key] = inductor_latency
            }

            # Store all values for stddev calculation
            if (values[key] == "") {
                values[key] = inductor_latency
            } else {
                values[key] = values[key] " " inductor_latency
            }
        }
        END {
            for (key in count) {
                n = count[key]
                avg = sum[key] / n

                # Calculate standard deviation
                split(values[key], arr, " ")
                sum_sq_diff = 0
                for (i=1; i<=n; i++) {
                    diff = arr[i] - avg
                    sum_sq_diff += diff * diff
                }
                stddev = sqrt(sum_sq_diff / n)

                print key, avg, min[key], max[key], stddev
            }
        }
        ' "$all_rerun_data" | while IFS=',' read -r model bs type avg min max stddev; do
            key="${model},${bs}"

            # Get reference value based on type
            ref_latency="0"
            if [[ "$type" == *"Eager"* ]] && [[ "$type" != *"Inductor"* ]]; then
                ref_latency="${ref_eager[$key]:-0}"
            elif [[ "$type" == *"Inductor"* ]]; then
                ref_latency="${ref_inductor[$key]:-0}"
            else
                # Try to get any reference value
                ref_latency="${ref_inductor[$key]:-${ref_eager[$key]:-0}}"
            fi

            # Calculate ratio (lower is better - new should be faster)
            ratio="0"
            if [[ "$ref_latency" != "0" ]] && [[ "$ref_latency" != "" ]] && [[ "$avg" != "0" ]]; then
                # Use awk for floating point calculation
                ratio=$(echo "$ref_latency / $avg" | awk '{printf("%.4f", $1 / $NF)}' 2>/dev/null || echo "0")
            fi

            # Determine status
            status="PASS"

            if [[ "$ratio" != "0" ]] && [[ "$ratio" != "" ]]; then
                # Compare using awk for accurate floating point comparison
                if (( $(echo "$ratio < $PERF_RERUN_THRESHOLD" | awk '{print ($1 < $2)}') )); then
                    status="REGRESSION"
                elif (( $(echo "$ratio > 1.05" | awk '{print ($1 > $2)}') )); then
                    status="IMPROVEMENT"
                fi
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
TORCH_VERSION_DETECTED=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "0.0.0")

echo "Detected PyTorch version: ${TORCH_VERSION_DETECTED}"

# Compare versions
if printf "%s\n%s\n2.0.2" "${TORCH_VERSION_DETECTED}" "${TORCH_VERSION_DETECTED}" | sort -nr | head -1 | grep -q "2.0.2"; then
    # Version >= 2.0.2 (newer API)
    if [[ "${MODE}" == "inference" ]]; then
        MODE_EXTRA="--inference"
    fi
else
    # Version < 2.0.2 (older API)
    MODE_EXTRA=""
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
        # Handle -k pattern
        MODEL_ONLY_EXTRA=("${MODEL_ONLY}")
    else
        # Handle --only pattern
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
[[ -n "${SHAPE_EXTRA}" ]] && IFS=' ' read -ra SHAPE_PARTS <<< "${SHAPE_EXTRA}" && CMD+=("${SHAPE_PARTS[@]}")
[[ -n "${PARTITION_FLAGS}" ]] && IFS=' ' read -ra PARTITION_PARTS <<< "${PARTITION_FLAGS}" && CMD+=("${PARTITION_PARTS[@]}")
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
echo "Executing command..." | tee -a "${LOG_DIR}/${LOG_NAME}_card${CARD}.log"
echo "Command: ${CMD[*]}" | tee -a "${LOG_DIR}/${LOG_NAME}_card${CARD}.log"
echo "=========================================" | tee -a "${LOG_DIR}/${LOG_NAME}_card${CARD}.log"

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
    echo "Processing CSV file: ${LOG_DIR}/${LOG_NAME}.csv"

    awk -F, -v suite="${SUITE}" -v dtype="${DTYPE}" -v mode="${MODE}" -v scenario="${SCENARIO}" '
    BEGIN {
        OFS=","
    }
    $1 != "dev" && NF >= 5 {
        model = $2
        batch_size = $3

        # Clean model name
        gsub(/"/, "", model)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", model)

        if (scenario == "performance") {
            # For performance tests
            ratio = $4 + 0
            inductor_time = $5 + 0
            eager_time = inductor_time * ratio

            # Handle different CSV formats
            if (eager_time > 0 && inductor_time > 0) {
                # Column 4 is eager time, column 5 is inductor time
                accuracy = "-1"
                eager = sprintf("%.4f", eager_time)
                inductor = sprintf("%.4f", inductor_time)
            } else if (NF >= 6) {
                # Alternative format: column 4 is ratio, column 5 is inductor time
                ratio = $4 + 0
                inductor_time = $5 + 0
                accuracy = $6

                if (ratio > 0 && inductor_time > 0) {
                    eager_time = inductor_time * ratio
                    eager = sprintf("%.4f", eager_time)
                    inductor = sprintf("%.4f", inductor_time)
                } else {
                    eager = "-1"
                    inductor = "-1"
                    accuracy = "-1"
                }
            } else {
                eager = "-1"
                inductor = "-1"
                accuracy = "-1"
            }
        } else {
            # For accuracy tests
            accuracy = $4
            eager = "-1"
            inductor = "-1"
        }

        print suite, dtype, mode, scenario, model, batch_size, accuracy, eager, inductor
    }' "${LOG_DIR}/${LOG_NAME}.csv" | tee -a "${RESULT_FILE}"

    echo "Results saved to: ${RESULT_FILE}"
    echo "Total results: $(($(wc -l < "${RESULT_FILE}") - 1))"
else
    echo "Warning: CSV result file not found: ${LOG_DIR}/${LOG_NAME}.csv"
fi

# Perform regression check if enabled
if [[ "${REGRESSION_CHECK}" == "true" ]]; then
    echo "=== Performing Regression Check ==="

    # Create temporary directory for regression analysis
    REGRESSION_TEMP_DIR=$(mktemp -d -t "regression_${SUITE}_${DTYPE}_${MODE}_XXXXXX")
    echo "Using temporary directory: ${REGRESSION_TEMP_DIR}"

    # Extract test cases from current run
    if ! extract_test_cases "${LOG_DIR}" "${REGRESSION_TEMP_DIR}/new.summary.csv"; then
        echo "Error: Failed to extract test cases from current run"
        rm -rf "${REGRESSION_TEMP_DIR}"
        exit 1
    fi

    # Extract test cases from reference
    if ! extract_test_cases "${REFERENCE_DIR}" "${REGRESSION_TEMP_DIR}/reference.summary.csv"; then
        echo "Error: Failed to extract test cases from reference directory"
        rm -rf "${REGRESSION_TEMP_DIR}"
        exit 1
    fi

    # Find performance regression cases
    if ! find_perf_reg_cases \
        "${REGRESSION_TEMP_DIR}/new.summary.csv" \
        "${REGRESSION_TEMP_DIR}/reference.summary.csv" \
        "${REGRESSION_TEMP_DIR}/regression.summary.csv"; then
        echo "Error: Failed to perform regression analysis"
        rm -rf "${REGRESSION_TEMP_DIR}"
        exit 1
    fi

    # Display regression summary
    echo "=== Regression Summary ==="
    echo "Regression analysis saved to: ${REGRESSION_TEMP_DIR}/regression.summary.csv"

    # Count regressions
    reg_count=0
    total_count=0

    if [[ -f "${REGRESSION_TEMP_DIR}/regression.summary.csv" ]]; then
        reg_count=$(awk -F',' 'NR>1 && $15 == "YES" {count++} END {print count+0}' "${REGRESSION_TEMP_DIR}/regression.summary.csv")
        total_count=$(awk 'NR>1 {count++} END {print count+0}' "${REGRESSION_TEMP_DIR}/regression.summary.csv")
    fi

    echo "Total comparable tests: ${total_count}"
    echo "Performance regressions detected: ${reg_count}"

    # Copy regression summary to log directory
    cp "${REGRESSION_TEMP_DIR}/regression.summary.csv" "${LOG_DIR}/regression_summary.csv"
    echo "Copied regression summary to: ${LOG_DIR}/regression_summary.csv"

    # Run performance rerun if regressions found and not skipping
    if [[ ${reg_count} -gt 0 ]] && [[ "${SKIP_ACCURACY}" != "true" ]]; then
        echo "=== Running Performance Rerun for Regressed Models ==="

        # Save current environment info before baseline run
        CURRENT_LOG_DIR="${LOG_DIR}/target"
        mkdir -p "${CURRENT_LOG_DIR}"
        python3 -m pip freeze > "${CURRENT_LOG_DIR}/environment.txt" 2>/dev/null || true

        # Run target tests (current environment)
        echo "--- Running Target Tests (Current Environment) ---"
        TARGET_LOG_DIR="${LOG_DIR}/target_rerun"
        if ! run_performance_tests \
            "${REGRESSION_TEMP_DIR}/regression.summary.csv" \
            "${PERF_RERUN_COUNT}" "target"; then
            echo "Warning: Target performance rerun encountered issues"
        fi

        # Setup and run baseline tests
        echo "--- Setting up Baseline Environment ---"
        BASELINE_LOG_DIR="${LOG_DIR}/baseline_rerun"

        if setup_baseline_environment "${BASELINE_TYPE}" "${TORCH_VERSION}" "${BASELINE_DIR}"; then
            echo "--- Running Baseline Tests ---"
            if ! run_performance_tests \
                "${REGRESSION_TEMP_DIR}/regression.summary.csv" \
                "${PERF_RERUN_COUNT}" "baseline"; then
                echo "Warning: Baseline performance rerun encountered issues"
            fi

            # Restore original environment
            restore_original_environment
        else
            echo "ERROR: Failed to setup baseline environment"
        fi

        # Compare target vs baseline results
        echo "=== Comparing Target vs Baseline Results ==="
        if [[ -d "${TARGET_LOG_DIR}" ]] && [[ -d "${BASELINE_LOG_DIR}" ]]; then
            compare_target_baseline "${TARGET_LOG_DIR}" "${BASELINE_LOG_DIR}" "${LOG_DIR}/comparison.csv"
        fi

    elif [[ "${SKIP_ACCURACY}" == "true" ]]; then
        echo "Skipping accuracy tests as requested."
    fi

    # Generate final report
    generate_final_report() {
        local report_file="${LOG_DIR}/final_regression_report.txt"

        echo "=== Final Regression Report ===" | tee "${report_file}"
        echo "Date: $(date)" | tee -a "${report_file}"
        echo "Reference Directory: ${REFERENCE_DIR}" | tee -a "${report_file}"
        echo "Current Run Directory: ${LOG_DIR}" | tee -a "${report_file}"
        echo "Regression Analysis: ${LOG_DIR}/regression_summary.csv" | tee -a "${report_file}"
        echo "Total Tests: ${total_count}" | tee -a "${report_file}"
        echo "Regressions Detected: ${reg_count}" | tee -a "${report_file}"

        if [[ -f "${LOG_DIR}/regression_analysis.csv" ]]; then
            echo "Rerun Analysis: ${LOG_DIR}/regression_analysis.csv" | tee -a "${report_file}"

            # Summarize rerun results
            if [[ -f "${LOG_DIR}/regression_analysis.csv" ]]; then
                echo "" | tee -a "${report_file}"
                echo "Rerun Analysis Summary:" | tee -a "${report_file}"
                awk -F',' '
                BEGIN {
                    total=0; pass=0; regress=0; improve=0;
                }
                NR>1 {
                    total++
                    if ($10 == "PASS") pass++
                    else if ($10 == "REGRESSION") regress++
                    else if ($10 == "IMPROVEMENT") improve++
                }
                END {
                    printf("  Total reruns: %d\n", total)
                    printf("  Pass: %d\n", pass)
                    printf("  Regressions: %d\n", regress)
                    printf("  Improvements: %d\n", improve)
                }' "${LOG_DIR}/regression_analysis.csv" | tee -a "${report_file}"
            fi
        fi

        # Check if comparison file exists
        if [[ -f "${LOG_DIR}/comparison.csv" ]]; then
            echo "" | tee -a "${report_file}"
            echo "Target vs Baseline Comparison:" | tee -a "${report_file}"
            cat "${LOG_DIR}/comparison.csv" | tee -a "${report_file}"
        fi

        echo "" | tee -a "${report_file}"
        echo "=== Report End ===" | tee -a "${report_file}"

        echo "Final report saved to: ${report_file}"
    }

    generate_final_report

    # Clean up temp directory
    rm -rf "${REGRESSION_TEMP_DIR}"
    echo "Cleaned up temporary directory"

    # Exit with error if regressions found
    if [[ ${reg_count} -gt 0 ]]; then
        echo "WARNING: Performance regressions detected!"
        echo "Exit code: 1"
        exit 1
    else
        echo "SUCCESS: No performance regressions detected."
        echo "Exit code: 0"
    fi
fi

echo "Script completed"
exit ${EXIT_CODE}
