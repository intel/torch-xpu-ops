#!/bin/bash
# Enhanced script for CPU/XPU/CUDA device Dynamo benchmark tests

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
  -r, --reference-dir DIR    reference result directory
  -h, --help                 Show this help message

Examples:
  $(basename "$0") --suite huggingface --dtype amp_bf16 --mode inference
  $(basename "$0") -s torchbench -d float16 -m training -c performance
  $(basename "$0") --model-only "GoogleFnet" --device cuda --card 1
  $(basename "$0") --num-shards 4 --shard-id 2 --model-only "-k T5"

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
            -l|--reference-dir)
                REFERENCE_DIR="$2"
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
    
    if [[ -n "${NUM_SHARDS}" && -n "${SHARD_ID}" ]]; then
        echo "Shards:         ${SHARD_ID}/${NUM_SHARDS}"
    fi
    
    if [[ -n "${MODEL_ONLY}" ]]; then
        echo "Model Filter:   ${MODEL_ONLY}"
    fi
    echo "========================================"
}

print_configuration

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
if printf "%s\n2.0.2\n%s" "${TORCH_VERSION}" | sort -V | head -1 | grep -q "^2.0.2$"; then
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

echo "Script completed."
exit ${EXIT_CODE}

# Function to extract unique test cases from CSV files
extract_test_cases() {
    local dir=$1
    local output_file=$2
    
    echo "Extracting test cases from $dir..." | tee -a "$LOG_FILE"
    
    # Find all CSV files and concatenate, removing header from all but first
    first_file=true
    for csv in $(find $dir -type f -name "*.summary.csv"); do
        if [ -f "$csv" ]; then
            if [ "$first_file" = true ]; then
                # First file: include header
                head -n 1 "$csv" > "$output_file"
                tail -n +2 "$csv" >> "$output_file"
                first_file=false
            else
                # Subsequent files: skip header
                tail -n +2 "$csv" >> "$output_file"
            fi
        fi
    done
    
    if [ ! -f "$output_file" ] || [ ! -s "$output_file" ]; then
        echo "ERROR: No CSV files found in $dir" | tee -a "$LOG_FILE"
        exit 1
    fi
    
    echo "Extracted $(wc -l < "$output_file") lines from $dir" | tee -a "$LOG_FILE"
}

# Function to find performance regression test cases
find_perf_reg_cases() {
    local new_file=$1
    local ref_file=$2
    local output_file=$3
    
    echo "Finding matching test cases..." | tee -a "$LOG_FILE"
    
    # Create a combined view with both results
    # Using awk to match on Suite,Dtype,Mode,Scenario,Model,BS
    awk -F',' '
    BEGIN {
        OFS=","
    }
    NR==FNR {
        # Read new results
        key = $1 "," $2 "," $3 "," $4 "," $5 "," $6
        new_eager = $7
        new_inductor = $8
        new_data[key] = new_eager "," new_inductor
        next
    }
    {
        # Read reference results
        key = $1 "," $2 "," $3 "," $4 "," $5 "," $6
        ref_eager = $7
        ref_inductor = $8
        
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
            if (eager_ratio < 0.9 || inductor_ratio < 0.9) {
                regression = "YES"
            }
            
            print $0, new_eager_val, new_inductor_val, eager_ratio, inductor_ratio, regression
        }
    }' "$new_file" "$ref_file" > "$output_file"
    
    # Add header to the combined file
    header="Suite,Dtype,Mode,Scenario,Model,BS,Ref_Eager,Ref_Inductor,New_Eager,New_Inductor,Eager_Ratio,Inductor_Ratio,Regression"
    sed -i "1s/^/$header\n/" "$output_file"
    
    echo "Generated combined comparison file: $output_file" | tee -a "$LOG_FILE"
}

extract_test_cases ${LOG_DIR}       /tmp/new.summary.csv
extract_test_cases ${REFERENCE_DIR} /tmp/reference.summary.csv
find_perf_reg_cases /tmp/new.summary.csv /tmp/reference.summary.csv /tmp/regression.summary.csv

# Function to extract regression cases
extract_regression_cases() {
    local combined_file=$1
    local regression_file=$2
    
    echo "Extracting regression cases..." | tee -a "$LOG_FILE"
    
    # Extract lines where Regression is YES
    awk -F',' 'NR==1 || $NF == "YES"' "$combined_file" > "$regression_file"
    
    regression_count=$(($(wc -l < "$regression_file") - 1))
    echo "Found $regression_count regression cases" | tee -a "$LOG_FILE"
    
    if [ $regression_count -gt 0 ]; then
        echo "Regression cases:" | tee -a "$LOG_FILE"
        tail -n +2 "$regression_file" | tee -a "$LOG_FILE"
    fi
}

# Function to generate rerun commands
generate_rerun_commands() {
    local regression_file=$1
    local commands_file=$2
    
    echo "Generating rerun commands..." | tee -a "$LOG_FILE"
    
    # Create header
    echo "# Rerun commands for regression cases" > "$commands_file"
    echo "# Generated on $(date)" >> "$commands_file"
    echo "" >> "$commands_file"
    
    # Skip header and process each regression case
    tail -n +2 "$regression_file" | while IFS=',' read -r suite dtype mode scenario model bs ref_eager ref_inductor new_eager new_inductor eager_ratio inductor_ratio regression; do
        # Clean up variables
        suite=$(echo "$suite" | tr -d '"')
        dtype=$(echo "$dtype" | tr -d '"')
        mode=$(echo "$mode" | tr -d '"')
        scenario=$(echo "$scenario" | tr -d '"')
        model=$(echo "$model" | tr -d '"')
        
        echo "# Regression: $suite $dtype $mode $scenario $model BS=$bs" >> "$commands_file"
        echo "#   Ref Eager: $ref_eager, New Eager: $new_eager, Ratio: $eager_ratio" >> "$commands_file"
        echo "#   Ref Inductor: $ref_inductor, New Inductor: $new_inductor, Ratio: $inductor_ratio" >> "$commands_file"
        echo "# Run new version:" >> "$commands_file"
        echo "# "  "$commands_file"
        echo "# Run reference version:" >> "$commands_file"
        echo "# " >> "$commands_file"
        echo "" >> "$commands_file"
    done
    
    echo "Rerun commands saved to: $commands_file" | tee -a "$LOG_FILE"
}

# Function to rerun specific test cases
rerun_test_cases() {
    local regression_file=$1
    local output_file="$OUTPUT_DIR/rerun_comparison.csv"
    
    echo "Rerunning regression cases..." | tee -a "$RERUN_LOG"
    echo "Date: $(date)" >> "$RERUN_LOG"
    echo "========================================" >> "$RERUN_LOG"
    
    # Create output file with header
    echo "Suite,Dtype,Mode,Scenario,Model,BS,Original_Ref_Eager,Original_New_Eager,Original_Eager_Ratio,Rerun_Ref_Eager,Rerun_New_Eager,Rerun_Eager_Ratio,Original_Ref_Inductor,Original_New_Inductor,Original_Inductor_Ratio,Rerun_Ref_Inductor,Rerun_New_Inductor,Rerun_Inductor_Ratio,Regression_Fixed" > "$output_file"
    
    # Skip header and process each regression case
    tail -n +2 "$regression_file" | while IFS=',' read -r suite dtype mode scenario model bs ref_eager ref_inductor new_eager new_inductor eager_ratio inductor_ratio regression; do
        # Clean up variables
        suite=$(echo "$suite" | tr -d '"')
        dtype=$(echo "$dtype" | tr -d '"')
        mode=$(echo "$mode" | tr -d '"')
        scenario=$(echo "$scenario" | tr -d '"')
        model=$(echo "$model" | tr -d '"')
        
        echo "Rerunning: $suite $dtype $mode $scenario $model BS=$bs" | tee -a "$RERUN_LOG"
        
        # Simulate rerun (replace with actual benchmark commands)
        # This is a placeholder - you need to implement actual rerun logic
        echo "  [SIMULATION] Running new version..." | tee -a "$RERUN_LOG"
        # Actual command: ./run_benchmark.py --suite "$suite" --dtype "$dtype" --mode "$mode" --scenario "$scenario" --model "$model" --bs "$bs"
        rerun_new_eager=$(awk -v min=0.8 -v max=1.2 'BEGIN{srand(); print min+rand()*(max-min)}')
        rerun_new_inductor=$(awk -v min=0.8 -v max=1.2 'BEGIN{srand(); print min+rand()*(max-min)}')
        
        echo "  [SIMULATION] Running reference version..." | tee -a "$RERUN_LOG"
        # Actual command: ./run_benchmark.py --suite "$suite" --dtype "$dtype" --mode "$mode" --scenario "$scenario" --model "$model" --bs "$bs" --reference
        rerun_ref_eager=$(awk -v min=0.8 -v max=1.2 'BEGIN{srand(); print min+rand()*(max-min)}')
        rerun_ref_inductor=$(awk -v min=0.8 -v max=1.2 'BEGIN{srand(); print min+rand()*(max-min)}')
        
        # Calculate new ratios
        if [ $(echo "$rerun_new_eager > 0" | bc -l 2>/dev/null || echo "1") ]; then
            rerun_eager_ratio=$(echo "$rerun_ref_eager / $rerun_new_eager" | bc -l 2>/dev/null || echo "0")
        else
            rerun_eager_ratio=0
        fi
        
        if [ $(echo "$rerun_new_inductor > 0" | bc -l 2>/dev/null || echo "1") ]; then
            rerun_inductor_ratio=$(echo "$rerun_ref_inductor / $rerun_new_inductor" | bc -l 2>/dev/null || echo "0")
        else
            rerun_inductor_ratio=0
        fi
        
        # Check if regression is fixed
        regression_fixed="YES"
        if [ $(echo "$rerun_eager_ratio < 0.9" | bc -l 2>/dev/null || echo "0") -eq 1 ] || [ $(echo "$rerun_inductor_ratio < 0.9" | bc -l 2>/dev/null || echo "0") -eq 1 ]; then
            regression_fixed="NO"
        fi
        
        # Write results
        echo "\"$suite\",\"$dtype\",\"$mode\",\"$scenario\",\"$model\",$bs,$ref_eager,$new_eager,$eager_ratio,$rerun_ref_eager,$rerun_new_eager,$rerun_eager_ratio,$ref_inductor,$new_inductor,$inductor_ratio,$rerun_ref_inductor,$rerun_new_inductor,$rerun_inductor_ratio,$regression_fixed" >> "$output_file"
        
        echo "  Results: New Eager=$rerun_new_eager, Ref Eager=$rerun_ref_eager, Ratio=$rerun_eager_ratio" | tee -a "$RERUN_LOG"
        echo "           New Inductor=$rerun_new_inductor, Ref Inductor=$rerun_ref_inductor, Ratio=$rerun_inductor_ratio" | tee -a "$RERUN_LOG"
        echo "  Regression fixed: $regression_fixed" | tee -a "$RERUN_LOG"
        echo "" >> "$RERUN_LOG"
    done
    
    echo "Rerun results saved to: $output_file" | tee -a "$LOG_FILE"
}
