#!/bin/bash
# Enhanced Benchmark Runner with proper CSV format handling
# Usage: ./run_benchmark.sh [OPTIONS]

set -euo pipefail

# Default configuration
declare -A DEFAULTS=(
    ["SUITE"]="huggingface"     # huggingface / torchbench / timm_models
    ["DTYPE"]="float32"         # float32 / float16 / amp / amp_bf16 / amp_fp16
    ["MODE"]="inference"        # inference / training
    ["SCENARIO"]="accuracy"     # accuracy / performance
    ["DEVICE"]="xpu"            # xpu / cuda / cpu
    ["CARD"]="0"                # 0 / 1 / 2 / 3 ...
    ["SHAPE"]="static"          # static / dynamic
    ["NUM_SHARDS"]=""           # num test shards
    ["SHARD_ID"]=""             # shard id
    ["MODEL_ONLY"]=""           # GoogleFnet / T5Small / ...
    ["WORKSPACE"]="$(pwd)"      # Workspace directory
    ["LOG_DIR"]=""              # Log directory
    ["TIMEOUT"]="10800"         # Timeout in seconds
    ["ITERATIONS"]="10"         # Number of iterations
    ["COLD_START"]="true"       # Include cold start latency
    ["CUDAGRAPHS"]="false"      # Enable cudagraphs
    ["DRY_RUN"]="false"         # Dry run mode
    ["VERBOSE"]="false"         # Verbose output
    ["BATCH_SIZE"]=""           # Specific batch size
)

# Import utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILS_FILE="${SCRIPT_DIR}/benchmark_utils.sh"

if [[ -f "${UTILS_FILE}" ]]; then
    source "${UTILS_FILE}"
else
    # Basic logging if utils not found
    log_info() { echo "[INFO] $*"; }
    log_success() { echo "[SUCCESS] $*"; }
    log_warning() { echo "[WARNING] $*"; }
    log_error() { echo "[ERROR] $*"; }
fi

# Function to display usage
show_usage() {
    cat <<EOF
Enhanced Benchmark Runner with proper CSV format handling

Usage: $(basename "$0") [OPTIONS]

Primary Options:
  -s, --suite SUITE          Test suite: huggingface, torchbench, timm_models
  -d, --dtype DTYPE          Data type: float32, float16, amp, amp_bf16, amp_fp16
  -m, --mode MODE            Mode: inference, training
  -c, --scenario SCENARIO    Scenario: accuracy, performance
  -e, --device DEVICE        Device: xpu, cuda, cpu
  -a, --card CARD            Device card index (default: 0)

Model Selection:
  -o, --model-only MODEL     Run only specific model (supports -k pattern)
  --batch-size SIZE          Specific batch size for all models

Execution Control:
  -p, --shape SHAPE          Shape: static, dynamic
  -n, --num-shards NUM       Number of shards for parallel execution
  -i, --shard-id ID          Shard ID for this execution
  -t, --timeout SEC          Timeout in seconds
  -it, --iterations NUM      Number of iterations
  --no-cold-start            Disable cold start latency measurement
  --enable-cudagraphs        Enable cudagraphs

Output Options:
  -w, --workspace DIR        Workspace directory
  -l, --log-dir DIR          Log directory (auto-generated)
  --output FILE              Custom output CSV file

Miscellaneous:
  --dry-run                  Show command but don't execute
  -v, --verbose              Verbose output
  -h, --help                 Show this help message

Examples:
  # Performance test with proper CSV parsing
  $(basename "$0") --suite huggingface --scenario performance --dtype float32

  # Accuracy test
  $(basename "$0") --suite torchbench --scenario accuracy --mode training

  # Specific model with batch size
  $(basename "$0") --model-only "AlbertForMaskedLM" --batch-size 4

  # Dry run to verify command
  $(basename "$0") --dry-run --verbose

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
            --batch-size)
                BATCH_SIZE="$2"
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
            --output)
                OUTPUT_CSV="$2"
                shift 2
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -it|--iterations)
                ITERATIONS="$2"
                shift 2
                ;;
            --no-cold-start)
                COLD_START="false"
                shift
                ;;
            --enable-cudagraphs)
                CUDAGRAPHS="true"
                shift
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                echo "Error: Unknown option: $1"
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

# Setup logging and directories
setup_environment() {
    # Auto-generate log directory if not specified
    if [[ -z "${LOG_DIR}" ]]; then
        local timestamp=$(date +%Y%m%d_%H%M%S)
        LOG_DIR="${WORKSPACE}/inductor_log/${SUITE}/${DTYPE}/${MODE}_${SCENARIO}_${DEVICE}_${timestamp}"
        if [[ -n "${SHARD_ID}" ]]; then
            LOG_DIR="${LOG_DIR}_shard${SHARD_ID}"
        fi
    fi

    # Create log directory
    if ! mkdir -p "${LOG_DIR}"; then
        log_error "Failed to create log directory: ${LOG_DIR}"
        exit 1
    fi

    # Set output CSV file
    if [[ -z "${OUTPUT_CSV}" ]]; then
        OUTPUT_CSV="${LOG_DIR}/results.csv"
    fi

    # Set summary file
    SUMMARY_FILE="${LOG_DIR}/summary.csv"

    # Main log file
    MAIN_LOG="${LOG_DIR}/run.log"

    # Config file
    CONFIG_FILE="${LOG_DIR}/config.txt"

    log_info "Log directory: ${LOG_DIR}"
    log_info "Output CSV: ${OUTPUT_CSV}"
    log_info "Summary file: ${SUMMARY_FILE}"
}

# Display configuration
print_configuration() {
    echo "========================================"
    echo "Benchmark Configuration"
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
    echo "Output CSV:     ${OUTPUT_CSV}"
    echo "Timeout:        ${TIMEOUT}s"
    echo "Iterations:     ${ITERATIONS}"
    echo "Cold Start:     ${COLD_START}"
    echo "Cudagraphs:     ${CUDAGRAPHS}"

    if [[ -n "${BATCH_SIZE}" ]]; then
        echo "Batch Size:     ${BATCH_SIZE}"
    fi

    if [[ -n "${NUM_SHARDS}" && -n "${SHARD_ID}" ]]; then
        echo "Shards:         ${SHARD_ID}/${NUM_SHARDS}"
    fi

    if [[ -n "${MODEL_ONLY}" ]]; then
        echo "Model Filter:   ${MODEL_ONLY}"
    fi
    echo "========================================"

    # Save configuration to file
    {
        echo "Benchmark Configuration"
        echo "======================="
        echo "Timestamp: $(date)"
        echo "Command: $0 $*"
        echo "Suite: ${SUITE}"
        echo "Data Type: ${DTYPE}"
        echo "Mode: ${MODE}"
        echo "Scenario: ${SCENARIO}"
        echo "Device: ${DEVICE}"
        echo "Card: ${CARD}"
        echo "Shape: ${SHAPE}"
        echo "Workspace: ${WORKSPACE}"
        echo "Log Directory: ${LOG_DIR}"
        echo "Output CSV: ${OUTPUT_CSV}"
        echo "Timeout: ${TIMEOUT}"
        echo "Iterations: ${ITERATIONS}"
        echo "Cold Start: ${COLD_START}"
        echo "Cudagraphs: ${CUDAGRAPHS}"
        if [[ -n "${BATCH_SIZE}" ]];then
            echo "Batch Size: ${BATCH_SIZE}"
        fi
        if [[ -n "${NUM_SHARDS}" ]];then
            echo "Num Shards: ${NUM_SHARDS}"
        fi
        if [[ -n "${SHARD_ID}" ]];then
            echo "Shard ID: ${SHARD_ID}"
        fi
        if [[ -n "${MODEL_ONLY}" ]];then
            echo "Model Filter: ${MODEL_ONLY}"
        fi
    } > "${CONFIG_FILE}"
}

# Build command line arguments
build_command() {
    local cmd=()

    # Basic command
    cmd+=("python" "benchmarks/dynamo/${SUITE}.py")
    cmd+=("--${SCENARIO}")
    cmd+=("--${REAL_DTYPE}")
    cmd+=("-d" "${DEVICE}")
    cmd+=("-n${ITERATIONS}")

    # Add optional parameters
    [[ -n "${DTYPE_EXTRA}" ]] && cmd+=("${DTYPE_EXTRA}")
    [[ -n "${MODE_EXTRA}" ]] && cmd+=("${MODE_EXTRA}")
    [[ -n "${SHAPE_EXTRA}" ]] && IFS=' ' read -ra SHAPE_PARTS <<< "${SHAPE_EXTRA}" && cmd+=("${SHAPE_PARTS[@]}")
    [[ -n "${PARTITION_FLAGS}" ]] && IFS=' ' read -ra PARTITION_PARTS <<< "${PARTITION_FLAGS}" && cmd+=("${PARTITION_PARTS[@]}")
    [[ ${#MODEL_ONLY_EXTRA[@]} -gt 0 ]] && cmd+=("${MODEL_ONLY_EXTRA[@]}")

    # Batch size if specified
    [[ -n "${BATCH_SIZE}" ]] && cmd+=("--batch-size" "${BATCH_SIZE}")

    # Backend and performance options
    cmd+=("--backend=inductor")
    [[ "${COLD_START}" == "true" ]] && cmd+=("--cold-start-latency")
    cmd+=("--timeout=${TIMEOUT}")
    [[ "${CUDAGRAPHS}" == "false" ]] && cmd+=("--disable-cudagraphs")

    # Output file
    cmd+=("--output=${OUTPUT_CSV}")

    echo "${cmd[@]}"
}

# Process performance CSV results
process_performance_csv() {
    local input_csv="$1"
    local output_summary="$2"

    log_info "Processing performance CSV: $(basename "$input_csv")"

    awk -F, -v suite="${SUITE}" -v dtype="${DTYPE}" -v mode="${MODE}" -v scenario="${SCENARIO}" '
    BEGIN {
        OFS=","
        print "Suite,Dtype,Mode,Scenario,Model,BS,Accuracy,Eager_Latency,Inductor_Latency,Speedup"
    }
    NR > 1 && $1 == "'"${DEVICE}"'" {
        model = $2
        batch_size = $3
        speedup = $4 + 0
        abs_latency = $5 + 0

        # Clean model name
        gsub(/"/, "", model)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", model)

        if (abs_latency > 0) {
            eager_latency = speedup * abs_latency
            inductor_latency = abs_latency
            accuracy = "-1"

            print suite, dtype, mode, scenario, model, batch_size, accuracy, \
                  eager_latency, inductor_latency, speedup
        } else {
            print suite, dtype, mode, scenario, model, batch_size, "ERROR", \
                  "-1", "-1", speedup
        }
    }' "${input_csv}" > "${output_summary}"
}

# Process accuracy CSV results
process_accuracy_csv() {
    local input_csv="$1"
    local output_summary="$2"

    log_info "Processing accuracy CSV: $(basename "$input_csv")"

    awk -F, -v suite="${SUITE}" -v dtype="${DTYPE}" -v mode="${MODE}" -v scenario="${SCENARIO}" '
    BEGIN {
        OFS=","
        print "Suite,Dtype,Mode,Scenario,Model,BS,Accuracy,Eager_Latency,Inductor_Latency,Speedup"
    }
    NR > 1 && $1 == "'"${DEVICE}"'" {
        model = $2
        batch_size = $3
        accuracy = $4

        # Clean model name
        gsub(/"/, "", model)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", model)

        # Check if accuracy passed
        if (accuracy ~ /pass/) {
            accuracy_status = "PASS"
        } else if (accuracy ~ /eager_two_runs_differ/) {
            accuracy_status = "EAGER_DIFF"
        } else if (accuracy ~ /fail/) {
            accuracy_status = "FAIL"
        } else {
            accuracy_status = accuracy
        }

        print suite, dtype, mode, scenario, model, batch_size, accuracy, \
              "-1", "-1", "-1"
    }' "${input_csv}" > "${output_summary}"
}

# Generate statistics
generate_statistics() {
    local summary_file="$1"
    local stats_file="$2"

    {
        echo "Benchmark Statistics"
        echo "===================="
        echo "Generated: $(date)"
        echo "Summary file: $(basename "$summary_file")"
        echo ""

        # Count total tests
        local total_tests=$(awk 'NR>1 && $5 != "" {count++} END {print count}' "$summary_file")
        echo "Total tests: ${total_tests}"

        if [[ "${SCENARIO}" == "performance" ]]; then
            echo ""
            echo "Performance Statistics:"
            echo "----------------------"

            awk -F',' '
            BEGIN {
                count=0
                total_speedup=0
                min_speedup=999
                max_speedup=0
                total_eager=0
                total_inductor=0

                speedup_buckets[0]=0  # < 0.5x
                speedup_buckets[1]=0  # 0.5-0.8x
                speedup_buckets[2]=0  # 0.8-1.0x
                speedup_buckets[3]=0  # 1.0-1.2x
                speedup_buckets[4]=0  # 1.2-1.5x
                speedup_buckets[5]=0  # > 1.5x
            }
            NR>1 && $8 != "-1" && $9 != "-1" {
                count++
                speedup = $10 + 0
                eager_latency = $8 + 0
                inductor_latency = $9 + 0

                total_speedup += speedup
                total_eager += eager_latency
                total_inductor += inductor_latency

                if (speedup < min_speedup) min_speedup = speedup
                if (speedup > max_speedup) max_speedup = speedup

                if (speedup < 0.5) speedup_buckets[0]++
                else if (speedup < 0.8) speedup_buckets[1]++
                else if (speedup < 1.0) speedup_buckets[2]++
                else if (speedup < 1.2) speedup_buckets[3]++
                else if (speedup < 1.5) speedup_buckets[4]++
                else speedup_buckets[5]++
            }
            END {
                if (count > 0) {
                    printf("  Models with valid data: %d\n", count)
                    printf("  Average speedup: %.3fx\n", total_speedup/count)
                    printf("  Minimum speedup: %.3fx\n", min_speedup)
                    printf("  Maximum speedup: %.3fx\n", max_speedup)
                    printf("  Average eager latency: %.2f ms\n", total_eager/count)
                    printf("  Average inductor latency: %.2f ms\n", total_inductor/count)
                    printf("\n  Speedup distribution:\n")
                    printf("    < 0.5x:  %3d models\n", speedup_buckets[0])
                    printf("    0.5-0.8x: %3d models\n", speedup_buckets[1])
                    printf("    0.8-1.0x: %3d models\n", speedup_buckets[2])
                    printf("    1.0-1.2x: %3d models\n", speedup_buckets[3])
                    printf("    1.2-1.5x: %3d models\n", speedup_buckets[4])
                    printf("    > 1.5x:  %3d models\n", speedup_buckets[5])
                }
            }
            ' "${summary_file}"

        elif [[ "${SCENARIO}" == "accuracy" ]]; then
            echo ""
            echo "Accuracy Statistics:"
            echo "-------------------"

            awk -F',' '
            BEGIN {
                pass=0
                fail=0
                eager_diff=0
                other=0
            }
            NR>1 {
                accuracy = $7
                if (accuracy == "PASS") pass++
                else if (accuracy == "FAIL") fail++
                else if (accuracy == "EAGER_DIFF") eager_diff++
                else other++
            }
            END {
                total = pass + fail + eager_diff + other
                printf("  PASS:           %3d (%.1f%%)\n", pass, (pass/total)*100)
                printf("  FAIL:           %3d (%.1f%%)\n", fail, (fail/total)*100)
                printf("  EAGER_DIFF:     %3d (%.1f%%)\n", eager_diff, (eager_diff/total)*100)
                printf("  Other:          %3d (%.1f%%)\n", other, (other/total)*100)
                printf("  Total:          %3d\n", total)
            }
            ' "${summary_file}"
        fi

        echo ""
        echo "Detailed results in: $(basename "$summary_file")"

    } > "${stats_file}"

    if [[ "${VERBOSE}" == "true" ]]; then
        cat "${stats_file}"
    fi
}

# Main execution function
main() {
    log_info "Starting benchmark execution"

    # Setup environment
    setup_environment

    # Build parameters
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

    # Mode extra parameter
    MODE_EXTRA=""
    if [[ "${MODE}" == "training" ]]; then
        MODE_EXTRA="--training"
    elif [[ "${MODE}" == "inference" ]]; then
        MODE_EXTRA="--inference"
    fi

    # Shape extra parameter
    SHAPE_EXTRA=""
    if [[ "${SHAPE}" == "dynamic" ]]; then
        SHAPE_EXTRA="--dynamic-shapes --dynamic-batch-only"
    fi

    # Partition flags
    PARTITION_FLAGS=""
    if [[ -n "${NUM_SHARDS}" && -n "${SHARD_ID}" ]] && [[ "${NUM_SHARDS}" -gt 1 ]]; then
        if [[ "${SHARD_ID}" -ge "${NUM_SHARDS}" ]]; then
            log_error "Shard ID (${SHARD_ID}) must be less than total shards (${NUM_SHARDS})"
            exit 1
        fi
        PARTITION_FLAGS="--total-partitions ${NUM_SHARDS} --partition-id ${SHARD_ID}"
        log_info "Running shard ${SHARD_ID} of ${NUM_SHARDS}"
    fi

    # Model only extra parameter
    MODEL_ONLY_EXTRA=()
    if [[ -n "${MODEL_ONLY}" ]]; then
        if [[ "${MODEL_ONLY}" == *"-k "* ]]; then
            MODEL_ONLY_EXTRA=("${MODEL_ONLY}")
        else
            MODEL_ONLY_EXTRA=("--only" "${MODEL_ONLY}")
        fi
        log_info "Model filter: ${MODEL_ONLY}"
    fi

    # Display configuration
    print_configuration

    # Build command
    CMD=$(build_command)

    echo ""
    log_info "Command to execute:"
    echo "${CMD}"
    echo ""

    # Dry run check
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "Dry run mode - command would be executed as:"
        echo "ZE_AFFINITY_MASK=\"${CARD}\" ${CMD}"
        echo ""
        log_info "To actually run, remove --dry-run flag"
        exit 0
    fi

    # Execute command
    log_info "Starting benchmark execution at $(date)"
    log_info "Full logs: ${MAIN_LOG}"

    # Set ulimit
    ulimit -n 1048576 2>/dev/null || true

    # Execute with logging
    {
        echo "========================================"
        echo "Benchmark Execution Log"
        echo "========================================"
        echo "Start time: $(date)"
        echo "Command: ${CMD}"
        echo "Environment: ZE_AFFINITY_MASK=${CARD}"
        echo "========================================"
        echo ""
    } > "${MAIN_LOG}"

    # Execute command
    start_time=$(date +%s)

    log_info "Executing benchmark..."
    if ZE_AFFINITY_MASK="${CARD}" \
        ${CMD} 2>&1 | tee -a "${MAIN_LOG}"; then
        EXIT_CODE=0
        log_success "Benchmark execution completed successfully"
    else
        EXIT_CODE=${PIPESTATUS[0]}
        log_error "Benchmark execution failed with exit code: ${EXIT_CODE}"
    fi

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    # Log completion
    {
        echo ""
        echo "========================================"
        echo "Completion time: $(date)"
        echo "Total duration: ${duration} seconds"
        echo "Exit code: ${EXIT_CODE}"
        echo "========================================"
    } >> "${MAIN_LOG}"

    # Process results if successful
    if [[ ${EXIT_CODE} -eq 0 ]] && [[ -f "${OUTPUT_CSV}" ]]; then
        log_info "Processing results..."

        if [[ "${SCENARIO}" == "performance" ]]; then
            process_performance_csv "${OUTPUT_CSV}" "${SUMMARY_FILE}"
        else
            process_accuracy_csv "${OUTPUT_CSV}" "${SUMMARY_FILE}"
        fi

        # Generate statistics
        generate_statistics "${SUMMARY_FILE}" "${LOG_DIR}/statistics.txt"

        log_success "Results processed successfully"
        log_info "Summary file: ${SUMMARY_FILE}"
        log_info "Statistics: ${LOG_DIR}/statistics.txt"

        # Show quick summary
        echo ""
        echo "Quick Summary:"
        echo "--------------"
        if [[ "${SCENARIO}" == "performance" ]]; then
            awk -F',' '
            NR>1 && $8 != "-1" && $9 != "-1" {
                count++
                total_speedup += $10
            }
            END {
                if (count > 0) {
                    printf("  Models with valid data: %d\n", count)
                    printf("  Average speedup: %.3fx\n", total_speedup/count)
                }
            }' "${SUMMARY_FILE}"
        else
            awk -F',' '
            BEGIN { pass=0; total=0 }
            NR>1 { total++ }
            $7 == "PASS" { pass++ }
            END {
                printf("  Total models: %d\n", total)
                printf("  PASS: %d (%.1f%%)\n", pass, (pass/total)*100)
            }' "${SUMMARY_FILE}"
        fi

    else
        log_warning "No results to process or execution failed"
        if [[ ! -f "${OUTPUT_CSV}" ]]; then
            log_error "Output CSV file not found: ${OUTPUT_CSV}"
        fi
    fi

    echo ""
    log_info "Benchmark completed"
    log_info "Duration: ${duration} seconds"
    log_info "Log directory: ${LOG_DIR}"
    log_info "Exit code: ${EXIT_CODE}"

    return ${EXIT_CODE}
}

# Run main function
main "$@"
exit $?

