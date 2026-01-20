#!/bin/bash
# Enhanced Regression Checker with Full Automation
# Features:
# 1. Check regression or not
# 2. Rerun target 3 times
# 3. Copy target python env and setup baseline env
# 4. Rerun baseline 3 times
# 5. Check the 3 avg regression or not
# 6. Final verification report

set -euo pipefail

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR=""
PYTHON_ENV_DIR=""
BASELINE_ENV_DIR=""
TEMP_DIR=""
REGRESSION_CONFIRMED=false

# Default configuration
declare -A DEFAULTS=(
    ["NEW_RESULTS_DIR"]=""      # New results directory (required)
    ["REFERENCE_DIR"]=""        # Reference results directory (required)
    ["OUTPUT_DIR"]=""           # Output directory for regression analysis
    ["REGRESSION_THRESHOLD"]="0.9"  # Performance regression threshold (0.9 = 90% of reference)
    ["RERUN_COUNT"]="3"         # Number of times to rerun performance tests
    ["SKIP_ACCURACY"]="false"   # Skip accuracy regression checks
    ["VERBOSE"]="false"         # Verbose output
    ["FAIL_ON_REGRESSION"]="true"  # Fail script if regressions found
    ["AUTO_SETUP_ENV"]="true"   # Automatically setup baseline environment
    ["PYTHON_EXECUTABLE"]="python"  # Python executable to use
    ["ENV_COPY_METHOD"]="conda" # conda|venv|pipenv
    ["BENCHMARK_SCRIPT"]="benchmarks/dynamo/huggingface.py"  # Benchmark script path
    ["TIMEOUT_SECONDS"]="3600"  # Timeout for each run
)

# Enhanced logging functions
log_info() { echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] \033[1;34mINFO\033[0m: $*"; }
log_success() { echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] \033[1;32mSUCCESS\033[0m: $*"; }
log_warning() { echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] \033[1;33mWARNING\033[0m: $*"; }
log_error() { echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] \033[1;31mERROR\033[0m: $*"; }
log_debug() { [[ "${VERBOSE}" == "true" ]] && echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] \033[1;36mDEBUG\033[0m: $*"; }

# Function to display usage
show_usage() {
    cat <<EOF
Enhanced Regression Checker with Full Automation

Usage: $(basename "$0") [OPTIONS]

Required Options:
  -n, --new-results DIR      Directory containing new benchmark results
  -r, --reference DIR        Directory containing reference benchmark results

Regression Options:
  -t, --threshold VALUE      Performance regression threshold (default: 0.9)
                             Values below threshold indicate regression
  --rerun-count NUM          Number of times to rerun performance tests (default: 3)
  --skip-accuracy            Skip accuracy regression checks

Environment Options:
  --python PATH              Python executable (default: python)
  --env-method METHOD        Environment copy method: conda, venv, or pipenv (default: conda)
  --no-env-setup             Skip automatic environment setup
  --benchmark-script PATH    Path to benchmark script (default: benchmarks/dynamo/huggingface.py)

Output Options:
  -o, --output DIR           Output directory for regression analysis
  --no-fail                  Don't fail script if regressions found

Miscellaneous:
  -v, --verbose              Verbose output
  -h, --help                 Show this help message

Examples:
  # Basic regression check with full automation
  $(basename "$0") --new-results ./new_run --reference ./baseline

  # With custom environment setup
  $(basename "$0") -n ./latest -r ./stable --env-method venv --python python3.9

  # Skip environment setup (use existing environments)
  $(basename "$0") -n ./test -r ./ref --no-env-setup

EOF
}

# Parse command-line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--new-results)
                NEW_RESULTS_DIR="$2"
                shift 2
                ;;
            -r|--reference)
                REFERENCE_DIR="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -t|--threshold)
                REGRESSION_THRESHOLD="$2"
                shift 2
                ;;
            --rerun-count)
                RERUN_COUNT="$2"
                shift 2
                ;;
            --skip-accuracy)
                SKIP_ACCURACY="true"
                shift
                ;;
            --no-fail)
                FAIL_ON_REGRESSION="false"
                shift
                ;;
            --no-env-setup)
                AUTO_SETUP_ENV="false"
                shift
                ;;
            --python)
                PYTHON_EXECUTABLE="$2"
                shift 2
                ;;
            --env-method)
                ENV_COPY_METHOD="$2"
                shift 2
                ;;
            --benchmark-script)
                BENCHMARK_SCRIPT="$2"
                shift 2
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

# Validate required arguments
validate_arguments() {
    if [[ -z "${NEW_RESULTS_DIR}" ]]; then
        log_error "New results directory not specified"
        show_usage
        exit 1
    fi

    if [[ -z "${REFERENCE_DIR}" ]]; then
        log_error "Reference directory not specified"
        show_usage
        exit 1
    fi

    if [[ ! -d "${NEW_RESULTS_DIR}" ]]; then
        log_error "New results directory does not exist: ${NEW_RESULTS_DIR}"
        exit 1
    fi

    if [[ ! -d "${REFERENCE_DIR}" ]]; then
        log_error "Reference directory does not exist: ${REFERENCE_DIR}"
        exit 1
    fi

    # Check Python executable
    if ! command -v "${PYTHON_EXECUTABLE}" &> /dev/null; then
        log_error "Python executable not found: ${PYTHON_EXECUTABLE}"
        exit 1
    fi

    # Check benchmark script exists if specified
    if [[ ! -f "${BENCHMARK_SCRIPT}" ]] && [[ "${BENCHMARK_SCRIPT}" != "benchmarks/dynamo/huggingface.py" ]]; then
        log_warning "Benchmark script not found: ${BENCHMARK_SCRIPT}"
    fi

    # Set output directory if not specified
    if [[ -z "${OUTPUT_DIR}" ]]; then
        OUTPUT_DIR="${NEW_RESULTS_DIR}/regression_analysis_$(date +%Y%m%d_%H%M%S)"
    fi

    # Create main output directory
    mkdir -p "${OUTPUT_DIR}" || {
        log_error "Failed to create output directory: ${OUTPUT_DIR}"
        exit 1
    }

    # Create workspace directory
    WORKSPACE_DIR="${OUTPUT_DIR}/workspace"
    mkdir -p "${WORKSPACE_DIR}" || {
        log_error "Failed to create workspace directory: ${WORKSPACE_DIR}"
        exit 1
    }

    # Create temp directory
    TEMP_DIR="${WORKSPACE_DIR}/temp"
    mkdir -p "${TEMP_DIR}"

    log_info "New results directory: ${NEW_RESULTS_DIR}"
    log_info "Reference directory: ${REFERENCE_DIR}"
    log_info "Output directory: ${OUTPUT_DIR}"
    log_info "Workspace directory: ${WORKSPACE_DIR}"
    log_info "Regression threshold: ${REGRESSION_THRESHOLD}"
    log_info "Rerun count: ${RERUN_COUNT}"
    log_info "Python executable: ${PYTHON_EXECUTABLE}"
    log_info "Environment method: ${ENV_COPY_METHOD}"
}

# Process summary file for regression analysis
process_summary_file() {
    local summary_file="$1"
    local output_file="$2"

    if [[ ! -f "${summary_file}" ]]; then
        log_error "Summary file not found: ${summary_file}"
        return 1
    fi

    # Check if it's a performance or accuracy summary
    local first_line=$(head -1 "${summary_file}")
    if [[ "${first_line}" == *"Eager_Latency"* ]] && [[ "${first_line}" == *"Inductor_Latency"* ]]; then
        log_info "Processing performance summary: $(basename "${summary_file}")"
        # Performance summary already has the format we need
        cp "${summary_file}" "${output_file}"
    else
        # Convert old format to new format
        log_info "Converting summary file: $(basename "${summary_file}")"
        awk -F',' '
        BEGIN {
            OFS=","
            print "Suite,Dtype,Mode,Scenario,Model,BS,Accuracy,Eager_Latency,Inductor_Latency,Speedup"
        }
        NR>1 {
            # Handle different formats
            if (NF >= 9) {
                # Format: Suite,Dtype,Mode,Scenario,Model,BS,Accuracy,Eager,Inductor
                suite = $1
                dtype = $2
                mode = $3
                scenario = $4
                model = $5
                bs = $6
                accuracy = $7
                eager = $8 + 0
                inductor = $9 + 0

                if (eager > 0 && inductor > 0) {
                    speedup = eager / inductor
                } else {
                    speedup = -1
                }

                print suite, dtype, mode, scenario, model, bs, accuracy, eager, inductor, speedup
            }
        }' "${summary_file}" > "${output_file}"
    fi

    return 0
}

# Analyze performance regression
analyze_performance_regression() {
    local new_summary="$1"
    local ref_summary="$2"
    local output_file="$3"

    awk -F',' -v threshold="${REGRESSION_THRESHOLD}" '
    BEGIN {
        OFS=","
        print "Model,BS,Ref_Eager,Ref_Inductor,New_Eager,New_Inductor,Eager_Ratio,Inductor_Ratio,Eager_Regression,Inductor_Regression,Regression_Type,Notes"
    }
    # Read reference data into arrays
    NR == FNR && FNR > 1 {
        key = $5 "," $6  # Model,BS
        ref_eager[key] = $8 + 0
        ref_inductor[key] = $9 + 0
        next
    }
    # Process new data
    FNR > 1 {
        model = $5
        bs = $6
        new_eager = $8 + 0
        new_inductor = $9 + 0
        key = model "," bs

        # Check if we have reference data
        if (key in ref_eager) {
            ref_e = ref_eager[key]
            ref_i = ref_inductor[key]

            # Calculate ratios (new/ref, higher is better)
            eager_ratio = 0
            inductor_ratio = 0
            eager_reg = "NO"
            inductor_reg = "NO"
            reg_type = ""
            notes = ""

            if (ref_e > 0 && new_eager > 0) {
                eager_ratio = ref_e / new_eager
                if (eager_ratio < threshold) {
                    eager_reg = "YES"
                    reg_type = (reg_type ? reg_type "|" : "") "Eager"
                    notes = "Eager performance below threshold"
                }
            } else {
                if (new_eager <= 0) notes = "Missing new eager latency"
                if (ref_e <= 0) notes = notes (notes ? "; " : "") "Missing reference eager latency"
            }

            if (ref_i > 0 && new_inductor > 0) {
                inductor_ratio = ref_i / new_inductor
                if (inductor_ratio < threshold) {
                    inductor_reg = "YES"
                    reg_type = (reg_type ? reg_type "|" : "") "Inductor"
                    notes = notes (notes ? "; " : "") "Inductor performance below threshold"
                }
            } else {
                if (new_inductor <= 0) notes = notes (notes ? "; " : "") "Missing new inductor latency"
                if (ref_i <= 0) notes = notes (notes ? "; " : "") "Missing reference inductor latency"
            }

            # Determine overall regression
            overall_reg = "NO"
            if (eager_reg == "YES" || inductor_reg == "YES") {
                overall_reg = "YES"
            }

            print model, bs, ref_e, ref_i, new_eager, new_inductor, \
                  eager_ratio, inductor_ratio, eager_reg, inductor_reg, \
                  reg_type, notes

        } else {
            # No reference data found
            print model, bs, "N/A", "N/A", new_eager, new_inductor, \
                  "N/A", "N/A", "NO", "NO", "", "No reference data"
        }
    }
    ' "${ref_summary}" "${new_summary}" > "${output_file}"

    # Count regressions
    local total=$(awk 'NR>1 {count++} END {print count}' "${output_file}")
    local regressions=$(awk -F',' 'NR>1 && ($9 == "YES" || $10 == "YES") {count++} END {print count}' "${output_file}")
    local eager_reg=$(awk -F',' 'NR>1 && $9 == "YES" {count++} END {print count}' "${output_file}")
    local inductor_reg=$(awk -F',' 'NR>1 && $10 == "YES" {count++} END {print count}' "${output_file}")

    echo "${regressions}"
}

# Step 1: Initial Regression Check
check_initial_regression() {
    log_info "Step 1: Performing initial regression check..."

    # Find summary files
    local new_summary_file=$(find "${NEW_RESULTS_DIR}" -type f -name "*summary.csv" | head -1)
    local ref_summary_file=$(find "${REFERENCE_DIR}" -type f -name "*summary.csv" | head -1)

    if [[ -z "${new_summary_file}" ]] || [[ -z "${ref_summary_file}" ]]; then
        log_error "Summary files not found in input directories"
        return 1
    fi

    log_info "Found new summary: $(basename "${new_summary_file}")"
    log_info "Found reference summary: $(basename "${ref_summary_file}")"

    # Process summary files
    local processed_new="${TEMP_DIR}/new_processed.csv"
    local processed_ref="${TEMP_DIR}/reference_processed.csv"

    process_summary_file "${new_summary_file}" "${processed_new}"
    process_summary_file "${ref_summary_file}" "${processed_ref}"

    # Analyze performance regression
    local perf_regression_file="${OUTPUT_DIR}/initial_regression.csv"
    local perf_regressions=$(analyze_performance_regression "${processed_new}" "${processed_ref}" "${perf_regression_file}")

    if [[ "${perf_regressions}" -gt 0 ]]; then
        log_warning "Initial check: ${perf_regressions} potential regression(s) found"
        REGRESSION_CONFIRMED=true
        return 1
    else
        log_success "Initial check: No regressions found"
        return 0
    fi
}

# Helper function to run benchmark
run_benchmark() {
    local models=("${@:1:$(($#-3))}")
    local run_dir="${@: -3:1}"
    local results_csv="${@: -2:1}"
    local run_log="${@: -1:1}"

    > "${run_log}"
    > "${results_csv}"

    for model_info in "${models[@]}"; do
        IFS='|' read -r model_name batch_size <<< "$model_info"

        log_debug "Running ${model_name} with BS=${batch_size}"

        # Build benchmark command
        local cmd=(
            "${PYTHON_EXECUTABLE}" "${BENCHMARK_SCRIPT}"
            "--inference"
            "--float32"
            "-d" "xpu"
            "-n10"
            "--only" "${model_name}"
            "--batch-size" "${batch_size}"
            "--backend=inductor"
            "--cold-start-latency"
            "--timeout=${TIMEOUT_SECONDS}"
            "--disable-cudagraphs"
            "--output=${run_dir}/${model_name}_bs${batch_size}.csv"
        )

        log_debug "Command: ${cmd[*]}"

        # Execute command with timeout
        if timeout "${TIMEOUT_SECONDS}" "${cmd[@]}" >> "${run_log}" 2>&1; then
            # Collect results
            local model_csv="${run_dir}/${model_name}_bs${batch_size}.csv"
            if [[ -f "${model_csv}" ]]; then
                # Extract relevant data
                awk -F',' -v model="${model_name}" -v bs="${batch_size}" '
                NR > 1 && $2 == model && $3 == bs {
                    speedup = $4 + 0
                    abs_latency = $5 + 0
                    if (abs_latency > 0) {
                        eager_latency = speedup * abs_latency
                        inductor_latency = abs_latency
                        print model "," bs "," eager_latency "," inductor_latency
                    }
                    exit
                }' "${model_csv}" >> "${results_csv}"
                log_debug "  Successfully processed ${model_name}"
            else
                log_warning "  No output CSV generated for ${model_name}"
            fi
        else
            log_warning "  Timeout or error running ${model_name}"
        fi
    done
}

# Process rerun results
process_rerun_results() {
    local input_csv="$1"
    local output_csv="$2"

    # Simple copy for now, can be enhanced for data cleaning
    if [[ -f "${input_csv}" ]]; then
        cp "${input_csv}" "${output_csv}" 2>/dev/null || true
    else
        > "${output_csv}"
    fi
}

# Calculate averages across runs
calculate_averages() {
    local rerun_dir="$1"
    local output_file="$2"

    # Calculate averages across all runs
    awk -F',' '
    BEGIN {
        OFS=","
        print "Model,BS,Avg_Eager,Avg_Inductor"
    }
    {
        key = $1 "," $2
        count[key]++
        eager_sum[key] += $3
        inductor_sum[key] += $4
    }
    END {
        for (key in count) {
            split(key, parts, ",")
            model = parts[1]
            bs = parts[2]
            if (count[key] > 0) {
                avg_eager = eager_sum[key] / count[key]
                avg_inductor = inductor_sum[key] / count[key]
                print model, bs, avg_eager, avg_inductor
            }
        }
    }' "${rerun_dir}"/run_*/latency.csv 2>/dev/null > "${output_file}"

    # Count models processed
    local model_count=$(awk 'NR>1 {count++} END {print count}' "${output_file}" 2>/dev/null || echo "0")
    log_debug "Calculated averages for ${model_count} models"
}

# Step 2: Rerun target 3 times
rerun_target() {
    log_info "Step 2: Rerunning target environment ${RERUN_COUNT} times..."

    local target_rerun_dir="${WORKSPACE_DIR}/target_rerun"
    mkdir -p "${target_rerun_dir}"

    # Read models with regression from initial check
    local regression_file="${OUTPUT_DIR}/initial_regression.csv"
    if [[ ! -f "${regression_file}" ]]; then
        log_error "Initial regression file not found"
        return 1
    fi

    # Extract models to rerun
    local models_to_rerun=()
    while IFS=',' read -r model bs ref_e ref_i new_e new_i eager_ratio inductor_ratio eager_reg inductor_reg reg_type notes; do
        [[ "$model" == "Model" ]] && continue
        if [[ "$eager_reg" == "YES" || "$inductor_reg" == "YES" ]]; then
            models_to_rerun+=("${model}|${bs}")
        fi
    done < "${regression_file}"

    if [[ ${#models_to_rerun[@]} -eq 0 ]]; then
        log_info "No models to rerun"
        return 0
    fi

    # Remove duplicates
    IFS=$'\n' unique_models=($(sort -u <<<"${models_to_rerun[*]}"))
    unset IFS

    log_info "Rerunning ${#unique_models[@]} models in target environment"

    # Run target rerun multiple times
    for ((run=1; run<=RERUN_COUNT; run++)); do
        log_info "Target rerun ${run}/${RERUN_COUNT}"
        local run_dir="${target_rerun_dir}/run_${run}"
        mkdir -p "${run_dir}"

        local run_log="${run_dir}/run.log"
        local results_csv="${run_dir}/results.csv"

        # Run benchmark
        run_benchmark "${unique_models[@]}" "${run_dir}" "${results_csv}" "${run_log}"

        # Process results
        process_rerun_results "${results_csv}" "${run_dir}/latency.csv"
    done

    # Calculate averages across all runs
    calculate_averages "${target_rerun_dir}" "${target_rerun_dir}/averages.csv"

    # Verify averages were calculated
    if [[ -f "${target_rerun_dir}/averages.csv" ]]; then
        local avg_count=$(awk 'NR>1 {count++} END {print count}' "${target_rerun_dir}/averages.csv" 2>/dev/null || echo "0")
        log_info "Calculated averages for ${avg_count} models"
    fi

    log_success "Target rerun completed"
    return 0
}

# Step 3: Copy target python env and setup baseline env
setup_baseline_env() {
    log_info "Step 3: Setting up baseline environment..."

    BASELINE_ENV_DIR="${WORKSPACE_DIR}/baseline_env"
    mkdir -p "${BASELINE_ENV_DIR}"

    # Get current environment info
    log_info "Current environment information:"
    "${PYTHON_EXECUTABLE}" --version
    "${PYTHON_EXECUTABLE}" -m pip --version 2>/dev/null || "${PYTHON_EXECUTABLE}" -m pip3 --version

    # Save current environment state
    local env_info_file="${BASELINE_ENV_DIR}/environment_info.txt"
    {
        echo "Python version:"
        "${PYTHON_EXECUTABLE}" --version
        echo -e "\nPip version:"
        "${PYTHON_EXECUTABLE}" -m pip --version 2>/dev/null || "${PYTHON_EXECUTABLE}" -m pip3 --version
        echo -e "\nInstalled packages:"
        "${PYTHON_EXECUTABLE}" -m pip list 2>/dev/null || "${PYTHON_EXECUTABLE}" -m pip3 list 2>/dev/null
    } > "${env_info_file}"

    # Copy environment based on selected method
    case "${ENV_COPY_METHOD}" in
        conda)
            setup_conda_env
            ;;
        venv)
            setup_venv
            ;;
        pipenv)
            setup_pipenv
            ;;
        *)
            log_error "Unsupported environment method: ${ENV_COPY_METHOD}"
            return 1
            ;;
    esac

    log_success "Baseline environment setup completed"
    return 0
}

setup_conda_env() {
    if ! command -v conda &> /dev/null; then
        log_error "Conda not found. Please install conda or use a different method."
        return 1
    fi

    local env_name="baseline_env_$(date +%s)"
    log_info "Creating conda environment: ${env_name}"

    # Create environment
    if conda info --envs | grep -q "^base"; then
        conda create --name "${env_name}" --clone base -y || {
            local python_version=$("${PYTHON_EXECUTABLE}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.8")
            conda create --name "${env_name}" python="${python_version}" -y
        }
    else
        local python_version=$("${PYTHON_EXECUTABLE}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.8")
        conda create --name "${env_name}" python="${python_version}" -y
    fi

    # Save environment spec
    conda list --name "${env_name}" > "${BASELINE_ENV_DIR}/conda_packages.txt" 2>/dev/null

    # Set environment variables for baseline runs
    export BASELINE_PYTHON="conda run -n ${env_name} python"
    export BASELINE_PIP="conda run -n ${env_name} pip"

    log_info "Conda environment created: ${env_name}"
}

setup_venv() {
    log_info "Creating virtual environment..."

    "${PYTHON_EXECUTABLE}" -m venv "${BASELINE_ENV_DIR}/venv"

    local venv_python="${BASELINE_ENV_DIR}/venv/bin/python"
    local venv_pip="${BASELINE_ENV_DIR}/venv/bin/pip"

    # Check if pip exists in venv
    if [[ ! -f "${venv_pip}" ]]; then
        venv_pip="${BASELINE_ENV_DIR}/venv/bin/pip3"
    fi

    # Upgrade pip
    "${venv_pip}" install --upgrade pip

    # Install packages from current environment
    log_info "Installing packages from current environment..."
    "${PYTHON_EXECUTABLE}" -m pip freeze 2>/dev/null > "${BASELINE_ENV_DIR}/requirements.txt" ||
    "${PYTHON_EXECUTABLE}" -m pip3 freeze 2>/dev/null > "${BASELINE_ENV_DIR}/requirements.txt"

    if [[ -s "${BASELINE_ENV_DIR}/requirements.txt" ]]; then
        "${venv_pip}" install -r "${BASELINE_ENV_DIR}/requirements.txt"
    else
        log_warning "No requirements found, installing basic packages"
        "${venv_pip}" install torch torchvision numpy pandas
    fi

    # Set environment variables
    export BASELINE_PYTHON="${venv_python}"
    export BASELINE_PIP="${venv_pip}"

    log_info "Virtual environment created"
}

setup_pipenv() {
    log_info "Setting up Pipenv environment..."

    if ! command -v pipenv &> /dev/null; then
        log_error "Pipenv not found. Please install pipenv or use a different method."
        return 1
    fi

    cd "${BASELINE_ENV_DIR}"

    # Initialize pipenv
    pipenv --python "${PYTHON_EXECUTABLE}"

    # Install packages from current environment
    "${PYTHON_EXECUTABLE}" -m pip freeze 2>/dev/null > "${BASELINE_ENV_DIR}/requirements.txt" ||
    "${PYTHON_EXECUTABLE}" -m pip3 freeze 2>/dev/null > "${BASELINE_ENV_DIR}/requirements.txt"

    if [[ -s "${BASELINE_ENV_DIR}/requirements.txt" ]]; then
        pipenv install -r "${BASELINE_ENV_DIR}/requirements.txt"
    else
        log_warning "No requirements found, installing basic packages"
        pipenv install torch torchvision numpy pandas
    fi

    # Set environment variables
    export BASELINE_PYTHON="pipenv run python"
    export BASELINE_PIP="pipenv run pip"

    log_info "Pipenv environment created"
}

# Step 4: Rerun baseline 3 times
rerun_baseline() {
    log_info "Step 4: Rerunning baseline environment ${RERUN_COUNT} times..."

    if [[ -z "${BASELINE_PYTHON}" ]] || [[ "${BASELINE_PYTHON}" == "python" ]]; then
        log_warning "Baseline Python not properly set, using current Python"
        export BASELINE_PYTHON="${PYTHON_EXECUTABLE}"
    fi

    local baseline_rerun_dir="${WORKSPACE_DIR}/baseline_rerun"
    mkdir -p "${baseline_rerun_dir}"

    # Read models from target rerun
    local target_models_file="${WORKSPACE_DIR}/target_rerun/averages.csv"
    if [[ ! -f "${target_models_file}" ]]; then
        log_error "Target rerun averages not found"
        return 1
    fi

    # Extract models to rerun in baseline
    local models_to_rerun=()
    while IFS=',' read -r model bs eager inductor; do
        [[ "$model" == "Model" ]] && continue
        models_to_rerun+=("${model}|${bs}")
    done < "${target_models_file}"

    if [[ ${#models_to_rerun[@]} -eq 0 ]]; then
        log_info "No models to rerun in baseline"
        return 0
    fi

    log_info "Rerunning ${#models_to_rerun[@]} models in baseline environment"

    # Test baseline Python
    log_debug "Testing baseline Python: ${BASELINE_PYTHON}"
    if ! ${BASELINE_PYTHON} --version &>/dev/null; then
        log_error "Baseline Python is not working: ${BASELINE_PYTHON}"
        return 1
    fi

    # Run baseline rerun multiple times
    for ((run=1; run<=RERUN_COUNT; run++)); do
        log_info "Baseline rerun ${run}/${RERUN_COUNT}"
        local run_dir="${baseline_rerun_dir}/run_${run}"
        mkdir -p "${run_dir}"

        local run_log="${run_dir}/run.log"
        local results_csv="${run_dir}/results.csv"

        # Run benchmark with baseline Python
        local original_python="${PYTHON_EXECUTABLE}"
        PYTHON_EXECUTABLE="${BASELINE_PYTHON}"
        run_benchmark "${models_to_rerun[@]}" "${run_dir}" "${results_csv}" "${run_log}"
        PYTHON_EXECUTABLE="${original_python}"

        # Process results
        process_rerun_results "${results_csv}" "${run_dir}/latency.csv"
    done

    # Calculate averages across all runs
    calculate_averages "${baseline_rerun_dir}" "${baseline_rerun_dir}/averages.csv"

    # Verify averages were calculated
    if [[ -f "${baseline_rerun_dir}/averages.csv" ]]; then
        local avg_count=$(awk 'NR>1 {count++} END {print count}' "${baseline_rerun_dir}/averages.csv" 2>/dev/null || echo "0")
        log_info "Calculated averages for ${avg_count} models"
    fi

    log_success "Baseline rerun completed"
    return 0
}

# Step 5: Check the 3 avg regression or not
check_average_regression() {
    log_info "Step 5: Checking average regression..."

    local target_avg_file="${WORKSPACE_DIR}/target_rerun/averages.csv"
    local baseline_avg_file="${WORKSPACE_DIR}/baseline_rerun/averages.csv"

    if [[ ! -f "${target_avg_file}" ]] || [[ ! -f "${baseline_avg_file}" ]]; then
        log_error "Average files not found"
        log_error "Target: ${target_avg_file}"
        log_error "Baseline: ${baseline_avg_file}"
        return 1
    fi

    # Compare averages
    local comparison_file="${OUTPUT_DIR}/average_comparison.csv"

    awk -F',' -v threshold="${REGRESSION_THRESHOLD}" '
    BEGIN {
        OFS=","
        print "Model,BS,Target_Eager,Target_Inductor,Baseline_Eager,Baseline_Inductor,Eager_Ratio,Inductor_Ratio,Eager_Reg,Inductor_Reg"
    }
    # Read target averages
    NR == FNR && FNR > 1 {
        key = $1 "," $2
        target_eager[key] = $3 + 0
        target_inductor[key] = $4 + 0
        next
    }
    # Read baseline averages
    FNR > 1 {
        model = $1
        bs = $2
        baseline_eager = $3 + 0
        baseline_inductor = $4 + 0
        key = model "," bs

        if (key in target_eager) {
            t_eager = target_eager[key]
            t_inductor = target_inductor[key]

            # Calculate ratios (target/baseline, higher is better)
            eager_ratio = 0
            inductor_ratio = 0
            eager_reg = "NO"
            inductor_reg = "NO"

            if (baseline_eager > 0 && t_eager > 0) {
                eager_ratio = t_eager / baseline_eager
                if (eager_ratio < threshold) {
                    eager_reg = "YES"
                }
            }

            if (baseline_inductor > 0 && t_inductor > 0) {
                inductor_ratio = t_inductor / baseline_inductor
                if (inductor_ratio < threshold) {
                    inductor_reg = "YES"
                }
            }

            printf "%s,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%s,%s\n",
                model, bs, t_eager, t_inductor, baseline_eager, baseline_inductor,
                eager_ratio, inductor_ratio, eager_reg, inductor_reg
        }
    }
    ' "${target_avg_file}" "${baseline_avg_file}" > "${comparison_file}"

    # Count regressions
    local total=$(awk 'NR>1 {count++} END {print count}' "${comparison_file}" 2>/dev/null || echo "0")
    local regressions=$(awk -F',' 'NR>1 && ($9 == "YES" || $10 == "YES") {count++} END {print count}' "${comparison_file}" 2>/dev/null || echo "0")

    log_info "Average comparison results:"
    log_info "  Total models compared: ${total}"
    log_info "  Regressions in averages: ${regressions}"

    if [[ "${regressions}" -gt 0 ]]; then
        log_warning "Average check: ${regressions} regression(s) confirmed"
        REGRESSION_CONFIRMED=true
        return 1
    else
        log_success "Average check: No regressions confirmed"
        return 0
    fi
}

# Step 6: Generate final verification report
generate_final_report() {
    log_info "Step 6: Generating final verification report..."

    local report_file="${OUTPUT_DIR}/final_verification_report.md"

    {
        echo "# Regression Verification Report"
        echo ""
        echo "## Executive Summary"
        echo ""
        if [[ "${REGRESSION_CONFIRMED}" == "true" ]]; then
            echo "❌ **REGRESSIONS CONFIRMED** - Performance degradation detected"
        else
            echo "✅ **NO REGRESSIONS CONFIRMED** - All tests passed"
        fi
        echo ""
        echo "## Test Details"
        echo ""
        echo "| Parameter | Value |"
        echo "|-----------|-------|"
        echo "| Test Date | $(date) |"
        echo "| New Results | \`${NEW_RESULTS_DIR}\` |"
        echo "| Reference Results | \`${REFERENCE_DIR}\` |"
        echo "| Regression Threshold | ${REGRESSION_THRESHOLD} |"
        echo "| Rerun Count | ${RERUN_COUNT} |"
        echo "| Environment Method | ${ENV_COPY_METHOD} |"
        echo "| Python Executable | \`${PYTHON_EXECUTABLE}\` |"
        echo ""
        echo "## Test Steps Summary"
        echo ""
        echo "### Step 1: Initial Regression Check"
        echo "- Compared new results with reference"
        echo "- Generated: \`initial_regression.csv\`"
        echo ""
        echo "### Step 2: Target Environment Rerun"
        echo "- Rerun problematic models ${RERUN_COUNT} times in current environment"
        echo "- Results: \`${WORKSPACE_DIR}/target_rerun/\`"
        echo ""
        echo "### Step 3: Baseline Environment Setup"
        echo "- Created baseline environment using ${ENV_COPY_METHOD}"
        echo "- Environment: \`${BASELINE_ENV_DIR}\`"
        echo ""
        echo "### Step 4: Baseline Environment Rerun"
        echo "- Rerun same models ${RERUN_COUNT} times in baseline environment"
        echo "- Results: \`${WORKSPACE_DIR}/baseline_rerun/\`"
        echo ""
        echo "### Step 5: Average Regression Check"
        echo "- Compared average performance between target and baseline"
        echo "- Generated: \`average_comparison.csv\`"
        echo ""
        echo "## Results Analysis"
        echo ""

        # Include regression details if any
        local comparison_file="${OUTPUT_DIR}/average_comparison.csv"
        local initial_regression_file="${OUTPUT_DIR}/initial_regression.csv"

        # Get initial regression count
        local initial_regressions=0
        if [[ -f "${initial_regression_file}" ]]; then
            initial_regressions=$(awk -F',' 'NR>1 && ($9 == "YES" || $10 == "YES") {count++} END {print count}' "${initial_regression_file}" 2>/dev/null || echo "0")
        fi

        echo "### Regression Statistics"
        echo "- Initial regressions detected: ${initial_regressions}"

        if [[ -f "${comparison_file}" ]]; then
            local final_regressions=$(awk -F',' 'NR>1 && ($9 == "YES" || $10 == "YES") {count++} END {print count}' "${comparison_file}" 2>/dev/null || echo "0")
            local total=$(awk 'NR>1 {count++} END {print count}' "${comparison_file}" 2>/dev/null || echo "0")

            echo "- Final regressions confirmed: ${final_regressions}"
            echo "- Total models compared: ${total}"

            if [[ "${total}" -gt 0 ]]; then
                local regression_rate=$(awk "BEGIN {printf \"%.1f%%\", ${final_regressions}/${total}*100}")
                echo "- Regression rate: ${regression_rate}"
            fi
            echo ""

            if [[ "${final_regressions}" -gt 0 ]]; then
                echo "### Models with Confirmed Regression"
                echo ""
                echo "| Model | Batch Size | Eager Ratio | Inductor Ratio | Regression Type |"
                echo "|-------|------------|-------------|----------------|-----------------|"
                awk -F',' '
                NR>1 && ($9 == "YES" || $10 == "YES") {
                    reg_type = ""
                    if ($9 == "YES") reg_type = "Eager"
                    if ($10 == "YES") {
                        if (reg_type != "") reg_type = reg_type " + "
                        reg_type = reg_type "Inductor"
                    }
                    printf "| %s | %s | %.3f | %.3f | %s |\n", $1, $2, $7, $8, reg_type
                }' "${comparison_file}" 2>/dev/null || echo "| Error parsing regression data | | | | |"
            fi
        else
            echo "- Final comparison: Not available"
            echo ""
        fi

        echo ""
        echo "## Files Generated"
        echo ""
        echo "| File | Description |"
        echo "|------|-------------|"
        echo "| \`initial_regression.csv\` | Initial regression analysis |"
        echo "| \`average_comparison.csv\` | Final average comparison |"
        echo "| \`workspace/\` | All intermediate results |"
        echo "| \`workspace/target_rerun/\` | Target environment rerun results |"
        echo "| \`workspace/baseline_rerun/\` | Baseline environment rerun results |"
        echo "| \`workspace/baseline_env/\` | Baseline environment setup |"
        echo ""
        echo "## Next Steps"
        echo ""
        if [[ "${REGRESSION_CONFIRMED}" == "true" ]]; then
            echo "1. Investigate confirmed regressions in \`average_comparison.csv\`"
            echo "2. Check individual run logs in workspace directories"
            echo "3. Review environment differences in \`workspace/baseline_env/environment_info.txt\`"
            echo "4. Consider if regression threshold needs adjustment"
            echo "5. Report to development team if regression is unexpected"
        else
            echo "1. No action required - all tests passed"
            echo "2. Consider lowering threshold for stricter checks"
            echo "3. Archive results for future reference"
        fi
        echo ""
        echo "---"
        echo "*Report generated by Enhanced Regression Checker*"

    } > "${report_file}"

    # Also create a summary text file
    local summary_file="${OUTPUT_DIR}/summary.txt"
    {
        echo "========================================"
        echo "FINAL REGRESSION VERIFICATION SUMMARY"
        echo "========================================"
        echo "Date: $(date)"
        echo "Status: $(if [[ "${REGRESSION_CONFIRMED}" == "true" ]]; then echo "FAIL - Regressions confirmed"; else echo "PASS - No regressions"; fi)"
        echo "Regression threshold: ${REGRESSION_THRESHOLD}"
        echo "Rerun count: ${RERUN_COUNT}"
        echo "Environment method: ${ENV_COPY_METHOD}"
        echo "Output directory: ${OUTPUT_DIR}"
        echo "========================================"

        # Show quick stats if available
        local comparison_file="${OUTPUT_DIR}/average_comparison.csv"
        if [[ -f "${comparison_file}" ]]; then
            echo ""
            echo "Quick Statistics:"
            local total=$(awk 'NR>1 {count++} END {print count}' "${comparison_file}" 2>/dev/null || echo "0")
            local regressions=$(awk -F',' 'NR>1 && ($9 == "YES" || $10 == "YES") {count++} END {print count}' "${comparison_file}" 2>/dev/null || echo "0")
            echo "Models compared: ${total}"
            echo "Confirmed regressions: ${regressions}"
        fi
        echo "========================================"
    } > "${summary_file}"

    # Display summary
    cat "${summary_file}"

    log_success "Final report generated: ${report_file}"
}

# Main execution function
main() {
    log_info "Starting enhanced regression verification"
    log_info "=========================================="

    # Validate arguments
    validate_arguments

    # Step 1: Initial regression check
    if ! check_initial_regression; then
        log_info "Proceeding with full verification due to initial regressions"
    else
        log_success "No regressions found in initial check"
        generate_final_report
        exit 0
    fi

    # Step 2: Rerun target environment
    log_info ""
    log_info "Step 2: Starting target rerun..."
    if ! rerun_target; then
        log_warning "Target rerun completed with warnings"
    fi

    # Step 3: Setup baseline environment
    log_info ""
    log_info "Step 3: Setting up baseline environment..."
    if [[ "${AUTO_SETUP_ENV}" == "true" ]]; then
        if ! setup_baseline_env; then
            log_error "Failed to setup baseline environment"
            exit 1
        fi
    else
        log_info "Skipping automatic environment setup"
        export BASELINE_PYTHON="${PYTHON_EXECUTABLE}"
    fi

    # Step 4: Rerun baseline environment
    log_info ""
    log_info "Step 4: Starting baseline rerun..."
    if ! rerun_baseline; then
        log_warning "Baseline rerun completed with warnings"
    fi

    # Step 5: Check average regression
    log_info ""
    log_info "Step 5: Checking average regression..."
    if ! check_average_regression; then
        log_warning "Average regression check found issues"
    fi

    # Step 6: Generate final report
    log_info ""
    log_info "Step 6: Generating final report..."
    generate_final_report

    # Exit based on regression status
    if [[ "${FAIL_ON_REGRESSION}" == "true" ]] && [[ "${REGRESSION_CONFIRMED}" == "true" ]]; then
        log_error "Regression verification FAILED - confirmed regressions found"
        exit 1
    fi

    log_success ""
    log_success "Regression verification completed successfully"
    log_info "All results saved to: ${OUTPUT_DIR}"

    return 0
}

# Run main function
main "$@"
exit $?
