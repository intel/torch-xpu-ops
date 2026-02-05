#!/bin/bash

set -xeo pipefail

# Configuration for default repositories
declare -A DEFAULT_REPOS=(
    ["pytorch"]="https://github.com/pytorch/pytorch.git"
    ["torch-xpu-ops"]="https://github.com/intel/torch-xpu-ops.git"
    ["onednn"]="https://github.com/uxlfoundation/oneDNN.git"
    ["triton"]="https://github.com/intel/intel-xpu-backend-for-triton.git"
)

# Function to extract repo and commit from version string
extract_repo_commit() {
    local version="$1"
    local default_repo="$2"

    if [[ "${version}" =~ ^https://.*@ ]]; then
        # Extract repo and commit from format: https://github.com/repo.git@commit
        echo "${version%@*}"
        echo "${version##*@}"
    else
        # Use default repo and provided commit hash/version
        echo "${default_repo}"
        echo "${version}"
    fi
}

# Function to handle OneAPI installation
install_oneapi() {
    local download_url="$1"
    local install_dir="${2:-${HOME}/intel/oneapi}"

    echo "Installing OneAPI from: ${download_url}"

    # Clean up previous installations
    rm -rf ~/.intel ~/intel /opt/intel || true

    # Download and install OneAPI
    if ! wget -q -O oneapi.sh "${download_url}"; then
        echo "ERROR: Failed to download OneAPI installer from ${download_url}"
        exit 1
    fi

    if ! bash oneapi.sh -a -s --eula accept --action install --install-dir "${install_dir}"; then
        echo "ERROR: OneAPI installation failed"
        exit 1
    fi

    # Set environment variable
    echo "XPU_ONEAPI_PATH=${install_dir}" >> "${GITHUB_ENV}"
    echo "OneAPI installation completed successfully at ${install_dir}"
}

# Function to set environment variables
set_env_var() {
    local var_name="$1"
    local var_value="$2"

    echo "${var_name}=${var_value}" >> "${GITHUB_ENV}"
    echo "Set ${var_name}=${var_value}"
}

# Function to validate required variables
validate_inputs() {
    local required_vars=("INPUTS_COMPONENT")

    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            echo "ERROR: ${var} is not set"
            exit 1
        fi
    done

    if [[ -z "${RUN_TYPE:-}" ]]; then
        echo "INFO: RUN_TYPE is not set, defaulting to 'target'"
        RUN_TYPE="target"
    fi

    # Validate RUN_TYPE value
    if [[ "${RUN_TYPE}" != "target" && "${RUN_TYPE}" != "baseline" ]]; then
        echo "ERROR: Invalid RUN_TYPE '${RUN_TYPE}'. Must be 'target' or 'baseline'"
        exit 1
    fi
}

# Function to get component version
get_component_version() {
    local run_type="$1"

    if [[ "${run_type}" == "target" ]]; then
        echo "${INPUTS_TARGET:-}"
    else
        echo "${INPUTS_BASELINE:-}"
    fi
}

# Function to process git-based component
process_git_component() {
    local component="$1"
    local version="$2"
    local prefix="${3:-${component^^}}"

    local default_repo="${DEFAULT_REPOS[$component]}"
    if [[ -z "${default_repo}" ]]; then
        echo "ERROR: No default repository configured for component: ${component}"
        exit 1
    fi

    read -r REPO COMMIT <<< "$(extract_repo_commit "${version}" "${default_repo}")"

    # Set environment variables
    set_env_var "${prefix}_REPO" "${REPO}"
    set_env_var "${prefix}_COMMIT" "${COMMIT}"

    echo "${component^} Repo: ${REPO}"
    echo "${component^} Commit: ${COMMIT}"
}

# Main execution
main() {
    # Validate inputs
    validate_inputs

    # Determine component version based on test type
    COMPONENT_VERSION="$(get_component_version "${RUN_TYPE}")"

    # Validate component version is set
    if [[ -z "${COMPONENT_VERSION}" ]]; then
        echo "ERROR: COMPONENT_VERSION is not set for RUN_TYPE=${RUN_TYPE}"
        echo "Check INPUTS_TARGET or INPUTS_BASELINE environment variables"
        exit 1
    fi

    echo "Processing component: ${INPUTS_COMPONENT}"
    echo "Component version: ${COMPONENT_VERSION}"
    echo "Test type: ${RUN_TYPE}"

    # Process based on component type
    case "${INPUTS_COMPONENT}" in
        "pytorch")
            process_git_component "pytorch" "${COMPONENT_VERSION}" "PYTORCH"
            ;;

        "torch-xpu-ops")
            process_git_component "torch-xpu-ops" "${COMPONENT_VERSION}" "TORCH_XPU_OPS"
            ;;

        "onednn")
            process_git_component "onednn" "${COMPONENT_VERSION}" "ONEDNN"
            ;;

        "oneapi")
            install_oneapi "${COMPONENT_VERSION}"
            ;;

        "triton")
            process_git_component "triton" "${COMPONENT_VERSION}" "TRITON"
            ;;

        *)
            echo "ERROR: Unsupported component: ${INPUTS_COMPONENT}"
            echo "Available components: ${!DEFAULT_REPOS[*]} oneapi"
            exit 1
            ;;
    esac

    echo "Component processing completed successfully"
}

# Only run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
