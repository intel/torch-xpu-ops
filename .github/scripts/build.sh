#!/bin/bash
set -xeo pipefail
export GIT_PAGER=cat

# ============================================
# Configuration and Defaults
# ============================================
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -A DEFAULTS=(
    ["WORKSPACE"]="/tmp/$(whoami)/pytorch-build"
    ["PYTORCH_REPO"]="https://github.com/pytorch/pytorch.git"
    ["PYTORCH_COMMIT"]="main"
    ["TORCH_XPU_OPS_REPO"]="https://github.com/intel/torch-xpu-ops.git"
    ["TORCH_XPU_OPS_COMMIT"]="main"
    ["BUILD_TYPE"]="release"
    ["BUILD_JOBS"]="$(nproc)"
)

# ============================================
# Global Variables
# ============================================
declare -gA REPO_COMMITS  # Store commit hashes for tracking

# ============================================
# Usage Documentation
# ============================================
show_usage() {
    cat << EOF
${SCRIPT_NAME} - Build PyTorch with XPU support

Usage:
    ${SCRIPT_NAME} [OPTIONS]

Options:
    --WORKSPACE=<path>            Working directory (default: ${DEFAULTS[WORKSPACE]})
    --PYTORCH_REPO=<url>          PyTorch repository URL (default: ${DEFAULTS[PYTORCH_REPO]})
    --PYTORCH_COMMIT=<ref>        PyTorch branch/tag/commit (default: ${DEFAULTS[PYTORCH_COMMIT]})
    --TORCH_XPU_OPS_REPO=<url>    Torch-XPU-Ops repository URL (default: ${DEFAULTS[TORCH_XPU_OPS_REPO]})
    --TORCH_XPU_OPS_COMMIT=<ref>  Torch-XPU-Ops branch/tag/commit or "pinned" (default: ${DEFAULTS[TORCH_XPU_OPS_COMMIT]})
    --ONEDNN_REPO=<url>           (Optional) oneDNN repository URL
    --ONEDNN_COMMIT=<ref>         (Optional) oneDNN branch/tag/commit
    --BUILD_TYPE=<type>           Build type: release or debug (default: ${DEFAULTS[BUILD_TYPE]})
    --BUILD_JOBS=<n>              Number of parallel build jobs (default: ${DEFAULTS[BUILD_JOBS]})
    -h, --help                    Show this help message

Environment Variables:
    XPU_ONEAPI_PATH               Path to existing oneAPI installation
    GITHUB_EVENT_NAME             If set to "pull_request", uses local torch-xpu-ops
    INPUTS_COMPONENT              Component being built (for oneDNN patching)

Examples:
    ${SCRIPT_NAME} --WORKSPACE=/build --PYTORCH_COMMIT=v2.5.0
    ${SCRIPT_NAME} --BUILD_TYPE=debug --BUILD_JOBS=4
    PYTORCH_COMMIT=v2.5.0 TORCH_XPU_OPS_COMMIT=pinned ./${SCRIPT_NAME}
EOF
}

# ============================================
# Logging Functions
# ============================================
log_info() {
    echo "[INFO][$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_warning() {
    echo "[WARNING][$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

log_error() {
    echo "[ERROR][$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

log_success() {
    echo "[SUCCESS][$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_step() {
    echo "================================================================================"
    echo "STEP: $*"
    echo "================================================================================"
}

# ============================================
# Utility Functions
# ============================================
validate_directory() {
    local dir="$1"
    local purpose="$2"

    if [[ ! -d "$(dirname "${dir}")" ]]; then
        mkdir -p "$(dirname "${dir}")" || {
            log_error "Failed to create parent directory for ${purpose}: $(dirname "${dir}")"
            return 1
        }
    fi

    if [[ ! -w "$(dirname "${dir}")" ]]; then
        log_error "No write permission for ${purpose} directory: $(dirname "${dir}")"
        return 1
    fi
    return 0
}

cleanup_workspace() {
    local workspace="$1"

    log_info "Cleaning workspace: ${workspace}"
    rm -rf "${workspace}/pytorch" "${workspace}/torch-xpu-ops" 2>/dev/null || true
}

get_pinned_commit() {
    local pytorch_dir="$1"
    local commit_file="${pytorch_dir}/third_party/xpu.txt"

    if [[ -f "${commit_file}" ]]; then
        cat "${commit_file}" | head -1 | tr -d '[:space:]'
    else
        log_warning "xpu.txt not found, using default commit"
        echo "main"
    fi
}

# ============================================
# Git Operations
# ============================================
clone_repository() {
    local repo_url="$1"
    local target_dir="$2"
    local commit_ref="${3:-}"
    local repo_name="$4"

    log_step "Cloning ${repo_name}"

    # Check if already exists and is git repo
    if [[ -d "${target_dir}/.git" ]]; then
        log_info "Git repository already exists at ${target_dir}, skipping clone"
        cd "${target_dir}" || return 1
        git fetch --all --tags || {
            log_warning "Failed to fetch updates for existing repository"
        }
    else
        log_info "Cloning from ${repo_url} into ${target_dir}"
        if ! git clone --depth 1 --recurse-submodules "${repo_url}" "${target_dir}"; then
            log_error "Failed to clone ${repo_name}"
            return 1
        fi
        cd "${target_dir}" || return 1
    fi

    # Checkout specific commit if provided
    if [[ -n "${commit_ref}" && "${commit_ref}" != "main" ]]; then
        log_info "Checking out ${repo_name} commit: ${commit_ref}"
        if ! git checkout "${commit_ref}" 2>/dev/null; then
            log_info "Commit not found locally, fetching..."
            git fetch --all --tags
            if ! git checkout "${commit_ref}"; then
                log_error "Failed to checkout ${commit_ref} for ${repo_name}"
                return 1
            fi
        fi
    fi

    # Update submodules for specific commit
    if [[ -n "${commit_ref}" && "${commit_ref}" != "main" ]]; then
        log_info "Updating submodules for specific commit"
        git submodule update --init --recursive --depth 1 || {
            log_warning "Failed to update submodules with depth, trying without depth limit"
            git submodule update --init --recursive
        }
    fi

    # Store repository info
    log_info "${repo_name} Repository Info:"
    git remote -v
    git branch -a || true
    git show -s --oneline

    # Save commit hash
    local commit_hash=$(git rev-parse HEAD)
    REPO_COMMITS["${repo_name}"]="${commit_hash}"
    echo "${commit_hash}" > "${target_dir}.commit"
    log_info "${repo_name} commit hash: ${commit_hash}"

    cd - >/dev/null || return 1
    return 0
}

# ============================================
# Dependency Management
# ============================================
setup_oneapi_packages() {
    log_step "Setting up oneAPI dependencies"

    if [[ -n "${XPU_ONEAPI_PATH}" ]]; then
        log_info "XPU_ONEAPI_PATH is set to: ${XPU_ONEAPI_PATH}"
        log_info "Using existing oneAPI installation"

        # Still install MKL packages
        python -m pip install --no-deps mkl-static==2025.3.0 mkl-include==2025.3.0 || {
            log_error "Failed to install MKL packages"
            return 1
        }
        return 0
    fi

    log_info "Installing oneAPI packages via pip"

    # Define packages with version ranges for flexibility
    local packages=(
        intel-cmplr-lib-rt==2025.3.1
        intel-cmplr-lib-ur==2025.3.1
        intel-cmplr-lic-rt==2025.3.1
        intel-sycl-rt==2025.3.1
        oneccl-devel==2021.17.1
        oneccl==2021.17.1
        impi-rt==2021.17.0
        onemkl-sycl-blas==2025.3.0
        onemkl-sycl-dft==2025.3.0
        onemkl-sycl-lapack==2025.3.0
        onemkl-sycl-rng==2025.3.0
        onemkl-sycl-sparse==2025.3.0
        dpcpp-cpp-rt==2025.3.1
        intel-opencl-rt==2025.3.1
        mkl==2025.3.0
        intel-openmp==2025.3.1
        tbb==2022.3.0
        tcmlib==1.4.1
        umf==1.0.2
        intel-pti==0.15.0
    )

    # Install packages individually for better error reporting
    for package in "${packages[@]}"; do
        log_info "Installing: ${package}"
        if ! python -m pip install --no-deps "${package}"; then
            log_warning "Failed to install ${package}, attempting with dependency resolution"
            python -m pip install "${package}" || {
                log_error "Critical failure installing ${package}"
                return 1
            }
        fi
    done

    # Export for pytorch build
    export PYTORCH_EXTRA_INSTALL_REQUIREMENTS="${packages[*]// /|}"
    log_info "Set PYTORCH_EXTRA_INSTALL_REQUIREMENTS"

    return 0
}

install_build_dependencies() {
    local pytorch_dir="$1"

    log_step "Installing build dependencies"

    cd "${pytorch_dir}" || return 1

    # Upgrade pip first
    python -m pip install --upgrade pip wheel setuptools

    # Install requirements
    if [[ -f "requirements.txt" ]]; then
        log_info "Installing PyTorch requirements"
        python -m pip install -r requirements.txt
    fi

    # Install additional tools
    local tools=("requests" "ninja" "patchelf")
    for tool in "${tools[@]}"; do
        if ! python -m pip show "${tool}" >/dev/null 2>&1; then
            log_info "Installing ${tool}"
            python -m pip install "${tool}"
        fi
    done

    cd - >/dev/null || return 1
    return 0
}

# ============================================
# Patching and Configuration
# ============================================
apply_patches() {
    local pytorch_dir="$1"

    log_step "Applying patches and configuration"

    cd "${pytorch_dir}" || return 1

    # Apply torch-xpu-ops patches if script exists
    local patch_script="third_party/torch-xpu-ops/.github/scripts/apply_torch_pr.py"
    if [[ -f "${patch_script}" ]]; then
        log_info "Applying torch-xpu-ops patches"
        if ! python "${patch_script}"; then
            log_warning "Patch script returned non-zero, but continuing build"
        fi
    else
        log_warning "Patch script not found: ${patch_script}"
    fi

    # Patch oneDNN configuration if specified
    if [[ -n "${ONEDNN_REPO}" && -n "${ONEDNN_COMMIT}" ]]; then
        log_info "Configuring oneDNN to use: ${ONEDNN_REPO}@${ONEDNN_COMMIT}"
        local cmake_file="cmake/Modules/FindMKLDNN.cmake"
        if [[ -f "${cmake_file}" ]]; then
            cp "${cmake_file}" "${cmake_file}.bak"
            sed -i \
                -e "s|^GIT_REPOSITORY .*|GIT_REPOSITORY ${ONEDNN_REPO}|" \
                -e "s|^GIT_TAG .*|GIT_TAG ${ONEDNN_COMMIT}|" \
                "${cmake_file}" || {
                log_error "Failed to patch oneDNN configuration"
                return 1
            }
            log_info "oneDNN configuration updated"
        else
            log_warning "FindMKLDNN.cmake not found, skipping oneDNN configuration"
        fi
    fi

    # Patch CMakeLists.txt to avoid git checkout during build
    local cmake_lists="caffe2/CMakeLists.txt"
    if [[ -f "${cmake_lists}" ]]; then
        cp "${cmake_lists}" "${cmake_lists}.bak"
        sed -i 's/checkout --quiet .*TORCH_XPU_OPS_COMMIT}/log -n 1/g' \
            "${cmake_lists}" || {
            log_error "Failed to patch CMakeLists.txt"
            return 1
        }
        log_info "CMakeLists.txt patched"
    fi

    # Show changes
    log_info "Applied patches summary:"
    git diff --stat 2>/dev/null || true

    cd - >/dev/null || return 1
    return 0
}

# ============================================
# Build Functions
# ============================================
build_pytorch() {
    local pytorch_dir="$1"
    local build_type="${2:-release}"
    local build_jobs="${3:-$(nproc)}"

    log_step "Building PyTorch (${build_type} mode)"

    cd "${pytorch_dir}" || return 1

    # Initialize submodules
    log_info "Initializing git submodules"
    git submodule sync --recursive
    git submodule update --init --recursive --jobs="${build_jobs}" || {
        log_warning "Submodule update had issues, but continuing"
    }

    # Set build environment
    export USE_STATIC_MKL=1
    export MAX_JOBS="${build_jobs}"
    export CMAKE_BUILD_TYPE="${build_type^}"  # Capitalize first letter

    # Additional build flags
    local build_flags=()
    if [[ "${build_type}" == "debug" ]]; then
        build_flags+=("DEBUG=1" "REL_WITH_DEB_INFO=1")
        export DEBUG=1
    fi

    # Build command
    local build_cmd="WERROR=1 python setup.py bdist_wheel ${build_flags[*]}"

    log_info "Build command: ${build_cmd}"
    log_info "Build environment:"
    log_info "  MAX_JOBS=${MAX_JOBS}"
    log_info "  USE_STATIC_MKL=${USE_STATIC_MKL}"
    log_info "  CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    log_info "  PYTORCH_EXTRA_INSTALL_REQUIREMENTS=${PYTORCH_EXTRA_INSTALL_REQUIREMENTS:-not set}"

    # Execute build
    local build_log="${pytorch_dir}/build.log"
    log_info "Starting build (logs: ${build_log})"

    if ! eval "${build_cmd}" 2>&1 | tee "${build_log}"; then
        log_error "PyTorch build failed"
        log_info "Last 50 lines of build log:"
        tail -50 "${build_log}"

        # Check for common issues
        if grep -q "Out of memory" "${build_log}"; then
            log_error "Build failed due to out of memory. Try reducing BUILD_JOBS."
        fi
        if grep -q "No space left on device" "${build_log}"; then
            log_error "Build failed due to disk space. Clean up and try again."
        fi

        return 1
    fi

    log_success "PyTorch build completed successfully"
    log_info "Build artifacts in: ${pytorch_dir}/dist/"

    cd - >/dev/null || return 1
    return 0
}

post_process_wheel() {
    local pytorch_dir="$1"
    local workspace="$2"

    log_step "Post-processing wheel"

    cd "${pytorch_dir}" || return 1

    # Find the built wheel
    local wheel_pattern="dist/torch-*.whl"
    local wheels=(${wheel_pattern})

    if [[ ${#wheels[@]} -eq 0 ]]; then
        log_error "No wheel files found matching: ${wheel_pattern}"
        return 1
    fi

    local wheel_file="${wheels[0]}"
    log_info "Found wheel: ${wheel_file}"

    # Apply rpath adjustment if script exists
    local rpath_script="third_party/torch-xpu-ops/.github/scripts/rpath.sh"
    if [[ -f "${rpath_script}" && -x "${rpath_script}" ]]; then
        log_info "Running rpath adjustment script"
        if ! bash "${rpath_script}" "${wheel_file}"; then
            log_warning "rpath script failed, but continuing"
        fi
    else
        log_warning "rpath script not found or not executable: ${rpath_script}"
    fi

    # Find modified wheel
    local modified_pattern="tmp/torch-*.whl"
    local modified_wheels=(${modified_pattern})
    local target_wheel

    if [[ ${#modified_wheels[@]} -gt 0 ]]; then
        target_wheel="${modified_wheels[0]}"
        log_info "Using modified wheel: ${target_wheel}"
    else
        target_wheel="${wheel_file}"
        log_info "Using original wheel: ${target_wheel}"
    fi

    # Install the wheel
    log_info "Installing wheel: ${target_wheel}"
    if ! python -m pip install --force-reinstall "${target_wheel}"; then
        log_error "Failed to install wheel"
        return 1
    fi

    # Copy to workspace
    local final_wheel="${workspace}/$(basename "${target_wheel}")"
    cp "${target_wheel}" "${final_wheel}" || {
        log_error "Failed to copy wheel to workspace"
        return 1
    }

    log_success "Wheel available at: ${final_wheel}"
    echo "${final_wheel}" > "${workspace}/BUILT_WHEEL.txt"

    cd - >/dev/null || return 1
    return 0
}

# ============================================
# Verification
# ============================================
verify_installation() {
    local workspace="$1"

    log_step "Verifying installation"

    # Collect environment info
    if [[ -f "${workspace}/pytorch/torch/utils/collect_env.py" ]]; then
        log_info "Collecting environment information"
        python "${workspace}/pytorch/torch/utils/collect_env.py"
    fi

    # Verify PyTorch installation
    log_info "Verifying PyTorch installation"

    local verification_script=$(cat << 'EOF'
import sys
import torch

print("=" * 70)
print("PyTorch Installation Verification")
print("=" * 70)

# Basic info
print(f"PyTorch Version: {torch.__version__}")
print(f"PyTorch Path: {torch.__file__}")
print(f"Python Version: {sys.version}")

# Build configuration
print("\nBuild Configuration:")
try:
    print(torch.__config__.show())
except Exception as e:
    print(f"Failed to get config: {e}")

# XPU support
print("\nXPU Support:")
try:
    print(f"XPU Compiled: {torch.xpu._is_compiled()}")
    print(f"XPU Available: {torch.xpu.is_available()}")
    print(f"XPU Device Count: {torch.xpu.device_count()}")

    if torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            props = torch.xpu.get_device_properties(i)
            print(f"  Device {i}: {props.name}")
            print(f"    Compute Capability: {props.major}.{props.minor}")
            print(f"    Total Memory: {props.total_memory / 1024**3:.2f} GB")
except Exception as e:
    print(f"XPU check failed: {e}")

print("=" * 70)
EOF
    )

    python -c "${verification_script}"

    # Final verification
    local xpu_compiled=$(python -c 'import torch; print(torch.xpu._is_compiled())' 2>/dev/null || echo "false")

    if [[ "${xpu_compiled,,}" != "true" ]]; then
        log_error "CRITICAL: XPU is not compiled in PyTorch!"
        return 1
    fi

    log_success "Verification passed!"
    return 0
}

# ============================================
# Main Function
# ============================================
main() {
    log_step "Starting PyTorch XPU Build"
    log_info "Script: ${SCRIPT_NAME}"
    log_info "Start time: $(date)"
    log_info "User: $(whoami)"
    log_info "Hostname: $(hostname)"

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                show_usage
                exit 0
                ;;
            --*=*)
                local arg="$1"
                local var_name="${arg#--}"
                local var_value="${var_name#*=}"
                var_name="${var_name%%=*}"
                var_name="${var_name^^}"  # Convert to uppercase
                var_name="${var_name//-/_}"  # Replace hyphens with underscores

                export "${var_name}"="${var_value}"
                log_info "Set ${var_name}=${var_value}"
                shift
                ;;
            *)
                log_error "Unknown argument: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Apply defaults for unset variables
    for key in "${!DEFAULTS[@]}"; do
        if [[ -z "${!key:-}" ]]; then
            export "${key}"="${DEFAULTS[$key]}"
            log_info "Using default ${key}=${DEFAULTS[$key]}"
        fi
    done

    # Validate and setup workspace
    WORKSPACE=$(realpath -m "${WORKSPACE}")
    validate_directory "${WORKSPACE}" "workspace" || exit 1
    cleanup_workspace "${WORKSPACE}"
    log_info "Workspace: ${WORKSPACE}"

    # Setup PyTorch
    clone_repository \
        "${PYTORCH_REPO}" \
        "${WORKSPACE}/pytorch" \
        "${PYTORCH_COMMIT}" \
        "PyTorch" || exit 1

    # Setup torch-xpu-ops
    local torch_xpu_ops_commit="${TORCH_XPU_OPS_COMMIT}"
    if [[ "${torch_xpu_ops_commit,,}" == "pinned" ]]; then
        torch_xpu_ops_commit=$(get_pinned_commit "${WORKSPACE}/pytorch")
        log_info "Using pinned commit: ${torch_xpu_ops_commit}"
    fi

    if [[ "${GITHUB_EVENT_NAME:-}" == "pull_request" ]] && [[ -d "${WORKSPACE}/torch-xpu-ops" ]]; then
        log_info "PR build detected, using local torch-xpu-ops"
        cp -r "${WORKSPACE}/torch-xpu-ops" "${WORKSPACE}/pytorch/third_party/" || {
            log_error "Failed to copy local torch-xpu-ops"
            exit 1
        }
    else
        clone_repository \
            "${TORCH_XPU_OPS_REPO}" \
            "${WORKSPACE}/pytorch/third_party/torch-xpu-ops" \
            "${torch_xpu_ops_commit}" \
            "torch-xpu-ops" || exit 1
    fi

    # Install dependencies
    install_build_dependencies "${WORKSPACE}/pytorch" || exit 1
    setup_oneapi_packages || exit 1

    # Apply patches
    apply_patches "${WORKSPACE}/pytorch" || exit 1

    # Build PyTorch
    build_pytorch "${WORKSPACE}/pytorch" "${BUILD_TYPE}" "${BUILD_JOBS}" || exit 1

    # Post-process
    post_process_wheel "${WORKSPACE}/pytorch" "${WORKSPACE}" || exit 1

    # Verify
    verify_installation "${WORKSPACE}" || exit 1

    # Summary
    log_step "Build Summary"
    log_success "Build completed successfully!"
    log_info "Build artifacts in: ${WORKSPACE}"
    log_info "Built wheel: $(cat "${workspace}/BUILT_WHEEL.txt" 2>/dev/null || ls "${workspace}"/*.whl 2>/dev/null | head -1)"
    log_info "Repository commits:"
    for repo in "${!REPO_COMMITS[@]}"; do
        log_info "  ${repo}: ${REPO_COMMITS[$repo]}"
    done
    log_info "End time: $(date)"
    log_info "Total time: $SECONDS seconds"
}

# ============================================
# Script Entry Point
# ============================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    SECONDS=0
    trap 'log_error "Build interrupted at line $LINENO"; exit 1' INT TERM
    main "$@"
fi
