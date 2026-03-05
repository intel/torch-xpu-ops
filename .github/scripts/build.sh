#!/bin/bash
# Enhanced build script for PyTorch with XPU support
# Usage:
#   ./build.sh [OPTIONS]
#   ./build.sh --WORKSPACE=<path> --PYTORCH=main --TORCH_XPU_OPS=main
set -xeuo pipefail
export GIT_PAGER=cat

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Default configurations
readonly DEFAULT_WORKSPACE="${WORKSPACE:-/tmp/pytorch_build}"
readonly DEFAULT_PYTORCH_REPO="https://github.com/pytorch/pytorch.git"
readonly DEFAULT_PYTORCH_COMMIT="main"
readonly DEFAULT_TORCH_XPU_OPS_REPO="https://github.com/intel/torch-xpu-ops.git"
readonly DEFAULT_TORCH_XPU_OPS_COMMIT="main"
readonly DEFAULT_ONEDNN_REPO="https://github.com/uxlfoundation/oneDNN.git"
readonly DEFAULT_ONEDNN_COMMIT="main"

# Set extra requirements for oneAPI components
export USE_STATIC_MKL=1
readonly DEFAULT_MKL_VERSION="2025.3.0"
if [[ -z "${XPU_ONEAPI_PATH:-}" ]]; then
    export PYTORCH_EXTRA_INSTALL_REQUIREMENTS=" \
        intel-cmplr-lib-rt==2025.3.2 | \
        intel-cmplr-lib-ur==2025.3.2 | \
        intel-cmplr-lic-rt==2025.3.2 | \
        intel-sycl-rt==2025.3.2 | \
        oneccl-devel==2021.17.2 | \
        oneccl==2021.17.2 | \
        impi-rt==2021.17.2 | \
        onemkl-sycl-blas==2025.3.1 | \
        onemkl-sycl-dft==2025.3.1 | \
        onemkl-sycl-lapack==2025.3.1 | \
        onemkl-sycl-rng==2025.3.1 | \
        onemkl-sycl-sparse==2025.3.1 | \
        dpcpp-cpp-rt==2025.3.2 | \
        intel-opencl-rt==2025.3.2 | \
        mkl==2025.3.1 | \
        intel-openmp==2025.3.2 | \
        tbb==2022.3.1 | \
        tcmlib==1.4.1 | \
        umf==1.0.3 | \
        intel-pti==0.16.0
fi

# Global variables
WORKSPACE=""
PYTORCH_REPO=""
PYTORCH_COMMIT=""
TORCH_XPU_OPS_REPO=""
TORCH_XPU_OPS_COMMIT=""
COMPONENT=""
COMPONENT_COMMIT=""
ONEDNN_REPO=""
ONEDNN_COMMIT=""

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Build failed with exit code: $exit_code"
    fi
    # Remove any temporary files if needed
    return $exit_code
}
trap cleanup EXIT

# Parse command line arguments
parse_arguments() {
    for arg in "$@"; do
        case "$arg" in
            --WORKSPACE=*)
                WORKSPACE="${arg#*=}"
                ;;
            --PYTORCH=*)
                PYTORCH="${arg#*=}"
                ;;
            --TORCH_XPU_OPS=*)
                TORCH_XPU_OPS="${arg#*=}"
                ;;
            --COMPONENT=*)
                COMPONENT="${arg#*=}"
                ;;
            --COMPONENT_COMMIT=*)
                COMPONENT_COMMIT="${arg#*=}"
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                log_warning "Unknown argument: $arg"
                ;;
        esac
    done
}

# Show usage information
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
  --WORKSPACE=<path>           Workspace directory (default: $DEFAULT_WORKSPACE)
  --PYTORCH=<spec>             PyTorch specification in format: repo@commit (default: ${DEFAULT_PYTORCH_REPO}@${DEFAULT_PYTORCH_COMMIT})
  --TORCH_XPU_OPS=<spec>       torch-xpu-ops specification in format: repo@commit (default: ${DEFAULT_TORCH_XPU_OPS_REPO}@${DEFAULT_TORCH_XPU_OPS_COMMIT})
  --COMPONENT=<name>           Component to patch (e.g., onednn)
  --COMPONENT_COMMIT=<spec>    Component commit specification
  --help, -h                   Show this help message

Examples:
  $0 --WORKSPACE=/build --PYTORCH=main
  $0 --PYTORCH=https://github.com/pytorch/pytorch.git@v2.3.0
  $0 --TORCH_XPU_OPS=feature/branch
EOF
}

# Validate inputs
validate_inputs() {
    # Set defaults if not provided
    WORKSPACE="${WORKSPACE:-$DEFAULT_WORKSPACE}"
    WORKSPACE=$(realpath -m "$WORKSPACE")

    if [[ -z "${PYTORCH:-}" ]]; then
        PYTORCH="${DEFAULT_PYTORCH_REPO}@${DEFAULT_PYTORCH_COMMIT}"
    fi

    if [[ -z "${TORCH_XPU_OPS:-}" ]]; then
        TORCH_XPU_OPS="$DEFAULT_TORCH_XPU_OPS_REPO}@${DEFAULT_TORCH_XPU_OPS_COMMIT}"
    fi

    # Validate workspace
    if ! mkdir -p "$WORKSPACE" 2>/dev/null; then
        log_error "Cannot create or access workspace directory: $WORKSPACE"
        exit 1
    fi

    log_info "Workspace: $WORKSPACE"
    log_info "PyTorch spec: $PYTORCH"
    log_info "Torch-XPU-Ops spec: $TORCH_XPU_OPS"
    log_info "Component: $COMPONENT"
}

# Extract repository and commit from specification
extract_repo_commit() {
    local spec="$1"
    local default_repo="$2"
    local default_commit="$3"
    local repo=""
    local commit=""

    if [[ "$spec" == *"@"* ]]; then
        repo="${spec%%@*}"
        commit="${spec##*@}"
    else
        # If no repo specified, assume it's just a commit/branch/tag for default repo
        repo="$default_repo"
        commit="$spec"
    fi

    # Ensure repo ends with .git
    if [[ "$repo" != *".git" ]] && [[ "$repo" == http* ]]; then
        repo="${repo}.git"
    fi

    echo "$repo $commit"
}

# Parse configurations
parse_configurations() {
    log_info "Parsing configurations..."

    # Parse PyTorch
    read -r PYTORCH_REPO PYTORCH_COMMIT <<< "$(extract_repo_commit \
        "$PYTORCH" \
        "${DEFAULT_PYTORCH_REPO}" \
        "${DEFAULT_PYTORCH_COMMIT}")"

    # Parse torch-xpu-ops
    read -r TORCH_XPU_OPS_REPO TORCH_XPU_OPS_COMMIT <<< "$(extract_repo_commit \
        "$TORCH_XPU_OPS" \
        "${DEFAULT_TORCH_XPU_OPS_REPO}" \
        "${DEFAULT_TORCH_XPU_OPS_COMMIT}")"

    log_info "PyTorch: $PYTORCH_REPO @ $PYTORCH_COMMIT"
    log_info "Torch-XPU-Ops: $TORCH_XPU_OPS_REPO @ $TORCH_XPU_OPS_COMMIT"
}

# Clone and checkout repository
clone_repo() {
    local repo_url="$1"
    local commit="$2"
    local target_dir="$3"
    local name="$4"

    log_info "Cloning $name..."

    if [[ -d "$target_dir" ]]; then
        log_warning "Directory $target_dir already exists, removing..."
        rm -rf "$target_dir"
    fi

    if ! git clone --recursive "$repo_url" "$target_dir"; then
        log_error "Failed to clone $name from $repo_url"
        exit 1
    fi

    cd "$target_dir"

    if ! git checkout "$commit" 2>/dev/null; then
        log_warning "Commit $commit not found, trying as branch..."
        if ! git checkout -b "build_$commit" "$commit" 2>/dev/null; then
            log_error "Failed to checkout $commit for $name"
            exit 1
        fi
    fi

    # Save commit hash
    local actual_commit=$(git rev-parse HEAD)
    log_info "$name commit: $actual_commit"
    echo "$actual_commit" > "$WORKSPACE/${name}.commit"

    # Show repository info
    log_info "$name repository info:"
    git remote -v | head -2
    git branch --show-current
    git show -s --oneline | head -1
}

# Setup PyTorch
setup_pytorch() {
    log_info "Setting up PyTorch..."
    clone_repo "$PYTORCH_REPO" "$PYTORCH_COMMIT" "$WORKSPACE/pytorch" "PyTorch"
}

# Setup torch-xpu-ops
setup_torch_xpu_ops() {
    log_info "Setting up torch-xpu-ops..."

    local pytorch_dir="$WORKSPACE/pytorch"
    local torch_xpu_ops_dir="$pytorch_dir/third_party/torch-xpu-ops"

    # Handle pinned version
    if [[ "${TORCH_XPU_OPS_COMMIT,,}" == "pinned" ]]; then
        TORCH_XPU_OPS_REPO="https://github.com/intel/torch-xpu-ops.git"
        if [[ -f "$pytorch_dir/third_party/xpu.txt" ]]; then
            TORCH_XPU_OPS_COMMIT=$(cat "$pytorch_dir/third_party/xpu.txt")
            log_info "Using pinned torch-xpu-ops commit: $TORCH_XPU_OPS_COMMIT"
        else
            log_error "xpu.txt not found for pinned version"
            exit 1
        fi
    fi

    # Clean up existing directory
    if [[ -d "$torch_xpu_ops_dir" ]]; then
        rm -rf "$torch_xpu_ops_dir"
    fi

    # Handle different scenarios
    if [[ "${GITHUB_EVENT_NAME:-}" == "pull_request" ]] && [[ -d "$WORKSPACE/torch-xpu-ops" ]]; then
        log_info "Using existing torch-xpu-ops from workspace (PR build)"
        cp -r "$WORKSPACE/torch-xpu-ops" "$torch_xpu_ops_dir"
    else
        clone_repo "$TORCH_XPU_OPS_REPO" "$TORCH_XPU_OPS_COMMIT" "$torch_xpu_ops_dir" "torch-xpu-ops"
    fi
}

# Apply patches and configurations
apply_patches() {
    local pytorch_dir="$WORKSPACE/pytorch"
    cd "$pytorch_dir"

    log_info "Applying patches..."

    # Apply torch-xpu-ops patches
    if [[ -f "third_party/torch-xpu-ops/.github/scripts/apply_torch_pr.py" ]]; then
        python -m pip install -q requests
        if ! python third_party/torch-xpu-ops/.github/scripts/apply_torch_pr.py; then
            log_warning "Failed to apply torch-xpu-ops patches"
        fi
    fi

    # Patch oneDNN if specified
    if [[ "$COMPONENT" == "onednn" ]] && [[ -n "$COMPONENT_COMMIT" ]] && [[ "${COMPONENT_COMMIT,,}" != 'pinned' ]]; then
        patch_onednn
    fi

    # Patch CMakeLists.txt to avoid git checkout during build
    patch_cmakelists
}

# Patch oneDNN configuration
patch_onednn() {
    log_info "Patching oneDNN configuration..."

    read -r ONEDNN_REPO ONEDNN_COMMIT <<< "$(extract_repo_commit \
        "$COMPONENT_COMMIT" \
        "${DEFAULT_ONEDNN_REPO}" \
        "${DEFAULT_ONEDNN_COMMIT}")"

    log_info "Configuring oneDNN to use: $ONEDNN_REPO@$ONEDNN_COMMIT"

    local cmake_file="cmake/Modules/FindMKLDNN.cmake"
    if [[ ! -f "$cmake_file" ]]; then
        log_warning "FindMKLDNN.cmake not found at $cmake_file"
        return 1
    fi

    # Backup original file
    cp "$cmake_file" "$cmake_file.bak"

    # Patch the file
    if ! sed -i \
        -e "s|GIT_REPOSITORY .*|GIT_REPOSITORY $ONEDNN_REPO|" \
        -e "s|GIT_TAG .*|GIT_TAG $ONEDNN_COMMIT|" \
        "$cmake_file"; then
        log_error "Failed to patch oneDNN configuration"
        cp "$cmake_file.bak" "$cmake_file"
        return 1
    fi

    log_success "oneDNN configuration updated"
}

# Patch CMakeLists.txt
patch_cmakelists() {
    local cmake_lists="caffe2/CMakeLists.txt"

    if [[ ! -f "$cmake_lists" ]]; then
        log_warning "$cmake_lists not found, skipping patch"
        return
    fi

    log_info "Patching $cmake_lists..."

    # Backup original
    cp "$cmake_lists" "$cmake_lists.bak"

    # Patch to avoid git checkout during build
    if ! sed -i 's/checkout --quiet .*TORCH_XPU_OPS_COMMIT}/log -n 1/g' \
        "$cmake_lists"; then
        log_error "Failed to patch CMakeLists.txt"
        cp "$cmake_lists.bak" "$cmake_lists"
        return 1
    fi

    log_success "CMakeLists.txt patched"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."

    local pytorch_dir="$WORKSPACE/pytorch"
    cd "$pytorch_dir"

    # Update submodules
    git submodule sync
    git submodule update --init --recursive

    # Install Python dependencies
    if [[ -f "requirements.txt" ]]; then
        python -m pip install -q -r requirements.txt
    fi

    # Install MKL
    python -m pip install -q \
        "mkl-static==$DEFAULT_MKL_VERSION" \
        "mkl-include==$DEFAULT_MKL_VERSION"


}

# Build PyTorch
build_pytorch() {
    log_info "Building PyTorch..."

    local pytorch_dir="$WORKSPACE/pytorch"
    cd "$pytorch_dir"

    # Show applied patches
    log_info "Applied patches summary:"
    git diff --stat 2>/dev/null || true

    # Build with warnings as errors
    if ! WERROR=1 python setup.py bdist_wheel; then
        log_error "PyTorch build failed"
        exit 1
    fi
}

# Post-build processing
post_build() {
    log_info "Post-build processing..."

    local pytorch_dir="$WORKSPACE/pytorch"
    cd "$pytorch_dir"

    # Install patchelf for wheel processing
    python -m pip install -q patchelf

    # Process wheel files
    local dist_dir="$pytorch_dir/dist"
    local tmp_dir="$pytorch_dir/tmp"

    rm -rf "$tmp_dir"
    mkdir -p "$tmp_dir"

    # Find and process wheel files
    local wheel_files=("$dist_dir"/torch-*.whl)
    if [[ ${#wheel_files[@]} -eq 0 ]]; then
        log_error "No wheel files found in $dist_dir"
        exit 1
    fi

    for wheel_file in "${wheel_files[@]}"; do
        log_info "Processing wheel: $(basename "$wheel_file")"

        # Run rpath script if available
        if [[ -f "third_party/torch-xpu-ops/.github/scripts/rpath.sh" ]]; then
            bash third_party/torch-xpu-ops/.github/scripts/rpath.sh "$wheel_file"
        fi

        # Install the wheel
        local processed_wheel="$tmp_dir/$(basename "$wheel_file")"
        if [[ -f "$processed_wheel" ]]; then
            python -m pip install --force-reinstall "$processed_wheel"
        fi
        break
    done
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."

    cd "$WORKSPACE"

    # Check environment
    if ! python "$WORKSPACE/pytorch/torch/utils/collect_env.py"; then
        log_warning "Failed to collect environment info"
    fi

    # Check torch configuration
    if ! python -c "import torch; print(torch.__config__.show())"; then
        log_error "Failed to import torch or show config"
        exit 1
    fi

    if ! python -c "import torch; print(torch.__config__.parallel_info())"; then
        log_warning "Failed to get parallel info"
    fi

    # Check XPU compilation
    local xpu_compiled
    xpu_compiled=$(python -c 'import torch; print(torch.xpu._is_compiled())' 2>/dev/null || echo "false")

    if [[ "${xpu_compiled,,}" != "true" ]]; then
        log_error "XPU is not compiled correctly"
        return 1
    fi

    log_success "XPU compilation verified: $xpu_compiled"
    return 0
}

# Save artifacts
save_artifacts() {
    log_info "Saving artifacts..."

    local pytorch_dir="$WORKSPACE/pytorch"
    local tmp_dir="$pytorch_dir/tmp"

    # Copy wheel files to workspace
    local wheel_files=("$tmp_dir"/torch-*.whl)
    if [[ ${#wheel_files[@]} -eq 0 ]]; then
        log_warning "No wheel files found in $tmp_dir"
        return 1
    fi

    for wheel_file in "${wheel_files[@]}"; do
        local dest="$WORKSPACE/$(basename "$wheel_file")"
        cp "$wheel_file" "$dest"
        log_success "Saved wheel: $dest"
    done

    # Save build info
    cat > "$WORKSPACE/build_info.txt" << EOF
Build Information:
- Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
- Workspace: $WORKSPACE
- PyTorch: $PYTORCH_REPO @ $PYTORCH_COMMIT
- Torch-XPU-Ops: $TORCH_XPU_OPS_REPO @ $TORCH_XPU_OPS_COMMIT
- Component: $COMPONENT
- Component Commit: $COMPONENT_COMMIT
EOF
}

# Main execution
main() {
    parse_arguments "$@"
    validate_inputs
    parse_configurations

    setup_pytorch
    setup_torch_xpu_ops
    apply_patches
    install_dependencies
    build_pytorch
    post_build

    if verify_installation; then
        save_artifacts
        log_success "Build completed successfully!"
        log_info "Wheel files are available in: $WORKSPACE"
    else
        log_error "Verification failed"
        exit 1
    fi
}

# Run main function
main "$@"
