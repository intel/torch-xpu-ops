#!/bin/bash
# Enhanced build script for PyTorch with XPU support
# Usage:
#   ./build.sh [OPTIONS]
#   ./build.sh --WORKSPACE=<path> --PYTORCH=main --TORCH_XPU_OPS=main
set -xeu -o pipefail

# -------------------- Configuration --------------------
# Git clone depth (set to 1 for shallow, empty for full history)
: "${GIT_DEPTH:=}"  # e.g., export GIT_DEPTH=1 for shallow clones

# Default repositories
readonly DEFAULT_PYTORCH_REPO="https://github.com/pytorch/pytorch.git"
readonly DEFAULT_PYTORCH_COMMIT="main"
readonly DEFAULT_TORCH_XPU_OPS_REPO="https://github.com/intel/torch-xpu-ops.git"
readonly DEFAULT_TORCH_XPU_OPS_COMMIT="main"
readonly DEFAULT_ONEDNN_REPO="https://github.com/uxlfoundation/oneDNN.git"
readonly DEFAULT_ONEDNN_COMMIT="main"

# Extra install requirements for oneAPI (used only if XPU_ONEAPI_PATH not set)
# MKL version
: "${MKL_VERSION:=2025.3.1}"
if [[ -z "${XPU_ONEAPI_PATH:-}" ]]; then
    export USE_STATIC_MKL=1
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
        intel-pti==0.16.0"
fi

# -------------------- Global Variables --------------------
WORKSPACE=""
PYTORCH_REPO=""
PYTORCH_COMMIT=""
TORCH_XPU_OPS_REPO=""
TORCH_XPU_OPS_COMMIT=""
COMPONENT=""
COMPONENT_COMMIT=""
ONEDNN_REPO=""
ONEDNN_COMMIT=""

# -------------------- Utility Functions --------------------
# Color definitions
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*" >&2; }
log_error()   { echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*" >&2; }

# Realpath with fallback (for systems without realpath)
realpath_fallback() {
    if command -v realpath &>/dev/null; then
        realpath -m "$1"
    else
        # Simple absolute path resolution (may not resolve symlinks)
        case "$1" in
            /*) echo "$1" ;;
            *) echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")" ;;
        esac
    fi
}

# Cleanup trap
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Build failed with exit code: $exit_code"
    fi
    # Add custom cleanup here if needed
    return $exit_code
}
trap cleanup EXIT

# -------------------- Argument Parsing --------------------
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

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
  --WORKSPACE=<path>           Workspace directory (default: /tmp/pytorch_build)
  --PYTORCH=<spec>             PyTorch specification (format: [repo@]commit, default: ${DEFAULT_PYTORCH_REPO}@${DEFAULT_PYTORCH_COMMIT})
  --TORCH_XPU_OPS=<spec>       torch-xpu-ops specification (format: [repo@]commit, default: ${DEFAULT_TORCH_XPU_OPS_REPO}@${DEFAULT_TORCH_XPU_OPS_COMMIT})
  --COMPONENT=<name>           Component to patch (e.g., onednn)
  --COMPONENT_COMMIT=<spec>    Component commit specification (format: [repo@]commit)
  --help, -h                   Show this help message

Environment variables:
  GIT_DEPTH                    If set to a positive integer, performs shallow clone with that depth.
  MKL_VERSION                  MKL version to install (default: 2025.3.0)
  XPU_ONEAPI_PATH              Path to oneAPI installation (skips extra requirements if set)

Examples:
  $0 --WORKSPACE=/build --PYTORCH=main
  $0 --PYTORCH=https://github.com/pytorch/pytorch.git@v2.3.0
  $0 --TORCH_XPU_OPS=feature/branch
EOF
}

# -------------------- Input Validation --------------------
validate_inputs() {
    # Set defaults
    WORKSPACE="${WORKSPACE:-/tmp/pytorch_build}"
    WORKSPACE=$(realpath_fallback "$WORKSPACE")

    PYTORCH="${PYTORCH:-${DEFAULT_PYTORCH_REPO}@${DEFAULT_PYTORCH_COMMIT}}"
    TORCH_XPU_OPS="${TORCH_XPU_OPS:-${DEFAULT_TORCH_XPU_OPS_REPO}@${DEFAULT_TORCH_XPU_OPS_COMMIT}}"

    # Create workspace if needed
    if ! mkdir -p "$WORKSPACE" 2>/dev/null; then
        log_error "Cannot create or access workspace: $WORKSPACE"
        exit 1
    fi

    log_info "Workspace: $WORKSPACE"
    log_info "PyTorch spec: $PYTORCH"
    log_info "Torch-XPU-Ops spec: $TORCH_XPU_OPS"
    log_info "Component: ${COMPONENT:-none}"
    log_info "Component commit: ${COMPONENT_COMMIT:-none}"
}

# -------------------- Repository/Commit Parsing --------------------
# Extracts repo URL and commit from a spec string.
# Usage: extract_repo_commit <spec> <default_repo> <default_commit>
# Returns: repo_url commit (space-separated)
extract_repo_commit() {
    local spec="$1"
    local default_repo="$2"
    local default_commit="$3"
    local repo="$default_repo"
    local commit="$default_commit"

    if [[ "$spec" == *"@"* ]]; then
        repo="${spec%%@*}"
        commit="${spec##*@}"
    else
        # If no '@', assume spec is a commit/branch/tag for the default repo
        commit="$spec"
    fi

    # Ensure repo URL ends with .git if it's a http(s) URL
    if [[ "$repo" =~ ^https?:// && "$repo" != *.git ]]; then
        repo="${repo}.git"
    fi

    echo "$repo $commit"
}

# -------------------- Clone Repository --------------------
# Clones a repository and checks out the specified commit.
# Usage: clone_repo <repo_url> <commit> <target_dir> <name>
clone_repo() {
    local repo_url="$1"
    local commit="$2"
    local target_dir="$3"
    local name="$4"
    local clone_args=("--recursive")

    if [[ -n "${GIT_DEPTH:-}" && "$GIT_DEPTH" -gt 0 ]]; then
        clone_args+=("--depth" "$GIT_DEPTH")
        log_info "Shallow clone depth: $GIT_DEPTH"
    fi

    log_info "Cloning $name from $repo_url ..."

    if [[ -d "$target_dir" ]]; then
        log_warning "Directory $target_dir already exists, removing..."
        rm -rf "$target_dir"
    fi

    if ! git clone "${clone_args[@]}" "$repo_url" "$target_dir"; then
        log_error "Failed to clone $name"
        exit 1
    fi

    cd "$target_dir"

    # Try to checkout the specified commit
    if ! git checkout "$commit" 2>/dev/null; then
        log_warning "Commit $commit not found, trying as remote branch..."
        if ! git checkout -b "build_$commit" "origin/$commit" 2>/dev/null; then
            log_error "Failed to checkout $commit for $name"
            exit 1
        fi
    fi

    # Record actual commit hash
    local actual_commit=$(git rev-parse HEAD)
    log_info "$name checked out at commit: $actual_commit"
    echo "$actual_commit" > "$WORKSPACE/${name}.commit"

    # Show repository info (first remote, current branch, latest commit)
    git remote -v
    git branch --show-current
    git show -s --oneline HEAD
    echo ""
}

# -------------------- Setup Functions --------------------
setup_pytorch() {
    log_info "Setting up PyTorch..."
    clone_repo "$PYTORCH_REPO" "$PYTORCH_COMMIT" "$WORKSPACE/pytorch" "PyTorch"
}

setup_torch_xpu_ops() {
    log_info "Setting up torch-xpu-ops..."

    local pytorch_dir="$WORKSPACE/pytorch"
    local torch_xpu_ops_dir="$pytorch_dir/third_party/torch-xpu-ops"

    # Handle pinned version (special value "pinned")
    if [[ "${TORCH_XPU_OPS_COMMIT,,}" == "pinned" ]]; then
        TORCH_XPU_OPS_REPO="$DEFAULT_TORCH_XPU_OPS_REPO"
        if [[ -f "$pytorch_dir/third_party/xpu.txt" ]]; then
            TORCH_XPU_OPS_COMMIT=$(cat "$pytorch_dir/third_party/xpu.txt")
            log_info "Using pinned torch-xpu-ops commit: $TORCH_XPU_OPS_COMMIT"
        else
            log_error "xpu.txt not found in PyTorch third_party; cannot use pinned version"
            exit 1
        fi
    fi

    # Remove existing directory if any
    if [[ -d "$torch_xpu_ops_dir" ]]; then
        rm -rf "$torch_xpu_ops_dir"
    fi

    # In a GitHub PR workflow, we may have a local copy of torch-xpu-ops
    if [[ "${GITHUB_EVENT_NAME:-}" == "pull_request" ]] && [[ -d "$WORKSPACE/torch-xpu-ops" ]]; then
        log_info "Using local torch-xpu-ops from workspace (PR build)"
        cp -r "$WORKSPACE/torch-xpu-ops" "$torch_xpu_ops_dir"
    else
        clone_repo "$TORCH_XPU_OPS_REPO" "$TORCH_XPU_OPS_COMMIT" "$torch_xpu_ops_dir" "torch-xpu-ops"
    fi
}

# -------------------- Patching Functions --------------------
apply_patches() {
    local pytorch_dir="$WORKSPACE/pytorch"
    cd "$pytorch_dir"

    log_info "Applying patches..."

    # Apply torch-xpu-ops patches (if any)
    if [[ -f "third_party/torch-xpu-ops/.github/scripts/apply_torch_pr.py" ]]; then
        python -m pip install requests
        if ! python third_party/torch-xpu-ops/.github/scripts/apply_torch_pr.py; then
            log_warning "Failed to apply torch-xpu-ops patches (continuing)"
        fi
    fi

    # Patch oneDNN if specified
    if [[ "$COMPONENT" == "onednn" ]] && [[ -n "$COMPONENT_COMMIT" ]] && [[ "${COMPONENT_COMMIT,,}" != 'pinned' ]]; then
        patch_onednn
    fi

    # Patch CMakeLists.txt to avoid git checkout during build (speeds up build)
    patch_cmakelists
}

patch_onednn() {
    log_info "Patching oneDNN configuration..."

    local pytorch_dir="$WORKSPACE/pytorch"
    cd "$pytorch_dir"

    read -r ONEDNN_REPO ONEDNN_COMMIT <<< "$(extract_repo_commit \
        "$COMPONENT_COMMIT" \
        "$DEFAULT_ONEDNN_REPO" \
        "$DEFAULT_ONEDNN_COMMIT")"

    log_info "Configuring oneDNN to use: $ONEDNN_REPO @ $ONEDNN_COMMIT"

    local cmake_file="cmake/Modules/FindMKLDNN.cmake"
    if [[ ! -f "$cmake_file" ]]; then
        log_error "FindMKLDNN.cmake not found at $cmake_file"
        return 1
    fi

    # Backup original
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

patch_cmakelists() {
    local pytorch_dir="$WORKSPACE/pytorch"
    local cmake_lists="$pytorch_dir/caffe2/CMakeLists.txt"

    if [[ ! -f "$cmake_lists" ]]; then
        log_warning "$cmake_lists not found, skipping patch"
        return
    fi

    log_info "Patching $cmake_lists to disable git checkout during build..."

    cp "$cmake_lists" "$cmake_lists.bak"

    # Replace 'checkout --quiet' with 'log -n 1' to avoid network calls
    if ! sed -i 's/checkout --quiet .*TORCH_XPU_OPS_COMMIT}/log -n 1/g' "$cmake_lists"; then
        log_error "Failed to patch CMakeLists.txt"
        cp "$cmake_lists.bak" "$cmake_lists"
        return 1
    fi

    log_success "CMakeLists.txt patched"
}

# -------------------- Dependencies --------------------
install_dependencies() {
    log_info "Installing build dependencies..."

    local pytorch_dir="$WORKSPACE/pytorch"
    cd "$pytorch_dir"

    # Update submodules
    git submodule sync
    git submodule update --init --recursive

    # Python dependencies
    if [[ -f "requirements.txt" ]]; then
        python -m pip install -r requirements.txt
    fi

    # Install MKL (static and includes)
    python -m pip install \
        "mkl-static==$MKL_VERSION" \
        "mkl-include==$MKL_VERSION"
}

# -------------------- Build --------------------
build_pytorch() {
    log_info "Building PyTorch..."

    local pytorch_dir="$WORKSPACE/pytorch"
    cd "$pytorch_dir"

    # Show applied patches summary
    git diff --stat 2>/dev/null || true

    # Build with warnings as errors
    if ! WERROR=1 python setup.py bdist_wheel; then
        log_error "PyTorch build failed"
        exit 1
    fi

    log_success "PyTorch build completed"
}

# -------------------- Post-Build Processing --------------------
post_build() {
    log_info "Post-build processing..."

    local pytorch_dir="$WORKSPACE/pytorch"
    cd "$pytorch_dir"

    # Install patchelf if not present
    python -m pip install patchelf

    local dist_dir="$pytorch_dir/dist"
    local tmp_dir="$pytorch_dir/tmp"
    mkdir -p "$tmp_dir"

    # Find wheel files
    local wheel_files=("$dist_dir"/torch-*.whl)
    if [[ ${#wheel_files[@]} -eq 0 ]]; then
        log_error "No wheel files found in $dist_dir"
        exit 1
    fi

    for wheel_file in "${wheel_files[@]}"; do
        log_info "Processing wheel: $(basename "$wheel_file")"

        # Run rpath script if available; it will modify the wheel in place or produce a new one.
        if [[ -f "third_party/torch-xpu-ops/.github/scripts/rpath.sh" ]]; then
            # We'll assume it modifies in place and then we use that file.
            if bash third_party/torch-xpu-ops/.github/scripts/rpath.sh "$wheel_file" "$tmp_dir"; then
                # If rpath.sh succeeded, install from the processed wheel
                local working_wheel=($tmp_dir/torch-*.whl)
                processed_wheel="$working_wheel"
            else
                log_warning "rpath.sh failed, using original wheel"
                processed_wheel="$wheel_file"
            fi
        else
            processed_wheel="$wheel_file"
        fi

        # Install the wheel (force reinstall to ensure clean state)
        python -m pip install --force-reinstall "$processed_wheel"
        log_success "Installed: $processed_wheel"
        cp "$processed_wheel" "$WORKSPACE/"
        break # Only process first wheel (there should be exactly one)
    done
}

# -------------------- Verification --------------------
verify_installation() {
    log_info "Verifying installation..."

    cd "$WORKSPACE"

    # Collect environment info (optional, but useful)
    if ! python "$WORKSPACE/pytorch/torch/utils/collect_env.py" 2>/dev/null; then
        log_warning "Failed to collect environment info"
    fi

    # Check PyTorch basics
    if ! python -c "import torch; print('PyTorch version:', torch.__version__)"; then
        log_error "Failed to import torch"
        return 1
    fi

    # Check XPU compilation
    local xpu_compiled
    xpu_compiled=$(python -c 'import torch; print(torch.xpu._is_compiled())' 2>/dev/null || echo "False")
    if [[ "${xpu_compiled,,}" != "true" ]]; then
        log_error "XPU is not compiled correctly (torch.xpu._is_compiled() returned $xpu_compiled)"
        return 1
    fi

    log_success "XPU compilation verified"

    # Save build info
    . /etc/os-release
    cat > "$WORKSPACE/build_info.log" << EOF
Build Information:
  Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
  Workspace: $WORKSPACE
  OS: $VERSION
  Kernel: $(uname -r)
  GCC: $(g++ -v 2>&1 |tail -n 1)
  Python: $(python -V 2>&1 |tail -n 1)
  PyTorch: $PYTORCH_REPO @ $(python -c "import torch; print(torch.version.git_version)" 2>&1 |tail -n 1)
  Torch-XPU-Ops: $TORCH_XPU_OPS_REPO @ $TORCH_XPU_OPS_COMMIT
  Component: ${COMPONENT:-none}
  Component Commit: ${COMPONENT_COMMIT:-none}
  MKL Version: $MKL_VERSION
EOF
    log_info "Build info saved to build_info.log"

    return 0
}

# -------------------- Main --------------------
main() {
    parse_arguments "$@"
    validate_inputs
    parse_configurations

    # Pre-flight checks
    command -v git >/dev/null 2>&1 || { log_error "git is required but not installed"; exit 1; }
    command -v python >/dev/null 2>&1 || { log_error "python is required but not installed"; exit 1; }
    command -v pip >/dev/null 2>&1 || { log_error "pip is required but not installed"; exit 1; }

    setup_pytorch
    setup_torch_xpu_ops
    apply_patches
    install_dependencies
    build_pytorch
    post_build

    if verify_installation; then
        log_success "Build completed successfully!"
        log_info "Wheel files are available in: $WORKSPACE"
    else
        log_error "Verification failed"
        exit 1
    fi
}

# Parse configurations (after inputs are set)
parse_configurations() {
    log_info "Parsing repository specifications..."

    read -r PYTORCH_REPO PYTORCH_COMMIT <<< "$(extract_repo_commit \
        "$PYTORCH" \
        "$DEFAULT_PYTORCH_REPO" \
        "$DEFAULT_PYTORCH_COMMIT")"

    read -r TORCH_XPU_OPS_REPO TORCH_XPU_OPS_COMMIT <<< "$(extract_repo_commit \
        "$TORCH_XPU_OPS" \
        "$DEFAULT_TORCH_XPU_OPS_REPO" \
        "$DEFAULT_TORCH_XPU_OPS_COMMIT")"

    log_info "PyTorch: $PYTORCH_REPO @ $PYTORCH_COMMIT"
    log_info "Torch-XPU-Ops: $TORCH_XPU_OPS_REPO @ $TORCH_XPU_OPS_COMMIT"
}

# Run main
main "$@"
