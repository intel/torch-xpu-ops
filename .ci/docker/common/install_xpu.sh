#!/bin/bash
set -euo pipefail
# Script used in CI and CD pipeline for Intel GPU driver installation

# Intel® software for general purpose GPU capabilities.
# Refer to https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html

# Users should update to the latest version as it becomes available

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

# Error handling
trap 'log_error "Script failed at line $LINENO. Command: $BASH_COMMAND"; exit 1' ERR

# Default configuration
readonly DEFAULT_XPU_DRIVER_TYPE="rolling"
readonly DEFAULT_XPU_DRIVER_VERSION=""

# Validate environment variables
validate_environment() {
    local -r os_id="${1}"
    local -r driver_type="${2}"

    if [[ -z "${driver_type}" ]]; then
        log_warn "XPU_DRIVER_TYPE not set, defaulting to '${DEFAULT_XPU_DRIVER_TYPE}'"
        export XPU_DRIVER_TYPE="${DEFAULT_XPU_DRIVER_TYPE}"
    fi

    # Normalize driver type to lowercase
    export XPU_DRIVER_TYPE="${XPU_DRIVER_TYPE,,}"

    case "${XPU_DRIVER_TYPE}" in
        lts|lts2|rolling)
            # Valid driver types
            ;;
        *)
            log_error "Invalid XPU_DRIVER_TYPE: ${XPU_DRIVER_TYPE}. Must be 'lts', 'lts2', or 'rolling'"
            exit 1
            ;;
    esac

    # Set driver version based on type
    case "${XPU_DRIVER_TYPE}" in
        lts)
            export XPU_DRIVER_VERSION="/lts/2350"
            ;;
        lts2)
            export XPU_DRIVER_VERSION="/lts/2523"
            ;;
        rolling)
            export XPU_DRIVER_VERSION=""
            ;;
    esac
}

# Common cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 2>/dev/null || true
}

# Common package installation function
install_packages() {
    local -r os_id="${1}"
    shift
    local -r packages=("$@")

    case "${os_id}" in
        ubuntu)
            apt-get install -y --no-install-recommends "${packages[@]}"
            ;;
        rhel|almalinux)
            dnf install -y "${packages[@]}"
            ;;
        sles)
            zypper install -y --no-recommends "${packages[@]}"
            ;;
    esac
}

function install_ubuntu() {
    . /etc/os-release

    local -r supported_lts_versions=("jammy")
    local -r supported_rolling_versions=("jammy" "noble")

    if [ "${XPU_DRIVER_TYPE}" == "lts" ]; then
        if [[ ! " ${supported_lts_versions[*]} " =~ " ${VERSION_CODENAME} " ]]; then
            log_error "Ubuntu version ${VERSION_CODENAME} with ${XPU_DRIVER_TYPE} not supported"
            log_error "Supported versions for LTS: ${supported_lts_versions[*]}"
            exit 1
        fi
    else
        if [[ ! " ${supported_rolling_versions[*]} " =~ " ${VERSION_CODENAME} " ]]; then
            log_error "Ubuntu version ${VERSION_CODENAME} with ${XPU_DRIVER_TYPE} not supported"
            log_error "Supported versions: ${supported_rolling_versions[*]}"
            exit 1
        fi
    fi

    log_info "Installing Intel GPU drivers for Ubuntu ${VERSION_CODENAME} (${XPU_DRIVER_TYPE})"

    # Update package list
    apt-get update -y

    # Install prerequisites
    install_packages "ubuntu" gpg gpg-agent wget ca-certificates

    # Add Intel GPU repository key
    local -r key_url="https://repositories.intel.com/gpu/intel-graphics.key"
    local -r keyring_path="/usr/share/keyrings/intel-graphics.gpg"

    if ! wget -qO - "${key_url}" | gpg --yes --dearmor --output "${keyring_path}"; then
        log_error "Failed to download and install Intel GPU repository key"
        exit 1
    fi

    # Add repository
    local -r repo_list="/etc/apt/sources.list.d/intel-gpu-${VERSION_CODENAME}.list"
    echo "deb [arch=amd64 signed-by=${keyring_path}] \
        https://repositories.intel.com/gpu/ubuntu ${VERSION_CODENAME}${XPU_DRIVER_VERSION} unified" \
        | tee "${repo_list}"

    # Update repository index
    apt-get update

    # Install xpu-smi and dependencies
    install_packages "ubuntu" flex bison xpu-smi

    # Install runtime packages based on driver type
    if [ "${XPU_DRIVER_TYPE}" == "lts" ]; then
        local -r runtime_packages=(
            intel-opencl-icd intel-level-zero-gpu level-zero
            intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2
            libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri
            libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers
            mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo hwinfo clinfo
        )
        install_packages "ubuntu" "${runtime_packages[@]}"

        # Development packages
        local -r dev_packages=(
            libigc-dev intel-igc-cm libigdfcl-dev libigfxcmrt-dev level-zero-dev
        )
        install_packages "ubuntu" "${dev_packages[@]}"
    else # rolling or lts2 driver
        local -r base_packages=(
            intel-opencl-icd libze-intel-gpu1 libze1
            intel-media-va-driver-non-free libmfx-gen1 libvpl2
            libegl-mesa0 libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri
            libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers
            mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo hwinfo clinfo intel-ocloc
        )

        if [ "${VERSION_CODENAME}" == "jammy" ]; then
            install_packages "ubuntu" "${base_packages[@]//libegl1-mesa-dev/libegl1-mesa}"
        else
            install_packages "ubuntu" "${base_packages[@]}"
        fi

        # Development packages
        install_packages "ubuntu" libigc-dev intel-igc-cm libigdfcl-dev libigfxcmrt-dev libze-dev
    fi

    log_info "Ubuntu installation completed successfully"
}

function install_rhel() {
    . /etc/os-release

    local -r os_id="${ID}"
    local version_id="${VERSION_ID}"

    # Define supported versions
    declare -A supported_versions
    supported_versions["lts"]="8.8 8.10 9.2 9.4 9.5"
    supported_versions["lts2"]="8.10 9.4 9.6 10.0"
    supported_versions["rolling"]="8.10 9.4 9.6"

    # Workaround for AlmaLinux
    if [[ "${os_id}" == "almalinux" ]]; then
        case "${XPU_DRIVER_TYPE}" in
            lts)
                version_id="8.8"
                ;;
            lts2|rolling)
                version_id="8.10"
                ;;
        esac
        log_warn "AlmaLinux detected, using RHEL ${version_id} compatibility mode"
    fi

    # Check version support
    if [[ ! " ${supported_versions[${XPU_DRIVER_TYPE}]} " =~ " ${version_id} " ]]; then
        log_error "${os_id^} version ${VERSION_ID} with ${XPU_DRIVER_TYPE} not supported"
        log_error "Supported versions for ${XPU_DRIVER_TYPE}: ${supported_versions[${XPU_DRIVER_TYPE}]}"
        exit 1
    fi

    log_info "Installing Intel GPU drivers for ${os_id^} ${version_id} (${XPU_DRIVER_TYPE})"

    # Install required tools
    install_packages "${os_id}" 'dnf-command(config-manager)'

    # Add repository
    local -r repo_url="https://repositories.intel.com/gpu/rhel/${version_id}${XPU_DRIVER_VERSION}/unified/intel-gpu-${version_id}.repo"
    dnf config-manager --add-repo "${repo_url}"

    # Install xpu-smi
    install_packages "${os_id}" flex bison xpu-smi

    # Install runtime packages
    local -r runtime_packages=(
        intel-opencl intel-media libmfxgen1 libvpl2
        level-zero intel-level-zero-gpu mesa-dri-drivers mesa-vulkan-drivers
        mesa-vdpau-drivers libdrm mesa-libEGL mesa-libgbm mesa-libGL
        mesa-libxatracker libvpl-tools intel-metrics-discovery
        intel-metrics-library intel-igc-core intel-igc-cm
        libva libva-utils intel-gmmlib libmetee intel-gsc intel-ocloc
    )

    if [ "${XPU_DRIVER_TYPE}" == "lts" ]; then
        dnf install --skip-broken -y "${runtime_packages[@]}"
    else
        dnf install --skip-broken -y "${runtime_packages[@]}"
    fi

    # Install development packages
    local -r dev_packages=(
        intel-igc-opencl-devel level-zero-devel intel-gsc-devel libmetee-devel
    )
    install_packages "${os_id}" "${dev_packages[@]}"

    # Install diagnostic tools from EPEL
    if [[ "${os_id}" == "rhel" ]] || [[ "${os_id}" == "almalinux" ]]; then
        if ! dnf repolist | grep -q epel; then
            log_warn "EPEL repository not enabled. Attempting to enable it..."
            dnf install -y epel-release || log_warn "Failed to enable EPEL"
        fi
        install_packages "${os_id}" hwinfo clinfo
    fi

    log_info "RHEL/AlmaLinux installation completed successfully"
}

function install_sles() {
    . /etc/os-release

    local -r version_sp="${VERSION_ID//./sp}"

    # Define supported versions
    declare -A supported_versions
    supported_versions["lts"]="15sp4 15sp5 15sp6"
    supported_versions["lts2"]="15sp4 15sp5 15sp6 15sp7"
    supported_versions["rolling"]="15sp4 15sp5 15sp6"

    if [[ ! " ${supported_versions[${XPU_DRIVER_TYPE}]} " =~ " ${version_sp} " ]]; then
        log_error "SLES version ${VERSION_ID} with ${XPU_DRIVER_TYPE} not supported"
        log_error "Supported versions for ${XPU_DRIVER_TYPE}: ${supported_versions[${XPU_DRIVER_TYPE}]}"
        exit 1
    fi

    log_info "Installing Intel GPU drivers for SLES ${VERSION_ID} (${XPU_DRIVER_TYPE})"

    # Add repository
    local -r repo_url="https://repositories.intel.com/gpu/sles/${version_sp}${XPU_DRIVER_VERSION}/unified/intel-gpu-${version_sp}.repo"
    zypper addrepo -f -r "${repo_url}"

    # Import repository key
    rpm --import https://repositories.intel.com/gpu/intel-graphics.key

    # Install required packages
    local -r base_packages=(
        lsb-release flex bison xpu-smi
        intel-level-zero-gpu level-zero intel-gsc intel-opencl intel-ocloc
        intel-media-driver libigfxcmrt7 libvpl2 libvpl-tools libmfxgen1
    )

    if [ "${XPU_DRIVER_TYPE}" == "lts" ]; then
        local -r packages=("${base_packages[@]}" libmfx1)
    else
        local -r packages=("${base_packages[@]}")
    fi

    install_packages "sles" "${packages[@]}"

    # Install development packages
    local -r dev_packages=(
        libigdfcl-devel intel-igc-cm libigfxcmrt-devel level-zero-devel
        clinfo libOpenCL1 libva-utils hwinfo
    )
    install_packages "sles" "${dev_packages[@]}"

    log_info "SLES installation completed successfully"
}

# Main execution
main() {
    log_info "Starting Intel GPU driver installation"

    # Determine OS
    local os_id
    if [[ -f /etc/os-release ]]; then
        os_id=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
    else
        log_error "Cannot determine OS: /etc/os-release not found"
        exit 1
    fi

    # Validate and setup environment
    validate_environment "${os_id}" "${XPU_DRIVER_TYPE:-}"

    # Install based on OS
    case "${os_id}" in
        ubuntu)
            install_ubuntu
            ;;
        rhel|almalinux)
            install_rhel
            ;;
        sles)
            install_sles
            ;;
        *)
            log_error "Unsupported OS: ${os_id}"
            log_error "Supported OS: Ubuntu, RHEL, AlmaLinux, SLES"
            exit 1
            ;;
    esac

    # Perform cleanup
    cleanup

    log_info "Intel GPU driver installation completed successfully"

    # Verify installation
    log_info "Verifying installation..."
    if command -v xpu-smi &> /dev/null; then
        log_info "xpu-smi installed successfully"
    else
        log_warn "xpu-smi not found in PATH"
    fi
}

# Only run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
