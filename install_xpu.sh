
#!/bin/bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Setup steps (pick any combination; if none given, prints this help):
  --clean               Clean old Intel dirs, apt caches, sources
  --common              Install common dev packages (gcc, cmake, git, etc.)
  --cpu-perf            Set CPU governor to performance
  --oneapi-dle          Install Intel oneAPI Deep Learning Essentials
  --conda               Install Miniforge (conda)
  --uv                  Install uv (Python package manager)

GPU driver (mutually exclusive — picks one, purges all others first):
  --driver-rolling      PVC rolling
  --driver-lts          PVC LTS  (2350)
  --driver-lts2         PVC LTS2 (2523)
  --driver-client       Client rolling (PPA)

Shorthand:
  --all                 All setup steps + PVC rolling driver

Tuning:
  --gcc-version VER     GCC version to install (default: 13)
  -h, --help            Show this help

Examples:
  sudo $(basename "$0") --all
  sudo $(basename "$0") --clean --common --uv --driver-lts2
  sudo $(basename "$0") --driver-rolling
  sudo $(basename "$0") --common --gcc-version 12
EOF
    exit 0
}

# ── Flags ──────────────────────────────────────────────────────────────
DO_CLEAN=false
DO_COMMON=false
DO_CPU_PERF=false
DO_ONEAPI_DLE=false
DO_CONDA=false
DO_UV=false
DRIVER=""          # rolling | lts | lts2 | client  (empty = skip)
GCC_VERSION=13

[[ $# -eq 0 ]] && usage

_set_driver() {
    if [[ -n "$DRIVER" && "$DRIVER" != "$1" ]]; then
        echo "Error: cannot combine --driver-${DRIVER} with --driver-${1}. Pick one."
        exit 1
    fi
    DRIVER="$1"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            DO_CLEAN=true; DO_COMMON=true; DO_CPU_PERF=true
            DO_ONEAPI_DLE=true; DO_CONDA=true; DO_UV=true
            _set_driver rolling
            ;;
        --clean)           DO_CLEAN=true ;;
        --common)          DO_COMMON=true ;;
        --cpu-perf)        DO_CPU_PERF=true ;;
        --oneapi-dle)      DO_ONEAPI_DLE=true ;;
        --conda)           DO_CONDA=true ;;
        --uv)              DO_UV=true ;;
        --driver-rolling)  _set_driver rolling ;;
        --driver-lts)      _set_driver lts ;;
        --driver-lts2)     _set_driver lts2 ;;
        --driver-client)   _set_driver client ;;
        --gcc-version)     shift; GCC_VERSION="${1:?--gcc-version requires a value}" ;;
        -h|--help)         usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

banner() { echo -e "\n================ $1 ================\n"; }

# ── Driver helpers ────────────────────────────────────────────────────

# Per-flavour package lists. Each function prints the packages to stdout.
# KMD (kernel mode driver) — shared by all PVC flavours.
_pkgs_kmd() {
    echo "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)" \
        flex bison intel-fw-gpu intel-i915-dkms xpu-smi
}

# UMD (user mode driver) — varies by flavour and OS version.
_pkgs_umd_lts() {
    echo \
        intel-opencl-icd intel-level-zero-gpu level-zero \
        intel-media-va-driver-non-free libmfxgen1 libvpl2 \
        libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
        libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
        mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo hwinfo clinfo \
        libigc-dev intel-igc-cm libigdfcl-dev libigfxcmrt-dev level-zero-dev
}

_pkgs_umd_lts2_2204() {
    echo \
        intel-opencl-icd libze-intel-gpu1 libze1 \
        intel-media-va-driver-non-free libmfx-gen1 libvpl2 \
        libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
        libglapi-mesa libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
        mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo hwinfo clinfo intel-ocloc \
        libigc-dev intel-igc-cm libigdfcl-dev libigfxcmrt-dev libze-dev
}

_pkgs_umd_lts2_2404() {
    echo \
        intel-opencl-icd libze-intel-gpu1 libze1 \
        intel-media-va-driver-non-free libmfx-gen1 libvpl2 \
        libegl-mesa0 libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
        libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
        mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo hwinfo clinfo intel-ocloc \
        libigc-dev intel-igc-cm libigdfcl-dev libigfxcmrt-dev libze-dev
}

# rolling 22.04 = same as lts2 22.04; rolling 24.04 = same as lts2 24.04
_pkgs_umd_rolling_2204() { _pkgs_umd_lts2_2204; }
_pkgs_umd_rolling_2404() { _pkgs_umd_lts2_2404; }

_pkgs_client() {
    echo \
        libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc \
        intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools libva-glx2 \
        va-driver-all vainfo libze-dev intel-ocloc xpu-smi
}

# Resolve the UMD package list for the current $DRIVER + $VERSION_ID.
_resolve_umd_packages() {
    case "$DRIVER" in
        lts)     _pkgs_umd_lts ;;
        lts2)
            case "$VERSION_ID" in
                22.04) _pkgs_umd_lts2_2204 ;;
                *)     _pkgs_umd_lts2_2404 ;;
            esac
            ;;
        rolling)
            case "$VERSION_ID" in
                22.04) _pkgs_umd_rolling_2204 ;;
                *)     _pkgs_umd_rolling_2404 ;;
            esac
            ;;
        client)  _pkgs_client ;;
    esac
}

# Collect the superset of ALL driver packages (union across every flavour).
_all_driver_packages() {
    {
        echo flex bison intel-fw-gpu intel-i915-dkms xpu-smi
        _pkgs_umd_lts
        _pkgs_umd_lts2_2204
        _pkgs_umd_lts2_2404
        _pkgs_client
    } | tr ' ' '\n' | sort -u
}

_purge_all_drivers() {
    banner "Purge all GPU driver packages"
    rm -rf /etc/apt/sources.list.d/intel-gpu*
    local pkgs
    mapfile -t pkgs < <(_all_driver_packages)
    for pkg in "${pkgs[@]}"; do
        apt-get purge -y "$pkg" 2>/dev/null || true
    done
    apt-get autoremove -y
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    apt-get autoclean && apt-get clean
}

_add_intel_gpu_repo() {
    local repo_line="$1"
    . /etc/os-release
    wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
        | gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
    echo "$repo_line" | tee /etc/apt/sources.list.d/intel-gpu-${VERSION_CODENAME}.list
    apt update
}

_install_pvc_driver() {
    . /etc/os-release
    # KMD
    local kmd_pkgs
    read -ra kmd_pkgs <<< "$(_pkgs_kmd)"
    apt install -y "${kmd_pkgs[@]}"
    # UMD
    local umd_pkgs
    read -ra umd_pkgs <<< "$(_resolve_umd_packages)"
    apt install -y "${umd_pkgs[@]}"
}

# ── 1. Clean ──────────────────────────────────────────────────────────
if $DO_CLEAN; then
    banner "Clean"
    rm -rf .intel ./intel /opt/intel .cache .config .condarc .conda .gitconfig
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    rm -rf /etc/apt/sources.list.d/intel-gpu*
    apt-get autoclean && apt-get clean
    apt update
fi

# ── 2. Common packages ───────────────────────────────────────────────
if $DO_COMMON; then
    banner "Install common packages (GCC ${GCC_VERSION})"
    apt-get install -y \
        wget curl vim libgomp1 pciutils ca-certificates \
        gnupg software-properties-common apt-transport-https \
        lsb-release sudo git unzip zip gh numactl rsync jq \
        gcc g++ gcc-${GCC_VERSION} g++-${GCC_VERSION} \
        build-essential cmake ninja-build pkg-config \
        autoconf automake libtool libgl1 \
        libglib2.0-0 libglib2.0-dev \
        zlib1g zlib1g-dev \
        python3 python3-dev python3-venv python3-pip

    update-alternatives --install /usr/bin/gcc  gcc  /usr/bin/gcc-${GCC_VERSION}  ${GCC_VERSION}
    update-alternatives --install /usr/bin/g++  g++  /usr/bin/g++-${GCC_VERSION}  ${GCC_VERSION}
    update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-${GCC_VERSION} ${GCC_VERSION}
    gcc -v && g++ -v && gcov -v
fi

# ── 3. uv ────────────────────────────────────────────────────────────
if $DO_UV; then
    banner "Install uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh -c 'env UV_INSTALL_DIR="/usr/local/bin" sh'
fi

# ── 4. CPU performance governor ──────────────────────────────────────
if $DO_CPU_PERF; then
    banner "Set CPU performance governor"
    apt-get install -y \
        linux-tools-common \
        "linux-tools-$(uname -r)" \
        "linux-cloud-tools-$(uname -r)"
    cpupower frequency-set -g performance
    cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c
fi

# ── 5. oneAPI Deep Learning Essentials ───────────────────────────────
if $DO_ONEAPI_DLE; then
    banner "Install oneAPI Deep Learning Essentials"
    DLE_URL="https://registrationcenter-download.intel.com/akdlm/IRC_NAS/3435dc45-055e-4f7a-86b1-779931772404/intel-deep-learning-essentials-2025.1.3.7_offline.sh"
    DLE_SH="intel-deep-learning-essentials-2025.1.3.7_offline.sh"
    wget -c "$DLE_URL" -O "$DLE_SH"
    bash "$DLE_SH" -a -s --eula accept --action install
    rm -f "$DLE_SH"
fi

# ── 6. Miniforge (conda) ────────────────────────────────────────────
if $DO_CONDA; then
    banner "Install Miniforge (conda)"
    MINIFORGE_SH="Miniforge3-$(uname)-$(uname -m).sh"
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/${MINIFORGE_SH}"
    bash "$MINIFORGE_SH" -b -f -p /opt/conda
    rm -f "$MINIFORGE_SH"
fi

# ── 7. GPU driver (purge → install) ─────────────────────────────────
if [[ -n "$DRIVER" ]]; then
    . /etc/os-release
    _purge_all_drivers

    case "$DRIVER" in
        rolling)
            banner "Install PVC rolling GPU driver"
            _add_intel_gpu_repo \
                "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu ${VERSION_CODENAME} unified"
            _install_pvc_driver
            ;;
        lts)
            banner "Install PVC LTS (2350) GPU driver"
            _add_intel_gpu_repo \
                "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu ${VERSION_CODENAME}/lts/2350 unified"
            _install_pvc_driver
            ;;
        lts2)
            banner "Install PVC LTS2 (2523) GPU driver"
            _add_intel_gpu_repo \
                "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu ${VERSION_CODENAME}/lts/2523 unified"
            _install_pvc_driver
            ;;
        client)
            banner "Install client rolling GPU driver (PPA)"
            apt update
            apt-get install -y software-properties-common
            add-apt-repository -y ppa:kobuk-team/intel-graphics
            client_pkgs=()
            read -ra client_pkgs <<< "$(_pkgs_client)"
            apt-get install -y "${client_pkgs[@]}"
            ;;
    esac
fi

banner "Done"
