#!/bin/bash
set -xeuo pipefail
# Install minimal system packages for Linux XPU test containers and set GCC_VERSION as default

: "${GCC_VERSION:?GCC_VERSION must be set}"
# shellcheck source=/dev/null
. /etc/os-release

apt-get update
if [ "${APT_UPGRADE:-0}" == "1" ]; then
    apt-get upgrade -y --no-install-recommends
fi

if [ "${VERSION_ID}" == "22.04" ]; then
    apt-get install -y --no-install-recommends ca-certificates software-properties-common
    add-apt-repository ppa:ubuntu-toolchain-r/test -y
    apt-get update
fi

packages=(
    # Core utilities & runner tools
    ca-certificates curl wget git sudo rsync
    # Device topology & affinity (detect-devices.sh, benchmark workers)
    numactl pciutils clinfo
    # Compiler toolchain for Inductor / Triton / C++ extensions
    make "gcc-${GCC_VERSION}" "g++-${GCC_VERSION}"
    # Runtime shared libraries for OpenCV in UT/E2E
    libgl1 libglib2.0-0
)
apt-get install -y --no-install-recommends "${packages[@]}"

for tool in gcc g++ gcov; do
    update-alternatives --install "/usr/bin/${tool}" "${tool}" "/usr/bin/${tool}-${GCC_VERSION}" 100
done
gcc -v && g++ -v && gcov -v

rm -rf /var/lib/apt/lists/*
