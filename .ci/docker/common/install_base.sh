#!/bin/bash
set -xeuo pipefail
# Install the base system packages and set GCC_VERSION as the default gcc

: "${GCC_VERSION:?GCC_VERSION must be set}"
# shellcheck source=/dev/null
. /etc/os-release

apt-get update
if [ "${APT_UPGRADE:-1}" == "1" ]; then
    apt-get upgrade -y --no-install-recommends
fi
apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg \
    software-properties-common \
    apt-transport-https \
    lsb-release
if [ "${VERSION_ID}" == "22.04" ]; then
    add-apt-repository ppa:ubuntu-toolchain-r/test -y
    apt-get update
fi

packages=(
    # System utilities
    wget curl sudo git unzip zip gh numactl rsync jq
    # Build tools
    gcc g++ "gcc-${GCC_VERSION}" "g++-${GCC_VERSION}"
    build-essential cmake ninja-build pkg-config autoconf automake libtool
    # Graphics libraries
    libgl1 libglib2.0-0 libglib2.0-dev zlib1g zlib1g-dev
    # Python libraries
    python3 python3-dev
)
apt-get install -y --no-install-recommends "${packages[@]}"

for tool in gcc g++ gcov; do
    update-alternatives --install "/usr/bin/${tool}" "${tool}" "/usr/bin/${tool}-${GCC_VERSION}" 100
done
gcc -v && g++ -v && gcov -v

rm -rf /var/lib/apt/lists/*
