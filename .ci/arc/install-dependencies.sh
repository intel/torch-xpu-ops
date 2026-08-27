#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

# ============================================================
# Base / GitHub Actions runner dependencies
# ============================================================

apt-get update

apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    acl \
    sudo \
    git \
    git-lfs \
    gh \
    jq \
    rsync \
    tar \
    tzdata \
    util-linux \
    zip \
    unzip \
    xz-utils \
    zstd \
    bzip2 \
    gzip \
    file \
    patch \
    tree \
    aria2 \
    gnupg \
    software-properties-common \
    lsb-release

# ============================================================
# Build tools
# ============================================================

apt-get install -y --no-install-recommends \
    build-essential \
    make \
    cmake \
    ninja-build \
    autoconf \
    automake \
    pkg-config \
    ccache

# ============================================================
# GCC
# ============================================================

apt-get install -y --no-install-recommends \
    gcc-12 \
    g++-12 \
    gcc-13 \
    g++-13 \
    gcc-14 \
    g++-14

# ============================================================
# Clang / LLVM
# ============================================================

apt-get install -y --no-install-recommends \
    clang \
    clang-tidy \
    clang-16 \
    clang-17 \
    clang-18

# ============================================================
# Python
# ============================================================

apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    pipx

# ============================================================
# Node.js 22 + npm
# ============================================================

curl -fsSL https://deb.nodesource.com/setup_22.x | bash -

apt-get update

apt-get install -y --no-install-recommends \
    nodejs

corepack enable || true

# ============================================================
# Java
# ============================================================

apt-get install -y --no-install-recommends \
    openjdk-17-jdk \
    openjdk-21-jdk

# ============================================================
# Go
# ============================================================

apt-get install -y --no-install-recommends \
    golang

# ============================================================
# Rust
# ============================================================

apt-get install -y --no-install-recommends \
    rustc \
    cargo

# ============================================================
# Ruby
# ============================================================

apt-get install -y --no-install-recommends \
    ruby-full

# ============================================================
# .NET
# ============================================================

apt-get install -y --no-install-recommends \
    dotnet-sdk-8.0

# ============================================================
# AWS CLI v2
#
# Do NOT use:
#   apt-get install awscli
#
# Ubuntu 24.04 may not provide awscli in the configured
# repositories. Install the official AWS CLI v2 bundle.
# ============================================================

AWSCLI_TMP="/tmp/aws"

rm -rf "$AWSCLI_TMP"
mkdir -p "$AWSCLI_TMP"

curl -fsSL \
    "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
    -o "$AWSCLI_TMP/awscliv2.zip"

unzip -q \
    "$AWSCLI_TMP/awscliv2.zip" \
    -d "$AWSCLI_TMP"

"$AWSCLI_TMP/aws/install" \
    --update

rm -rf "$AWSCLI_TMP"

# ============================================================
# Ansible
# ============================================================

apt-get install -y --no-install-recommends \
    ansible

# ============================================================
# Terraform
# ============================================================

curl -fsSL https://apt.releases.hashicorp.com/gpg \
    | gpg --dearmor \
    -o /usr/share/keyrings/hashicorp-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] \
https://apt.releases.hashicorp.com \
$(. /etc/os-release && echo "$VERSION_CODENAME") main" \
    > /etc/apt/sources.list.d/hashicorp.list

apt-get update

apt-get install -y --no-install-recommends \
    terraform

# ============================================================
# Git LFS
# ============================================================

git lfs install --system

# ============================================================
# Cleanup
# ============================================================

apt-get autoremove -y
apt-get clean

rm -rf \
    /var/lib/apt/lists/* \
    /tmp/* \
    /var/tmp/*

# ============================================================
# Verification
# ============================================================

echo
echo "========================================"
echo " Ubuntu 24.04 Actions Test Environment"
echo "========================================"

echo
echo "--- Compilers ---"
gcc --version | head -1
gcc-12 --version | head -1
gcc-13 --version | head -1
gcc-14 --version | head -1
clang --version | head -1

echo
echo "--- Build ---"
cmake --version | head -1
ninja --version
ccache --version | head -1

echo
echo "--- Git ---"
git --version
git-lfs --version
gh --version | head -1

echo
echo "--- Python ---"
python3 --version
pip3 --version

echo
echo "--- Node ---"
node --version
npm --version

echo
echo "--- Java ---"
java -version 2>&1 | head -1

echo
echo "--- Go ---"
go version

echo
echo "--- Rust ---"
rustc --version
cargo --version

echo
echo "--- Ruby ---"
ruby --version

echo
echo "--- .NET ---"
dotnet --version

echo
echo "--- AWS CLI ---"
aws --version

echo
echo "--- Ansible ---"
ansible --version | head -1

echo
echo "--- Terraform ---"
terraform --version | head -1

echo
echo "========================================"
echo " Dependencies installed successfully."
echo "========================================"
