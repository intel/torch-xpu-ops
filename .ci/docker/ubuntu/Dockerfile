# Use explicit version for better reproducibility
ARG UBUNTU_VERSION=24.04

FROM ubuntu:${UBUNTU_VERSION} AS builder

# Environment configuration
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=UTC

# Install XPU driver - moved earlier to leverage Docker cache
ARG XPU_DRIVER_TYPE
ENV XPU_DRIVER_TYPE=${XPU_DRIVER_TYPE}

COPY ./common/install_xpu.sh /tmp/install_xpu.sh
RUN bash /tmp/install_xpu.sh && \
    rm -f /tmp/install_xpu.sh

# Base system updates and essential packages
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get upgrade -y --no-install-recommends && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        software-properties-common \
        apt-transport-https \
        lsb-release && \
    rm -rf /var/lib/apt/lists/*

# Add repositories and install packages in single layer for better caching
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        # System utilities
        wget \
        curl \
        sudo \
        git \
        unzip \
        zip \
        gh \
        numactl \
        rsync \
        jq \
        # Build tools
        gcc g++ \
        cmake \
        ninja-build \
        pkg-config \
        autoconf \
        automake \
        libtool \
        # Graphics libraries
        libgl1 \
        libglib2.0-0 \
        libglib2.0-dev \
        zlib1g \
        zlib1g-dev \
        # Python and pip
        python3 \
        python3-dev \
        python3-venv \
        python3-pip && \
    # Cleanup
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf \
        /var/lib/apt/lists/* \
        /tmp/* \
        /var/tmp/* \
        /usr/share/doc/* \
        /usr/share/man/*

# Install Python packages
# RUN pip3 install --no-cache-dir --upgrade \
#     pip \
#     setuptools \
#     wheel && \
#     pip3 install --no-cache-dir \
#     numpy \
#     pandas \
#     scipy \
#     psutil

# Create sudoers.d directory and add nopasswd rule
RUN echo "ALL ALL=(ALL:ALL) NOPASSWD: ALL" |tee /etc/sudoers.d/nopasswd && \
    chmod 0440 /etc/sudoers.d/nopasswd
# Verify sudoers syntax (important for safety)
RUN sudo visudo -c

# Create necessary directories with proper permissions
RUN mkdir -p /workspace
# Set working directory
WORKDIR /workspace

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD which git && which cmake && which g++ || exit 1

# Default command
CMD ["bash"]
