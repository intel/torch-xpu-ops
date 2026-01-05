# Use explicit version for better reproducibility
ARG UBUNTU_VERSION=24.04

FROM ubuntu:${UBUNTU_VERSION} AS builder

# Environment configuration
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=UTC

# Create jenkins user with explicit UID/GID for better compatibility
RUN groupadd -g 1000 jenkins && \
    useradd -m -u 1000 -g jenkins -s /bin/bash jenkins && \
    echo "jenkins ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/jenkins && \
    chmod 0440 /etc/sudoers.d/jenkins

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
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        autoconf \
        automake \
        libtool \
        # Libraries
        libgl1-mesa-glx \
        libglib2.0-0 \
        libglib2.0-dev \
        zlib1g \
        zlib1g-dev \
        libnl-genl-3-200 \
        libnl-3-200 \
        libnl-route-3-200 \
        # Python and pip (if needed for pytorch)
        python3 \
        python3-pip \
        python3-venv \
        python3-dev && \
    # Cleanup
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf \
        /var/lib/apt/lists/* \
        /tmp/* \
        /var/tmp/* \
        /usr/share/doc/* \
        /usr/share/man/*

# Install Python packages (if needed for pytorch benchmark)
# Uncomment and modify as needed
RUN pip3 install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel && \
    pip3 install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    psutil

# Create necessary directories with proper permissions
RUN mkdir -p /workspace && \
    chown -R jenkins:jenkins /workspace

# Switch to jenkins user
USER jenkins:jenkins

# Set working directory
WORKDIR /workspace

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD which git && which cmake && which g++ || exit 1

# Default command
CMD ["bash"]
