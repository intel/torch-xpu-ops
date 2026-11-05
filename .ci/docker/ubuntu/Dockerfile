# Use explicit version for better reproducibility
ARG UBUNTU_VERSION=24.04

FROM ubuntu:${UBUNTU_VERSION} AS builder

# Environment configuration
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=UTC

# Install XPU driver - moved earlier to leverage Docker cache
ARG GCC_VERSION
ENV GCC_VERSION=${GCC_VERSION}
ARG XPU_DRIVER_TYPE
ENV XPU_DRIVER_TYPE=${XPU_DRIVER_TYPE}

# Base system updates and essential packages
COPY ./common/install_base.sh /tmp/install_base.sh
RUN --mount=type=cache,target=/var/cache/apt \
    bash /tmp/install_base.sh && \
    rm -f /tmp/install_base.sh

COPY ./common/install_xpu.sh /tmp/install_xpu.sh
RUN bash /tmp/install_xpu.sh && \
    rm -f /tmp/install_xpu.sh

# Install Conda
RUN wget -qO /tmp/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh && \
    bash /tmp/Miniforge3.sh -b -p /opt/conda && \
    rm /tmp/Miniforge3.sh && \
    /opt/conda/bin/conda clean -all -y

ENV PATH=/opt/conda/bin:$PATH

# Create sudoers.d directory and add nopasswd rule
RUN echo "ALL ALL=(ALL:ALL) NOPASSWD: ALL" |tee /etc/sudoers.d/nopasswd && \
    chmod 0440 /etc/sudoers.d/nopasswd
# Verify sudoers syntax (important for safety)
RUN sudo visudo -c

# Create necessary directories with proper permissions
RUN mkdir -p /workspace && chmod 777 -R /opt /workspace
# Set working directory
WORKDIR /workspace

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD which git && which cmake && which g++ || exit 1

# Fix /bin/sh to use bash (required for Copilot runtime which uses `set -o pipefail`)
RUN ln -sf /bin/bash /bin/sh && \
    sh -c 'set -o pipefail && echo "pipefail works"'

# Default command
CMD ["bash"]
