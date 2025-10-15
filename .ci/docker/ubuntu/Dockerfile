ARG UBUNTU_VERSION

FROM ubuntu:${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND noninteractive

# install Intel GPU driver LTS2, refer to https://dgpu-docs.intel.com/driver/installation-lts2.html
ARG XPU_DRIVER_TYPE
ENV XPU_DRIVER_TYPE ${XPU_DRIVER_TYPE}

COPY ./common/install_xpu.sh install_xpu.sh
RUN bash ./install_xpu.sh && rm -f install_xpu.sh

# install extra packages for pytorch benchmark
RUN apt-get update && \
    apt-get install -y wget curl sudo git unzip zip gh numactl rsync jq && \
    apt-get install -y gcc g++ cmake libgl1 zlib1g-dev libglib2.0-dev && \
    apt-get install -y libnl-genl-3-200 && \
    apt-get autoclean && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


USER jenkins
CMD ["bash"]
