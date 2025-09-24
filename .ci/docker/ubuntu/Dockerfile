ARG UBUNTU_VERSION

FROM ubuntu:${UBUNTU_VERSION}

ARG UBUNTU_VERSION

ENV DEBIAN_FRONTEND noninteractive

ENV http_proxy http://proxy-dmz.intel.com:912
ENV https_proxy http://proxy-dmz.intel.com:912

# install Intel GPU driver LTS2, refer to https://dgpu-docs.intel.com/driver/installation-lts2.html
ENV XPU_DRIVER_TYPE LTS2
COPY ./common/install_xpu.sh install_xpu.sh
RUN bash ./install_xpu.sh && rm -f install_xpu.sh

# install extra packages for pytorch benchmark
RUN apt-get update
RUN apt-get install -y gcc g++ cmake libgl1 zlib1g-dev libglib2.0-dev
RUN apt-get install -y wget curl sudo git unzip zip gh numactl rsync jq

# Cleanup package manager
RUN apt-get autoclean && apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


USER jenkins
CMD ["bash"]
