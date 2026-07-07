ARG BASE_IMAGE=intelgpu/manylinux2_28-builder:xpu_2619
FROM ${BASE_IMAGE}

RUN rm -rf /opt/intel/oneapi/
RUN wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/8170208e-86db-4faa-a0d6-1ecf62699574/intel-deep-learning-essentials-2026.0.0.624_offline.sh
RUN bash intel-deep-learning-essentials-2026.0.0.624_offline.sh -a --eula accept -s --action remove || true
RUN bash intel-deep-learning-essentials-2026.0.0.624_offline.sh -a --eula accept -s --install-dir=/opt/intel/oneapi
RUN bash intel-deep-learning-essentials-2026.0.0.624_offline.sh -a --eula accept -s --install-dir=/opt/intel/oneapi --repair
RUN rm -f intel-deep-learning-essentials-2026.0.0.624_offline.sh
RUN rm -rf /tmp/*
RUN rm -rf ~/.cache

# Default command
CMD ["bash"]
