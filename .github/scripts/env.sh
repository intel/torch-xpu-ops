#!/bin/bash

source /opt/intel/oneapi/compiler/latest/env/vars.sh
source /opt/intel/oneapi/pti/latest/env/vars.sh
source /opt/intel/oneapi/umf/latest/env/vars.sh
source /opt/intel/oneapi/ccl/latest/env/vars.sh
source /opt/intel/oneapi/mpi/latest/env/vars.sh
export USE_STATIC_MKL=1
export USE_ONEMKL=1
export USE_XCCL=1
export PYTORCH_EXTRA_INSTALL_REQUIREMENTS=" \
        intel-cmplr-lib-rt==2025.1.1 | \
        intel-cmplr-lib-ur==2025.1.1 | \
        intel-cmplr-lic-rt==2025.1.1 | \
        intel-sycl-rt==2025.1.1 | \
        oneccl-devel==2021.15.1; platform_system == 'Linux' and platform_machine == 'x86_64' | \
        oneccl==2021.15.1; platform_system == 'Linux' and platform_machine == 'x86_64' | \
        impi-rt==2021.15.0; platform_system == 'Linux' and platform_machine == 'x86_64' | \
        onemkl-sycl-blas==2025.1.0 | \
        onemkl-sycl-dft==2025.1.0 | \
        onemkl-sycl-lapack==2025.1.0 | \
        onemkl-sycl-rng==2025.1.0 | \
        onemkl-sycl-sparse==2025.1.0 | \
        dpcpp-cpp-rt==2025.1.1 | \
        intel-opencl-rt==2025.1.1 | \
        mkl==2025.1.0 | \
        intel-openmp==2025.1.1 | \
        tbb==2022.1.0 | \
        tcmlib==1.3.0 | \
        umf==0.10.0 | \
        intel-pti==0.12.0
"
