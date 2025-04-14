#!/bin/bash

if [ "$1" != "nightly_wheel" ];then
    source /opt/intel/oneapi/compiler/latest/env/vars.sh
    source /opt/intel/oneapi/umf/latest/env/vars.sh
    source /opt/intel/oneapi/pti/latest/env/vars.sh
    source /opt/intel/oneapi/ccl/latest/env/vars.sh
    source /opt/intel/oneapi/mpi/latest/env/vars.sh
    source /opt/intel/oneapi/mkl/latest/env/vars.sh
    export PYTORCH_EXTRA_INSTALL_REQUIREMENTS="\
        intel-cmplr-lib-rt==2025.0.5; platform_system == 'Linux' |\
        intel-cmplr-lib-ur==2025.0.5; platform_system == 'Linux' |\
        intel-cmplr-lic-rt==2025.0.5; platform_system == 'Linux' |\
        intel-sycl-rt==2025.0.5; platform_system == 'Linux' |\
        impi-devel==2021.14.2; platform_system == 'Linux' |\
        oneccl-devel==2021.14.1; platform_system == 'Linux' |\
        mkl-devel==2025.0.1; platform_system == 'Linux' |\
        onemkl-sycl-dft==2025.0.1; platform_system == 'Linux' |\
        tcmlib==1.2.0 | umf==0.9.1 | intel-pti==0.10.2 \
    "
else
    echo "Don't need to source DL-Essential for nightly wheel"
fi
