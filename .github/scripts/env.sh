#!/bin/bash

if [ "$1" != "nightly_wheel" ];then
    source /opt/intel/oneapi/compiler/latest/env/vars.sh
    source /opt/intel/oneapi/umf/latest/env/vars.sh
    source /opt/intel/oneapi/pti/latest/env/vars.sh
    source /opt/intel/oneapi/ccl/latest/env/vars.sh
    source /opt/intel/oneapi/mpi/latest/env/vars.sh
    source /opt/intel/oneapi/mkl/latest/env/vars.sh
    export PYTORCH_EXTRA_INSTALL_REQUIREMENTS="\
        intel-cmplr-lib-rt==2025.0.5 |\
        intel-cmplr-lib-ur==2025.0.5 |\
        intel-cmplr-lic-rt==2025.0.5 |\
        intel-sycl-rt==2025.0.5 |\
        impi-devel==2021.14.2 |\
        oneccl-devel==2021.14.1 |\
        mkl-devel==2025.0.1 |\
        onemkl-sycl-dft==2025.0.1 |\
        tcmlib==1.2.0 | umf==0.9.1 | intel-pti==0.10.2 \
    "
else
    echo "Don't need to source DL-Essential for nightly wheel"
fi
