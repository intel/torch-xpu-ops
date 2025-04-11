#!/bin/bash

if [ "$1" != "nightly_wheel" ];then
    source /opt/intel/oneapi/compiler/latest/env/vars.sh
    source /opt/intel/oneapi/umf/latest/env/vars.sh
    source /opt/intel/oneapi/pti/latest/env/vars.sh
    source /opt/intel/oneapi/ccl/latest/env/vars.sh
    source /opt/intel/oneapi/mpi/latest/env/vars.sh
    source /opt/intel/oneapi/mkl/latest/env/vars.sh
    export PYTORCH_EXTRA_INSTALL_REQUIREMENTS="\
        intel-cmplr-lib-rt; platform_system == 'Linux' |\
        intel-cmplr-lib-ur; platform_system == 'Linux' |\
        intel-cmplr-lic-rt; platform_system == 'Linux' |\
        intel-sycl-rt; platform_system == 'Linux' |\
        impi-devel; platform_system == 'Linux' |\
        oneccl-devel; platform_system == 'Linux' |\
        mkl-devel; platform_system == 'Linux' |\
        onemkl-sycl-dft; platform_system == 'Linux' |\
        tcmlib | umf | intel-pti \
    "
else
    echo "Don't need to source DL-Essential for nightly wheel"
fi
