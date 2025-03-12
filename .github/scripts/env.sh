#!/bin/bash

if [ "$1" == "build" ];then
    source /opt/intel/oneapi/compiler/latest/env/vars.sh
    source /opt/intel/oneapi/umf/latest/env/vars.sh
    source /opt/intel/oneapi/pti/latest/env/vars.sh
    source /opt/intel/oneapi/ccl/latest/env/vars.sh
    source /opt/intel/oneapi/mpi/latest/env/vars.sh
    source /opt/intel/oneapi/mkl/latest/env/vars.sh
else
    pip install intel-cmplr-lib-rt intel-cmplr-lib-ur intel-cmplr-lic-rt intel-sycl-rt tcmlib umf intel-pti
    echo "Don't need to source DL-Essential for nightly wheel"
fi
