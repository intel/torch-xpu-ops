#!/bin/bash

if [ "$1" != "nightly_wheel" ];then
    source /opt/intel/oneapi/compiler/latest/env/vars.sh
    source /opt/intel/oneapi/umf/latest/env/vars.sh
    source /opt/intel/oneapi/pti/latest/env/vars.sh
    source /opt/intel/oneapi/ccl/latest/env/vars.sh
    source /opt/intel/oneapi/mpi/latest/env/vars.sh
else
    echo "Don't need to source DL-Essential for nightly wheel"
fi
