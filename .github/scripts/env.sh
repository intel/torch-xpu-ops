#!/bin/bash

if [ "$1" != "nightly_wheel" ];then
    source /home/sdp/intel/oneapi/compiler/latest/env/vars.sh
    source /home/sdp/intel/oneapi/umf/latest/env/vars.sh
    source /home/sdp/intel/oneapi/pti/latest/env/vars.sh
    source /home/sdp/intel/oneapi/ccl/latest/env/vars.sh
    source /home/sdp/intel/oneapi/mpi/latest/env/vars.sh
    source /home/sdp/intel/oneapi/mkl/latest/env/vars.sh
else
    echo "Don't need to source DL-Essential for nightly wheel"
fi
