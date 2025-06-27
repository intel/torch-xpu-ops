#!/bin/bash

source ${HOME}/intel/oneapi/compiler/latest/env/vars.sh
source ${HOME}/intel/oneapi/pti/latest/env/vars.sh
source ${HOME}/intel/oneapi/umf/latest/env/vars.sh
source ${HOME}/intel/oneapi/ccl/latest/env/vars.sh
source ${HOME}/intel/oneapi/mpi/latest/env/vars.sh
icpx --version
sycl-ls
