#!/bin/bash

source /opt/intel/oneapi/compiler/2025.1/env/vars.sh
source /opt/intel/oneapi/pti/latest/env/vars.sh
source /opt/intel/oneapi/umf/0.10/env/vars.sh
source /opt/intel/oneapi/ccl/2021.15/env/vars.sh
source /opt/intel/oneapi/mpi/2021.15/env/vars.sh
icpx --version
sycl-ls
