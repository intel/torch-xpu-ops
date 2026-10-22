#!/bin/bash

XPU_ONEAPI_PATH="${XPU_ONEAPI_PATH:-"/opt/intel/oneapi"}"

for component in compiler pti umf tcm ccl mpi; do
    # oneAPI is installed on the runner, outside the repo, so shellcheck cannot follow it.
    # shellcheck source=/dev/null
    source "${XPU_ONEAPI_PATH}/${component}/latest/env/vars.sh"
done
icpx --version
sycl-ls
