#!/bin/bash

DEPENDENCY_DIR=${DEPENDENCY_DIR:-$GITHUB_WORKSPACE/../dependencies}
source ${DEPENDENCY_DIR}/DPCPP_JGS/setvars.sh
export _profiling_tools_root=${DEPENDENCY_DIR}/PROFILING_TOOLS_JGS/
export LD_LIBRARY_PATH=${_profiling_tools_root}/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-$(dirname "$(which conda)")/../}:${_profiling_tools_root}:${CMAKE_PREFIX_PATH}
grep "Build URL" ${DEPENDENCY_DIR}/DPCPP_JGS/version.txt
grep "Build URL" ${DEPENDENCY_DIR}/PROFILING_TOOLS_JGS/version.txt
icpx --version
sycl-ls
