#!/bin/bash

DEPENDENCY_DIR=${DEPENDENCY_DIR:-$GITHUB_WORKSPACE/../dependencies}
source ${DEPENDENCY_DIR}/DPCPP_JGS/setvars.sh
export _profiling_tools_root=${DEPENDENCY_DIR}/PROFILING_TOOLS_JGS/
export LD_LIBRARY_PATH=${_profiling_tools_root}/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-$(dirname "$(which conda)")/../}:${_profiling_tools_root}:${CMAKE_PREFIX_PATH}

WORK_ROOT=${WORK_ROOT:-$HOME/trees/jgs}
echo -e "\n==================== Coral version ===================="
grep -A 10 "coral:" $WORK_ROOT/build-promotions/build.yml | grep -e asset_version -e revision | sed 's/^[ \t]*//'
grep g_commit_id $WORK_ROOT/CORAL-SIM/build/Release/coral_infra/commit_info.cpp | sed 's/^[ \t]*//'

echo -e "\n==================== XeSim version ===================="
grep -A 5 "cobalt:" $WORK_ROOT/build-promotions/build.yml | grep -e asset_name -e asset_version | sed 's/^[ \t]*//'

echo -e "\n==================== Driver version ===================="
grep "gfx_driver:" $WORK_ROOT/dependencies/NEO/config.yml | sed 's/^[ \t]*//'

echo -e "\n==================== Components version ===================="
grep "Build URL" ${DEPENDENCY_DIR}/DPCPP_JGS/version.txt
grep "Build URL" ${DEPENDENCY_DIR}/PROFILING_TOOLS_JGS/version.txt

icpx --version
sycl-ls
