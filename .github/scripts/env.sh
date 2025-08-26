#!/bin/bash

DEPENDENCY_DIR=${DEPENDENCY_DIR:-$GITHUB_WORKSPACE/../dependencies}
PYTORCH_VERSION="${1:-master_next}"

source ${DEPENDENCY_DIR}/DPCPP_JGS/setvars.sh
if [ "${PYTORCH_VERSION}" == "master_next" ];then
    export _profiling_tools_root=${DEPENDENCY_DIR}/PROFILING_TOOLS_JGS/
    export LD_LIBRARY_PATH=${_profiling_tools_root}/lib:$LD_LIBRARY_PATH
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-$(dirname "$(which conda)")/../}:${_profiling_tools_root}:${CMAKE_PREFIX_PATH}
fi
if [ "${PYTORCH_VERSION}" == "master_next" ];then
    WORK_ROOT=${WORK_ROOT:-$HOME/trees/jgs}
    echo -e "\n==================== Coral version ===================="
    grep -A 10 "coral:" $WORK_ROOT/build-promotions/build.yml | grep -e asset_version -e revision | sed 's/^[ \t]*//'
    grep g_commit_id $WORK_ROOT/CORAL-SIM/build/Release/coral_infra/commit_info.cpp | sed 's/^[ \t]*//'

    echo -e "\n==================== XeSim version ===================="
    grep -A 5 "cobalt:" $WORK_ROOT/build-promotions/build.yml | grep -e asset_name -e asset_version | sed 's/^[ \t]*//'

    echo -e "\n==================== Driver version ===================="
    grep "gfx_driver:" $WORK_ROOT/dependencies/NEO/config.yml | sed 's/^[ \t]*//'

    echo -e "\n==================== PTI version ===================="
    grep "Build URL" ${DEPENDENCY_DIR}/PROFILING_TOOLS_JGS/version.txt
fi

echo -e "\n==================== DPCPP version ===================="
grep "Build URL" ${DEPENDENCY_DIR}/DPCPP_JGS/version.txt

if [ "${PYTORCH_VERSION}" == "CRI_master_next" ];then
    echo -e "\n==================== Enable CRI test ENV ===================="
    source ${HOME}/env_cri.sh
fi

icpx --version
sycl-ls
