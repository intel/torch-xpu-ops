# SYCL compiler and runtime setup
if(NOT SYCLTOOLKIT_FOUND)
  # Avoid package conflict introduced in PyTorch cmake
  # find_package(SYCLToolkit REQUIRED)
  include(${TORCH_XPU_OPS_ROOT}/cmake/Modules/FindSYCLToolkit.cmake)
  if(NOT SYCLTOOLKIT_FOUND)
    message("Can NOT find SYCL compiler tool kit!")
    return()
  endif()
endif()

# Try to find SYCL compiler version.hpp header
find_file(SYCL_VERSION
    NAMES version.hpp
    PATHS
        ${SYCL_INCLUDE_DIR}
    PATH_SUFFIXES
        sycl
        sycl/CL
        sycl/CL/sycl
    NO_DEFAULT_PATH)

if(NOT SYCL_VERSION)
  message("Can NOT find SYCL version file!")
  return()
endif()

find_library(SYCL_LIBRARIES sycl-preview HINTS ${SYCL_LIBRARY_DIR})
# On Windows, currently there's no sycl.lib. Only sycl7.lib with version suffix,
# where the current version of the SYCL runtime is 7.
# Until oneAPI adds support to sycl.lib without the version suffix,
# sycl_runtime_version needs to be hardcoded and uplifted when SYCL runtime version uplifts.
# TODO: remove this when sycl.lib is supported on Windows
if(WIN32)
  set(sycl_runtime_version 7)
  find_library(
    SYCL_LIBRARIES
    NAMES "sycl${sycl_runtime_version}"
    HINTS ${SYCL_LIBRARY_DIR}
  )
  if(SYCL_LIBRARIES STREQUAL "SYCL_LIBRARIES-NOTFOUND")
    message(FATAL_ERROR "Cannot find a SYCL library on Windows")
  endif()
endif()

set(SYCL_COMPILER_VERSION)
file(READ ${SYCL_VERSION} version_contents)
string(REGEX MATCHALL "__SYCL_COMPILER_VERSION +[0-9]+" VERSION_LINE "${version_contents}")
list(LENGTH VERSION_LINE ver_line_num)
if(${ver_line_num} EQUAL 1)
  string(REGEX MATCHALL "[0-9]+" SYCL_COMPILER_VERSION "${VERSION_LINE}")
endif()

# offline compiler of SYCL compiler
set(IGC_OCLOC_VERSION)
find_program(OCLOC_EXEC ocloc)
if(OCLOC_EXEC)
  set(drv_ver_file "${PROJECT_BINARY_DIR}/OCL_DRIVER_VERSION")
  file(REMOVE ${drv_ver_file})
  execute_process(COMMAND ${OCLOC_EXEC} query OCL_DRIVER_VERSION WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  if(EXISTS ${drv_ver_file})
    file(READ ${drv_ver_file} drv_ver_contents)
    string(STRIP "${drv_ver_contents}" IGC_OCLOC_VERSION)
  endif()
endif()

find_package(SYCL REQUIRED)
if(NOT SYCL_FOUND)
  message("Can NOT find SYCL cmake helpers module!")
  return()
endif()
