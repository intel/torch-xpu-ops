# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

option(USE_ONEMKL_XPU "Build with ONEMKL XPU support" ON)

if(DEFINED ENV{USE_ONEMKL_XPU})
  set(USE_ONEMKL_XPU $ENV{USE_ONEMKL_XPU})
endif()

message(STATUS "USE_ONEMKL_XPU is set to ${USE_ONEMKL_XPU}")

if(NOT USE_ONEMKL_XPU)
  return()
endif()

find_package(ONEMKL)
if(NOT ONEMKL_FOUND)
  message(FATAL_ERROR "oneMKL not found or installation is incomplete; see warnings above for details.")
endif()

set(TORCH_XPU_OPS_ONEMKL_INCLUDE_DIR ${ONEMKL_INCLUDE_DIR})

set(TORCH_XPU_OPS_ONEMKL_LIBRARIES ${ONEMKL_LIBRARIES})

# mkl_intel_thread requires libiomp5 (Intel OpenMP runtime), which is shipped
# with the compiler, not MKL.  Find it from the SYCL compiler library directory.
if(NOT WIN32)
  find_library(
    IOMP5_LIBRARY
    NAMES libiomp5.so
    HINTS ${SYCL_LIBRARY_DIR}
    NO_DEFAULT_PATH)
  if(NOT IOMP5_LIBRARY)
    message(FATAL_ERROR "libiomp5.so not found in ${SYCL_LIBRARY_DIR}; "
      "ensure the Intel compiler environment is set up correctly.")
  endif()
  list(APPEND TORCH_XPU_OPS_ONEMKL_LIBRARIES ${IOMP5_LIBRARY})
endif()

# --start-group/--end-group are GNU ld options; MSVC link does not know them.
if(NOT WIN32)
  list(PREPEND TORCH_XPU_OPS_ONEMKL_LIBRARIES "-Wl,--start-group")
  list(APPEND TORCH_XPU_OPS_ONEMKL_LIBRARIES "-Wl,--end-group")
endif()
