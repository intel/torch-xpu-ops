# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

if(NOT __SYCLTLA_INCLUDED)
  set(__SYCLTLA_INCLUDED TRUE)
  include(FetchContent)
  FetchContent_Declare(
      repo-sycl-tla
      GIT_REPOSITORY https://github.com/intel/sycl-tla.git
      GIT_TAG        v0.6
      GIT_SHALLOW    OFF
  )
  FetchContent_GetProperties(repo-sycl-tla)
  if(NOT repo-sycl-tla_POPULATED)
    FetchContent_Populate(repo-sycl-tla)
  endif()
  set(SYCLTLA_INCLUDE_DIRS ${repo-sycl-tla_SOURCE_DIR}/include
                           ${repo-sycl-tla_SOURCE_DIR}/applications
                           ${repo-sycl-tla_SOURCE_DIR}/tools/util/include)
  set(SYCLTLA_COMPILE_DEFINITIONS CUTLASS_ENABLE_SYCL SYCL_INTEL_TARGET)
endif()
