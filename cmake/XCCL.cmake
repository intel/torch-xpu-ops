# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

if(NOT __XCCL_INCLUDED)
  set(__XCCL_INCLUDED TRUE)

  # XCCL_ROOT, XCCL_LIBRARY_DIR, XCCL_INCLUDE_DIR are handled by FindXCCL.cmake.
  find_package(XCCL REQUIRED)
  if(NOT XCCL_FOUND)
    set(PYTORCH_FOUND_XCCL FALSE)
    message(WARNING "${XCCL_NOT_FOUND_MESSAGE}")
    return()
  endif()

  set(PYTORCH_FOUND_XCCL TRUE)
  add_library(torch::xccl INTERFACE IMPORTED)
  set_property(
    TARGET torch::xccl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${XCCL_INCLUDE_DIR})
  # Wrap with --no-as-needed,...,--as-needed so it remains in NEEDED.
  # Once the weak attribute is removed upstream, this wrapper can be dropped.
  set_property(
    TARGET torch::xccl PROPERTY INTERFACE_LINK_LIBRARIES
    "-Wl,--no-as-needed,${XCCL_LIBRARY},--as-needed")

  # onecclAllToAllV was added during the oneCCL 2022.2 cycle, so the version
  # macros cannot tell whether a given 2022.2 header provides it. Probe the
  # header instead. decltype avoids needing the symbol at link time.
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES ${XCCL_INCLUDE_DIR})
  check_cxx_source_compiles("
    #include <oneapi/ccl.h>
    using alltoallv_t = decltype(&onecclAllToAllV);
    int main() { return 0; }
    " XCCL_HAS_ALLTOALLV)
  unset(CMAKE_REQUIRED_INCLUDES)
  if(XCCL_HAS_ALLTOALLV)
    set_property(
      TARGET torch::xccl APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
      XCCL_HAS_ALLTOALLV)
  endif()
endif()
