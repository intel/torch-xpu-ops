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
  # oneCCL declares all C-API symbols (oneccl*) with __attribute__((weak)) in
  # <oneapi/ccl.h>. Under the default --as-needed link policy this would cause
  # the libccl shared objects to be dropped from libtorch_xpu.so dependencies
  # (no strong undefined reference is generated), and the weak symbols would
  # resolve to NULL at runtime -> segfault on the first oneccl* call.
  # Wrap them with --no-as-needed,...,--as-needed so they remain in NEEDED.
  set_property(
    TARGET torch::xccl PROPERTY INTERFACE_LINK_LIBRARIES
    "-Wl,--no-as-needed,${XCCL_LIBRARY},${XCCL_LIBRARY_2_0},--as-needed")
endif()
