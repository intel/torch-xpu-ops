# Copyright 2020-2025 Intel Corporation
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
  set_property(
    TARGET torch::xccl PROPERTY INTERFACE_LINK_LIBRARIES
    ${XCCL_LIBRARY} ${XCCL_LIBRARY_2_0})
endif()
