# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# XCCL sources

file(GLOB xccl_h "*.hpp")
file(GLOB xccl_cpp "*.cpp")
list(REMOVE_ITEM xccl_cpp "${CMAKE_CURRENT_SOURCE_DIR}/NanCheck_XPU.cpp")

list(APPEND ATen_XPU_XCCL_SRCS ${xccl_cpp})
list(APPEND ATen_XPU_SYCL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/NanCheck_XPU.cpp")

set(ATen_XPU_XCCL_SRCS ${ATen_XPU_XCCL_SRCS} PARENT_SCOPE)
set(ATen_XPU_SYCL_SRCS ${ATen_XPU_SYCL_SRCS} PARENT_SCOPE)

# Why copy the header file to the build directory?
# We want register XCCL backend to PyTorch c10d in torch/csrc/distributed/c10d/init.cpp#L27-L29.
# To align with other backends, we need to copy the header file to the build torch/csrc/distributed/c10d directory.
# Further solution is add find path for torch/csrc/distributed/c10d/init.cpp#L27-L29.
foreach(HEADER  ${xccl_h})
  file(COPY ${HEADER} DESTINATION "${CMAKE_BINARY_DIR}/torch/csrc/distributed/c10d")
endforeach()
