# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Define archive static library target - torch_xpu_ops
include(${TORCH_XPU_OPS_ROOT}/cmake/Codegen.cmake)

set(ATen_XPU_MKL_SRCS)
set(ATen_XPU_NATIVE_CPP_SRCS)
set(ATen_XPU_SYCL_SRCS)
set(ATen_XPU_SYCLTLA_SRCS)
set(ATen_XPU_XCCL_SRCS)

set(ATen_XPU_INCLUDE_DIRS ${TORCH_XPU_OPS_ROOT}/src CACHE STRING "ATen XPU Include directory")
set(ATen_XPU_FLASH_ATTN_DIRS
  ${TORCH_XPU_OPS_ROOT}/src/ATen/native/transformers/xpu/flash_attn
  CACHE STRING "ATen XPU Flash Attention directory"
)

add_subdirectory(ATen)
if(USE_C10D_XCCL)
  add_subdirectory(xccl)
endif()
# With the increasement of bin size, we have to split libtorch_xpu.so into
# multiple libraries. Because of strict linkage requirements on Windows,
# we add extra logics to resolve, 1) Cyclic dependence, 2) Make symbols visible.
# To isolate impact and keep both maintainable, there are separate code pieces
# to setup CMAKE targets. Will revert splitting library and unify these once
# compression of device code is supported by SYCL/backend compiler.
if(WIN32)
  include(BuildOnWindows.cmake)
else()
  include(BuildOnLinux.cmake)
endif()
