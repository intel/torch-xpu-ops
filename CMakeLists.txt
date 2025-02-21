# torch-xpu-ops: XPU implementation for PyTorch ATen

# inputs:
#
#  TORCH_XPU_OPS_INCLUDE_DIRS
#  -- As a PyTorch "submodule", need configure the variable to inform
#     PyTorch includes.
#
#  PYTORCH_INSTALL_DIR
#  -- In "standalone" build mode, need input the variable for deducing PyTorch
#     compilation flags and includes.

# outputs:
#
#  TORCH_XPU_OPS_FOUND
#  -- The flag to indicate whether ATen XPU impl are included successfully or not.
#
#  torch_xpu_ops
#  -- Static archive library target

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(PROJECT_NAME "torch-xpu-ops")
set(PROJECT_VERSION "2.3.0")
# Avoid SYCL compiler error
if(NOT WIN32)
  string(APPEND CMAKE_CXX_FLAGS " -Wno-error")
endif()

cmake_policy(SET CMP0048 NEW)
project(${PROJECT_NAME} VERSION "${PROJECT_VERSION}" LANGUAGES C CXX)

set(TORCH_XPU_OPS_FOUND FALSE)

set(TORCH_XPU_OPS_ROOT ${PROJECT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${TORCH_XPU_OPS_ROOT}/cmake/Modules)

include(${TORCH_XPU_OPS_ROOT}/cmake/SYCL.cmake)
include(${TORCH_XPU_OPS_ROOT}/cmake/ONEMKL.cmake)
include(${TORCH_XPU_OPS_ROOT}/cmake/BuildFlags.cmake)

option(USE_XCCL "Build with XCCL support" OFF)
option(USE_C10D_XCCL "Build with XCCL support for C10D" OFF)

# -- [ Re-generate the macros file for https://github.com/pytorch/pytorch/pull/147161
macro(update_caffe2_macros_file)
  configure_file(
    ${TORCH_ROOT}/caffe2/core/macros.h.in
    ${CMAKE_BINARY_DIR}/caffe2/core/macros.h)
endmacro()

if(NOT DEFINED USE_XCCL)
  # Propagate the option to PyTorch
  caffe2_update_option(USE_XCCL OFF)
  update_caffe2_macros_file()
endif()

if(USE_XCCL)
  include(${TORCH_XPU_OPS_ROOT}/cmake/XCCL.cmake)
  if(NOT PYTORCH_FOUND_XCCL)
    # Propagate the option to PyTorch
    caffe2_update_option(USE_XCCL OFF)
    update_caffe2_macros_file()
  endif()
endif()

# Propagate the option to PyTorch
if(USE_DISTRIBUTED AND USE_XCCL AND USE_C10D_XCCL)
  caffe2_update_option(USE_C10D_XCCL ON)
else()
  caffe2_update_option(USE_C10D_XCCL OFF)
endif()

if(BUILD_TEST)
  add_subdirectory(${TORCH_XPU_OPS_ROOT}/test/sycl ${CMAKE_BINARY_DIR}/test_sycl)
endif()

if(NOT TORCH_ROOT)
  message("Can NOT find TORCH_ROOT, please integrate torch-xpu-ops in PyTorch correctly ...")
  return()
endif()

if(NOT TORCH_XPU_OPS_INCLUDE_DIRS)
  message("Can NOT find TORCH_XPU_OPS_INCLUDE_DIRS, require PyTorch includes ...")
  return()
endif()

if(TORCH_XPU_OPS_PYTORCH_DEPS)
  sycl_include_external_dependencies(${TORCH_XPU_OPS_PYTORCH_DEPS})
endif()

# Only for debugging. Save building time by shrinking translation unit scope.
set(BUILD_SEPARATE_OPS $ENV{BUILD_SEPARATE_OPS})
set(BUILD_SPLIT_KERNEL_LIB $ENV{BUILD_SPLIT_KERNEL_LIB})
add_subdirectory(${TORCH_XPU_OPS_ROOT}/src)

set(TORCH_XPU_OPS_FOUND TRUE)
