# torch-xpu-ops: XPU implementation for PyTorch ATen

# outputs:
#
#  PYTORCH_FOUND_XPU
#  -- The flag to indicate whether XPU backend stacks are setup successfully or not.
#
#  libtorch_xpu_ops
#  -- Static archive library target

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(${TORCH_XPU_OPS_PROJ_NAME} VERSION ${CMAKE_PROJECT_VERSION})

set(PYTORCH_FOUND_XPU FALSE)

if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux")
  message("torch-xpu-ops only supports Linux system so far. We will support more systems in future.")
  return()
endif()

set(TORCH_XPU_OPS_ROOT ${PROJECT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${TORCH_XPU_OPS_ROOT}/cmake/Modules)

include(${TORCH_XPU_OPS_ROOT}/cmake/SYCL.cmake)
include(${TORCH_XPU_OPS_ROOT}/cmake/BuildFlags.cmake)

if(BUILD_TEST)
  add_subdirectory(${TORCH_XPU_OPS_ROOT}/test/sycl ${CMAKE_BINARY_DIR}/test_sycl)
endif()

set(PYTORCH_FOUND_XPU TRUE)

message(STATUS "XPU found")
