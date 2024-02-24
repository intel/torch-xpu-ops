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

# The library couples with PyTorch. Here are two possible building processes:
# 1. standalone - Need pre-install PyTorch. Introduce PyTorch depdenceise through PyTorch installation dirctory.
# 2. submodule - Build as a submodule of PyTorch. Introduce PyTorch dependecies through PyTorch sources directory.
if(PYTORCH_USE_XPU)
  # submodule
else()
  # standalone
  set(Torch_COMP_VERION "2.3.0")

  if(NOT PYTORCH_INSTALL_DIR)
    message(FATAL_ERROR "Cannot find PYTORCH_INSTALL_DIR in standalone build mode, please set -DPYTORCH_INSTALL_DIR ...")
  endif()

  set(Torch_DIR ${PYTORCH_INSTALL_DIR}/share/cmake/Torch)
  find_package(Torch REQUIRED)

  set(Caffe2_DIR ${PYTORCH_INSTALL_DIR}/share/cmake/Caffe2)
  find_package(Caffe2 REQUIRED)

  set(PYTORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS} ${CAFFE2_INCLUDE_DIRS})
endif()

add_subdirectory(${TORCH_XPU_OPS_ROOT}/src ${CMAKE_BINARY_DIR}/torch_xpu_ops)

set(PYTORCH_FOUND_XPU TRUE)

message(STATUS "XPU found")
