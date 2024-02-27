# torch-xpu-ops: XPU implementation for PyTorch ATen

# inputs:
#
#  TORCH_XPU_OPS_BUILD_MODE
#  -- "submodule" or "standalone", "submodule" by default
#
#  TORCH_XPU_OPS_PYTORCH_INCLUDE_DIRS
#  -- In "submodule" build mode, need configure the variable to inform
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
cmake_policy(SET CMP0048 NEW)
project(${PROJECT_NAME} VERSION "${PROJECT_VERSION}" LANGUAGES C CXX)

if(NOT TORCH_XPU_OPS_BUILD_MODE)
  set(TORCH_XPU_OPS_BUILD_MODE "submodule")
endif()

set(TORCH_XPU_OPS_FOUND FALSE)

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
# 1. standalone - Need pre-install PyTorch. Introduce PyTorch dependencies through PyTorch installation dirctory.
# 2. submodule - Build as a submodule of PyTorch. Introduce PyTorch dependecies through PyTorch sources directory.
if(TORCH_XPU_OPS_BUILD_MODE MATCHES "submodule")
  # submodule
  if(NOT TORCH_XPU_OPS_PYTORCH_INCLUDE_DIRS)
    message("Can NOT find TORCH_XPU_OPS_PYTORCH_INCLUDE_DIRS in submodule build mode, please set the varaible before including torch-xpu-ops root CMakeLists.txt")
    return()
  endif()
elseif(TORCH_XPU_OPS_BUILD_MODE MATCHES "standalone")
  # standalone
  set(Torch_COMP_VERION ${PROJECT_VERSION})

  if(NOT PYTORCH_INSTALL_DIR)
    message(FATAL_ERROR "Can NOT find PYTORCH_INSTALL_DIR in standalone build mode, please set -DPYTORCH_INSTALL_DIR ...")
  endif()

  set(Torch_DIR ${PYTORCH_INSTALL_DIR}/share/cmake/Torch)
  find_package(Torch REQUIRED)

  set(Caffe2_DIR ${PYTORCH_INSTALL_DIR}/share/cmake/Caffe2)
  find_package(Caffe2 REQUIRED)

  set(TORCH_XPU_OPS_PYTORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS} ${CAFFE2_INCLUDE_DIRS})
else()
  message("Unknown build mode. Please see README for build.")
endif()

if(TORCH_XPU_OPS_PYTORCH_DEPS)
  sycl_include_external_dependencies(${TORCH_XPU_OPS_PYTORCH_DEPS})
endif()

add_subdirectory(${TORCH_XPU_OPS_ROOT}/src)

set(TORCH_XPU_OPS_FOUND TRUE)
