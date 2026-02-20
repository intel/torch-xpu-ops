# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# SYCL compiler and runtime setup
if(NOT SYCLTOOLKIT_FOUND)
  # Avoid package conflict introduced in PyTorch cmake
  # find_package(SYCLToolkit REQUIRED)
  include(${TORCH_XPU_OPS_ROOT}/cmake/Modules/FindSYCLToolkit.cmake)
  if(NOT SYCLTOOLKIT_FOUND)
    message("Can NOT find SYCL compiler tool kit!")
    return()
  endif()
endif()

find_package(SYCL REQUIRED)
