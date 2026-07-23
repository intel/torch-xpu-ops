# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Build on Windows

set(TORCH_XPU_OPS_LIBRARIES)
set(SYCL_LINK_LIBRARIES_KEYWORD PRIVATE)

add_library(
  torch_xpu_ops
  STATIC
  ${ATen_XPU_MKL_SRCS}
  ${ATen_XPU_NATIVE_CPP_SRCS})
target_compile_definitions(torch_xpu_ops PRIVATE TORCH_XPU_BUILD_MAIN_LIB)

if(BUILD_SEPARATE_OPS)
  # Do not link torch_xpu here: torch_xpu whole-archive-links torch_xpu_ops
  # (see PyTorch's caffe2/CMakeLists.txt), so a torch_xpu_ops -> torch_xpu
  # edge would form a link cycle. torch_cpu/c10 have no such reverse edge.
  target_link_libraries(torch_xpu_ops PUBLIC torch_cpu c10)
  set(_sycl_libs_for_rsp)
  foreach(sycl_src ${ATen_XPU_SYCL_SRCS})
    get_filename_component(name ${sycl_src} NAME_WLE)
    set(sycl_lib torch-xpu-ops-sycl-${name})
    # Keep split SYCL targets static on Windows so torch_xpu remains the only
    # shared-library link that resolves cross-target XPU helper symbols.
    sycl_add_library(
      ${sycl_lib}
      STATIC
      SYCL_SOURCES ${sycl_src})
    target_link_libraries(torch_xpu_ops PUBLIC ${sycl_lib})
    list(APPEND _sycl_libs_for_rsp ${sycl_lib})
    list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_lib})

    # Decouple with PyTorch cmake definition.
    install(TARGETS ${sycl_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  endforeach()
  list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops)

  # On Windows with many separate SYCL libs, passing all -WHOLEARCHIVE flags
  # directly on the command line exceeds the OS limit. Write them to a linker
  # response file and expose it via torch_xpu_ops INTERFACE link options so any
  # consumer (torch_xpu) picks it up without modifying PyTorch's CMakeLists.
  set(_wholearchive_rsp "${CMAKE_BINARY_DIR}/torch_xpu_sycl_wholearchive.rsp")
  set(_wholearchive_content "")
  foreach(_lib IN LISTS _sycl_libs_for_rsp)
    string(APPEND _wholearchive_content "-WHOLEARCHIVE:$<TARGET_FILE:${_lib}>\n")
  endforeach()
  file(GENERATE
    OUTPUT "${_wholearchive_rsp}"
    CONTENT "${_wholearchive_content}")
  target_link_options(torch_xpu_ops INTERFACE "@${_wholearchive_rsp}")
else()
  # On Windows, it is not possible to combine all obj files into one library
  # because the obj files of kernels compiled on Windows are much larger than
  # those on Linux. If they are combined into one, the library size will exceed
  # 4GB, which conflicts with the size limit of a single library on Windows.
  # We will combine the libraries on Windows into one after the compiler is fixed.
  # Split SYCL kernels into 2 libraries: common kernels (matched below) vs. the rest.
  set(sycl_common_regex "(Foreach|Reduce|Unary|Binary|Copy|Pow|Activation|Norm|Loss|Resize|Distribution|Polynomial)")
  set(ATen_XPU_SYCL_COMMON_SRCS ${ATen_XPU_SYCL_SRCS})
  set(ATen_XPU_SYCL_OTHERS_SRCS ${ATen_XPU_SYCL_SRCS})
  list(FILTER ATen_XPU_SYCL_COMMON_SRCS INCLUDE REGEX "${sycl_common_regex}")
  list(FILTER ATen_XPU_SYCL_OTHERS_SRCS EXCLUDE REGEX "${sycl_common_regex}")
  # Common kernel lib
  set(sycl_common_lib torch_xpu_ops_sycl_common_kernels)
  sycl_add_library(
    ${sycl_common_lib}
    STATIC
    SYCL_SOURCES ${ATen_XPU_SYCL_COMMON_SRCS})
  target_compile_definitions(${sycl_common_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_common_lib})

  # Other kernel lib
  set(sycl_lib torch_xpu_ops_sycl_kernels)
  sycl_add_library(
    ${sycl_lib}
    STATIC
    SYCL_SOURCES ${ATen_XPU_SYCL_OTHERS_SRCS})
  target_compile_definitions(${sycl_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_lib})

  target_link_libraries(torch_xpu_ops PUBLIC
      "$<LINK_LIBRARY:WHOLE_ARCHIVE,${sycl_common_lib},${sycl_lib}>"
  )
  list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops)
endif()
set(SYCL_LINK_LIBRARIES_KEYWORD)

torch_xpu_ops_finalize_targets(c10_xpu torch_cpu)
