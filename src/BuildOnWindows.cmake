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

macro(setup_common_libraries)
  add_library(
    torch_xpu_ops
    STATIC
    ${ATen_XPU_MKL_SRCS}
    ${ATen_XPU_NATIVE_CPP_SRCS})
  target_compile_definitions(torch_xpu_ops PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  target_link_libraries(torch_xpu_ops PUBLIC torch_cpu)
  target_link_libraries(torch_xpu_ops PUBLIC c10)
endmacro()

# Shared library that stores function pointers for generator operations.
# kernel DLLs call xpu_hal.dll instead of linking torch_xpu.dll directly,
# breaking the cyclic dependency.  torch_xpu.dll registers the real
# implementations at init time via xpu_hal::registerXPUGeneratorBridge().
# xpu_hal.dll only links c10 — no dependency on torch_xpu.
set(xpu_hal_srcs ${TORCH_XPU_OPS_ROOT}/src/hal/XPUHal.cpp)
if(NOT TARGET xpu_hal)
  add_library(xpu_hal SHARED ${xpu_hal_srcs})
  target_link_libraries(xpu_hal PUBLIC c10 c10_xpu)
  target_include_directories(xpu_hal PUBLIC ${TORCH_XPU_OPS_ROOT}/src)
  target_include_directories(xpu_hal PRIVATE ${ATen_XPU_INCLUDE_DIRS})
  set_target_properties(xpu_hal PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  list(APPEND TORCH_XPU_OPS_LIBRARIES xpu_hal)
  install(TARGETS xpu_hal EXPORT Caffe2Targets DESTINATION "${TORCH_INSTALL_LIB_DIR}")
endif()

if(BUILD_SEPARATE_OPS)
  setup_common_libraries()
  foreach(sycl_src ${ATen_XPU_SYCL_SRCS})
    get_filename_component(name ${sycl_src} NAME_WLE)
    set(sycl_lib torch-xpu-ops-sycl-${name})
    sycl_add_library(
      ${sycl_lib}
      SHARED
      SYCL_SOURCES ${sycl_src})
    target_link_libraries(torch_xpu_ops PUBLIC ${sycl_lib})
    # On Windows, DLL symbols are not exported by default. Each kernel DLL
    # must export its host-side SYCL entry points so consumers (torch_xpu.dll)
    # can resolve them from the import lib.
    set_target_properties(${sycl_lib} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    # kernel DLLs call xpu_hal:: functions instead of linking torch_xpu.dll
    target_link_libraries(${sycl_lib} PRIVATE xpu_hal)
    list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_lib})

    # Decouple with PyTorch cmake definition.
    install(TARGETS ${sycl_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  endforeach()
  list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops)
else()
  # On Windows, it is not possible to combine all obj files into one library
  # because the obj files of kernels compiled on Windows are much larger than
  # those on Linux. If they are combined into one, the library size will exceed
  # 4GB, which conflicts with the size limit of a single library on Windows.
  # We will combine the libraries on Windows into one after the compiler is fixed.
  add_library(
    torch_xpu_ops
    STATIC
    ${ATen_XPU_MKL_SRCS}
    ${ATen_XPU_NATIVE_CPP_SRCS})
  target_compile_definitions(torch_xpu_ops PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
 # Split SYCL kernels into 2 libraries as categories 1) Common (Unary+Binary+Reduce+Pow+Copy+Activation+Foreach) 2) Others.
  set(ATen_XPU_SYCL_COMMON_SRCS)
  set(ATen_XPU_SYCL_OTHERS_SRCS)
  foreach(sycl_src ${ATen_XPU_SYCL_SRCS})
    if(sycl_src MATCHES "(Foreach|Reduce|Unary|Binary|Copy|Pow|Activation|Norm|Loss|Resize|Distribution|Polynomial)")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    else()
      list(APPEND ATen_XPU_SYCL_OTHERS_SRCS ${sycl_src})
    endif()
  endforeach()
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

  target_link_libraries(torch_xpu_ops
      PUBLIC
      ${sycl_common_lib}
      ${sycl_lib}
  )
  target_link_options(torch_xpu_ops PUBLIC
      "-WHOLEARCHIVE:$<TARGET_FILE:${sycl_common_lib}>"
      "-WHOLEARCHIVE:$<TARGET_FILE:${sycl_lib}>"
  )
  list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops)
endif()
set(SYCL_LINK_LIBRARIES_KEYWORD)

torch_xpu_ops_finalize_targets(c10_xpu torch_cpu)
