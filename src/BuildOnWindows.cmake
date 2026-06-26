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
    # Kernel DLLs reference torch_xpu symbols (e.g., XPUGeneratorImpl::
    # philox_xpu_state, getDefaultXPUGenerator). These live in torch_xpu.dll
    # which is always loaded first at runtime, so symbols resolve at load time.
    # We cannot link torch_xpu.lib directly — ninja detects a build cycle:
    #   torch_xpu.dll → kernel_DLL.dll → torch_xpu.lib
    # /FORCE:UNRESOLVED suppresses LNK2019 for these runtime-resolvable symbols.
    target_link_options(${sycl_lib} PRIVATE /FORCE:UNRESOLVED)
    # On Windows, DLL symbols are not exported by default. Each kernel DLL
    # must export its host-side SYCL entry points so consumers (torch_xpu.dll)
    # can resolve them from the import lib.
    set_target_properties(${sycl_lib} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_lib})

    # Decouple with PyTorch cmake definition.
    install(TARGETS ${sycl_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  endforeach()
  # Resolve same-module cross-kernel DLL symbol references on Windows.
  # When Foo*Kernels.cpp calls a function defined in FooKernels.cpp, both
  # are separate SHARED DLLs. STATIC torch_xpu_ops cannot forward PUBLIC
  # dependencies on Windows, so we add explicit directional PRIVATE links.
  # These deps are acyclic (infra kernel → specialization only).
  set(SYCL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ATen/native/xpu/sycl")
  macro(sycl_dep caller provider)
    if(TARGET torch-xpu-ops-sycl-${caller} AND TARGET torch-xpu-ops-sycl-${provider})
      target_link_libraries(torch-xpu-ops-sycl-${caller} PRIVATE torch-xpu-ops-sycl-${provider})
    endif()
  endmacro()
  # ReduceOpsKernels → specialized reduce files
  sycl_dep(ReduceSumProdKernels ReduceOpsKernels)
  sycl_dep(ReduceMomentKernels ReduceOpsKernels)
  sycl_dep(ReduceAMinMaxKernel ReduceOpsKernels)
  sycl_dep(ReduceArgMaxKernel ReduceOpsKernels)
  sycl_dep(ReduceArgMinKernel ReduceOpsKernels)
  sycl_dep(ReduceLogicKernels ReduceOpsKernels)
  # IndexingKernels → Indexing wrappers
  sycl_dep(Indexing IndexingKernels)
  # DistributionKernels → specialized distributions
  sycl_dep(DistributionUniform DistributionKernels)
  sycl_dep(DistributionNormal DistributionKernels)
  sycl_dep(DistributionRandomKernel DistributionKernels)
  sycl_dep(DistributionBernoulli DistributionKernels)
  sycl_dep(DistributionExponentialKernel DistributionKernels)
  sycl_dep(DistributionLogNormalKernel DistributionKernels)
  sycl_dep(DistributionCauchyKernel DistributionKernels)
  sycl_dep(DistributionGeometricKernel DistributionKernels)
  # BinaryKernels → BinaryDiv specializations
  sycl_dep(BinaryDivTrueKernel BinaryKernels)
  sycl_dep(BinaryDivTruncKernel BinaryKernels)
  sycl_dep(BinaryDivFloorKernel BinaryKernels)
  # Chebyshev
  sycl_dep(ChebyshevPolynomialTKernel ChebyshevPolynomialKernels)
  sycl_dep(ChebyshevPolynomialUKernel ChebyshevPolynomialKernels)
  sycl_dep(ChebyshevPolynomialVKernel ChebyshevPolynomialKernels)
  sycl_dep(ChebyshevPolynomialWKernel ChebyshevPolynomialKernels)
  sycl_dep(ShiftedChebyshevPolynomialTKernel ShiftedChebyshevPolynomialKernels)
  sycl_dep(ShiftedChebyshevPolynomialUKernel ShiftedChebyshevPolynomialKernels)
  sycl_dep(ShiftedChebyshevPolynomialVKernel ShiftedChebyshevPolynomialKernels)
  sycl_dep(ShiftedChebyshevPolynomialWKernel ShiftedChebyshevPolynomialKernels)
  # Embedding
  sycl_dep(EmbeddingBag EmbeddingBagKernels)
  sycl_dep(Embedding EmbeddingKernels)
  # Dropout, FusedAdam, GridSampler
  sycl_dep(Dropout DropoutKernels)
  sycl_dep(FusedAdamAmsgradKernels FusedAdamKernels)
  sycl_dep(FusedAdamWAmsgradKernels FusedAdamWKernels)
  sycl_dep(GridSampler GridSamplerKernels)
  # Histogram
  sycl_dep(HistogramddKernels HistogramKernels)
  # Foreach ternary
  sycl_dep(ForeachTernaryKernels ForeachTernaryOpListKernels)
  sycl_dep(ForeachTernaryKernels ForeachTernaryOpScalarKernels)
  sycl_dep(ForeachTernaryKernels ForeachTernaryOpScalarListKernels)
  sycl_dep(ForeachTernaryKernels ForeachTernaryOpTensorKernels)
  # Cross-module utility calls
  sycl_dep(BatchNormKernels ResizeKernel)
  sycl_dep(UnaryComplexKernels CopyKernel)
  sycl_dep(UnaryComplexKernels UnarySignKernels)
  sycl_dep(PowKernels UnaryKernels)
  sycl_dep(PowKernels UnaryFractionKernels)
  sycl_dep(Shape ShapeKernels)
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
    string(REGEX MATCH "Binary" IS_BINARY ${sycl_src})
    string(REGEX MATCH "Unary" IS_UNARY ${sycl_src})
    string(REGEX MATCH "Pow" IS_POW ${sycl_src})
    string(REGEX MATCH "Copy" IS_COPY ${sycl_src})
    string(REGEX MATCH "Reduce" IS_REDUCE ${sycl_src})
    string(REGEX MATCH "Activation" IS_ACTIVATION ${sycl_src})
    string(REGEX MATCH "Foreach" IS_FOREACH ${sycl_src})
    string(REGEX MATCH "Polynomial" IS_POLY ${sycl_src})
    string(REGEX MATCH "Norm" IS_NORM ${sycl_src})
    string(REGEX MATCH "Loss" IS_LOSS ${sycl_src})
    string(REGEX MATCH "Resize" IS_RESIZE ${sycl_src})
    string(REGEX MATCH "Distribution" IS_DISTRIBUTION ${sycl_src})

    if(NOT IS_FOREACH STREQUAL "")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    elseif(NOT IS_REDUCE STREQUAL "")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    elseif(NOT IS_UNARY STREQUAL "" OR NOT IS_BINARY STREQUAL "")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    elseif(NOT IS_COPY STREQUAL "" OR NOT IS_POW STREQUAL "")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    elseif(NOT IS_ACTIVATION STREQUAL "")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    elseif(NOT IS_NORM STREQUAL "")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    elseif(NOT IS_LOSS STREQUAL "")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    elseif(NOT IS_RESIZE STREQUAL "")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    elseif(NOT IS_DISTRIBUTION STREQUAL "")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    elseif(NOT IS_POLY STREQUAL "")
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

foreach(lib ${TORCH_XPU_OPS_LIBRARIES})
  # Align with PyTorch compile options PYTORCH_SRC_DIR/cmake/public/utils.cmake
  torch_compile_options(${lib})
  target_compile_options_if_supported(${lib} "-Wno-deprecated-copy")
  target_compile_options(${lib} PRIVATE ${TORCH_XPU_OPS_FLAGS})

  target_include_directories(${lib} PUBLIC ${TORCH_XPU_OPS_INCLUDE_DIRS})
  target_include_directories(${lib} PUBLIC ${ATen_XPU_INCLUDE_DIRS})
  target_include_directories(${lib} PUBLIC ${SYCL_INCLUDE_DIR})

  target_link_libraries(${lib} PUBLIC ${SYCL_LIBRARY})
  target_link_libraries(${lib} PUBLIC c10_xpu)
  target_link_libraries(${lib} PUBLIC torch_cpu)
  target_link_libraries(${lib} PRIVATE ATEN_XPU_FILES_GEN_LIB)
endforeach()
