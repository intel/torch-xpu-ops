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
  target_link_libraries(torch_xpu_ops PUBLIC torch_xpu torch_cpu c10)
  setup_common_libraries()
  # torch-xpu-ops-sycl-Comm-kernels includes SYCL kernels shared in different other SYCL kernels.
  # SYCL free func: These SYCL kernels are moved from headers into torch-xpu-ops-sycl-Comm-kernels to resolve 'redefinition issue'.
  sycl_add_library(
      torch-xpu-ops-sycl-Comm-kernels
      SHARED
      SYCL_SOURCES ${ATen_XPU_SYCL_COMM_SRCS})
  list(APPEND TORCH_XPU_OPS_LIBRARIES torch-xpu-ops-sycl-Comm-kernels)
  # Decouple with PyTorch cmake definition.
  install(TARGETS torch-xpu-ops-sycl-Comm-kernels DESTINATION "${TORCH_INSTALL_LIB_DIR}")

  foreach(sycl_src ${ATen_XPU_SYCL_SRCS})
    cmake_path(GET sycl_src STEM LAST_ONLY name)
    set(sycl_lib torch-xpu-ops-sycl-${name})
    sycl_add_library(
      ${sycl_lib}
      SHARED
      SYCL_SOURCES ${sycl_src})
    target_link_libraries(torch_xpu_ops_aten PUBLIC ${sycl_lib})
    target_link_libraries(${sycl_lib} PUBLIC torch-xpu-ops-sycl-Comm-kernels)
    list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_lib})

    # Decouple with PyTorch cmake definition.
    install(TARGETS ${sycl_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  endforeach()
  list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops)

else()
  # On Windows the SYCL kernel objects cannot be combined into a single static
  # library: their combined size exceeds the 4GB limit of a Windows static
  # library. The kernel sources are therefore distributed over several static
  # libraries so no single archive approaches 4GB

  # Common kernels (native/xpu/sycl/comm/*.cpp) hold symbols shared by the kernel
  # libraries, so they get their own library that every kernel library links.
  set(sycl_common_lib torch_xpu_ops_sycl_common_kernels)
  sycl_add_library(
    ${sycl_common_lib}
    STATIC
    SYCL_SOURCES ${ATen_XPU_SYCL_COMM_SRCS})
  target_compile_definitions(${sycl_common_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_common_lib})

  # Kernel objects must be kept so their static initializers register the
  # kernels, hence WHOLE_ARCHIVE for every kernel library below.
  set(_wholearchive_libs ${sycl_common_lib})

  # Split the SYCL kernel sources across a fixed set of static libraries, one per
  # named group, by matching the source file name against a regex.
  # The regexes are evaluated in order and the first match wins, so the trailing
  # catch-all ".*" group collects everything not claimed by an earlier group.
  set(_sycl_group_names
    binary
    elementwise_nn
    unary_special
    reduce_activation
    foreach_index_sort
    misc)
  # Binary elementwise kernels (heaviest single family)
  set(_sycl_group_regex_binary
    "^Binary|^NestedTensorBinaryOpsKernels|^SparseBinaryOpIntersectionKernels")
  # Remaining elementwise ops + embedding + the NN layers (pool/conv/upsample/...)
  set(_sycl_group_regex_elementwise_nn
    "^Pow|^MaxMinElementwise|^PointwiseOps|^StepKernels|^Copy|^Fill|^Lerp|^Copysign|^Abs|^GcdLcm|^LogAddExp|^Complex|^Embedding|Pool|^UpSample|Conv|^ReflectionPad|^ReplicationPad|Roi|^NMS|^Im2Col|^Col2Im|^RNN|^Rrelu|^Attention|^Distance|^GridSampler|^MaxUnpooling")
  # Unary + special functions + distributions + fused optimizers
  set(_sycl_group_regex_unary_special
    "^Unary|ChebyshevPolynomial|HermitePolynomial|LegendrePolynomial|LaguerrePolynomial|Bessel|AiryAi|Zeta|IGamma|SphericalBessel|^Distribution|^Fused")
  # Reductions + activations + triangular + sparse
  set(_sycl_group_regex_reduce_activation
    "^Reduce|^Activation|^Triangular|^Sparse")
  # Foreach + indexing + sort/topk + loss
  set(_sycl_group_regex_foreach_index_sort
    "^Foreach|^Indexing|TopK|^Sorting|^Loss|MultiMargin|MultiLabel")
  # Everything else (compare, norm, scan, shape, linalg, quantization, misc)
  set(_sycl_group_regex_misc ".*")

  # Bucket every kernel source into the first group whose regex matches its name
  foreach(_group ${_sycl_group_names})
    set(_sycl_group_srcs_${_group})
  endforeach()
  foreach(sycl_src ${ATen_XPU_SYCL_SRCS})
    get_filename_component(_name ${sycl_src} NAME_WLE)
    foreach(_group ${_sycl_group_names})
      if(_name MATCHES "${_sycl_group_regex_${_group}}")
        list(APPEND _sycl_group_srcs_${_group} ${sycl_src})
        break()
      endif()
    endforeach()
  endforeach()

  # One static library per group.
  #
  # The CMake target names are short (xpuk0, xpuk1, ...); the descriptive group
  # name is applied to the final .lib via OUTPUT_NAME. Reason: FindSYCL derives
  # the generated object path from the target name and pays for it twice:
  #   <bin dir>/CMakeFiles/<target>.dir[/<config>]/<target>_gen_<hash>_<source>.obj
  # so a descriptive target name (e.g. torch_xpu_ops_sycl_foreach_index_sort_kernels,
  # 45 characters) embedded twice costs ~90 of the 260 characters of the Windows
  # MAX_PATH limit; the longest kernel source names then leave almost no room for
  # the build directory and lib.exe fails to read the object back (LNK1181).
  set(_group_index 0)
  foreach(_group ${_sycl_group_names})
    if(NOT _sycl_group_srcs_${_group})
      math(EXPR _group_index "${_group_index} + 1")
      continue()
    endif()
    set(sycl_lib xpuk${_group_index})
    sycl_add_library(${sycl_lib} STATIC SYCL_SOURCES ${_sycl_group_srcs_${_group}})
    set_target_properties(${sycl_lib} PROPERTIES OUTPUT_NAME torch_xpu_ops_sycl_${_group}_kernels)
    target_compile_definitions(${sycl_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
    target_link_libraries(${sycl_lib} PUBLIC ${sycl_common_lib})
    list(APPEND _wholearchive_libs ${sycl_lib})
    list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_lib})
    math(EXPR _group_index "${_group_index} + 1")
  endforeach()

  string(REPLACE ";" "," _wholearchive_list "${_wholearchive_libs}")
  target_link_libraries(torch_xpu_ops PUBLIC
      "$<LINK_LIBRARY:WHOLE_ARCHIVE,${_wholearchive_list}>")

  list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops)
endif()
set(SYCL_LINK_LIBRARIES_KEYWORD)

torch_xpu_ops_finalize_targets(c10_xpu torch_cpu)
