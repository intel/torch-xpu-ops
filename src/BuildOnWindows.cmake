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
  # Every SYCL kernel source becomes its own DLL, mirroring Linux
  # configuration. On Windows however, a DLL must resolve all its symbols
  # when linked, but torch_xpu.dll and the kernel DLLs call each other
  # (torch_xpu <-> kernels, kernel <-> kernel), so none of them can be linked
  # first. An import library needs only the exported names, which the object
  # files already carry, so create all of them up front with lib /DEF
  # ("pre-link import libraries"). The build then runs:
  #   1. compile all objects
  #   2. pre_implib/<dll>.lib for torch_xpu and every kernel DLL
  #   3. link the kernel DLLs against those
  #   4. link torch_xpu.dll against the kernel DLLs' import libraries
  # Kernel DLLs depend on the pre-link import library files only, never on
  # another DLL target (not even Comm): that would wait for its link and
  # recreate the cycle. Only Ninja supports such file-level dependencies.
  if(NOT CMAKE_GENERATOR MATCHES "Ninja")
    message(FATAL_ERROR
      "Windows separate-ops builds (BUILD_SEPARATE_OPS, Debug/RelWithDebInfo) "
      "require a Ninja generator, got '${CMAKE_GENERATOR}'.")
  endif()
  if(NOT TARGET torch_xpu)
    message(FATAL_ERROR
      "Windows separate-ops builds need the torch_xpu target to be defined "
      "before torch-xpu-ops is added.")
  endif()

  # torch_xpu_ops is whole-archived into torch_xpu itself, so linking
  # torch_xpu here would make a cycle, linking only torch_cpu here.
  target_link_libraries(torch_xpu_ops PUBLIC torch_cpu c10)

  set(_pre_implib_dir "${CMAKE_CURRENT_BINARY_DIR}/pre_implib")
  set(_pre_implib_rsp_suffix "")
  if(CMAKE_CONFIGURATION_TYPES)
    string(APPEND _pre_implib_dir "/$<CONFIG>")
    set(_pre_implib_rsp_suffix ".$<CONFIG>")
  endif()

  # Import library of <dll_target> from its object files. /Brepro and
  # copy_if_different keep it unchanged unless the exports change, so DLLs
  # linking it are not relinked needlessly.
  function(_torch_xpu_ops_add_pre_implib out_var dll_target)
    set(_lib "${_pre_implib_dir}/${dll_target}.lib")
    set(_tmp_lib "${_pre_implib_dir}/tmp/${dll_target}.lib")
    set(_rsp "${CMAKE_CURRENT_BINARY_DIR}/pre_implib/${dll_target}${_pre_implib_rsp_suffix}.rsp")
    file(GENERATE
      OUTPUT "${_rsp}"
      CONTENT "\"$<JOIN:${ARGN},\"\n\">\"\n")
    add_custom_command(
      OUTPUT "${_lib}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${_pre_implib_dir}/tmp"
      COMMAND "${CMAKE_AR}" /nologo /Brepro /DEF
        "/NAME:$<TARGET_FILE_NAME:${dll_target}>" "/OUT:${_tmp_lib}" "@${_rsp}"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_tmp_lib}" "${_lib}"
      DEPENDS ${ARGN} "${_rsp}"
      COMMENT "Generating pre-link import library for ${dll_target}"
      VERBATIM)
    set(${out_var} "${_lib}" PARENT_SCOPE)
  endfunction()

  # torch_python imports at::xpu::sleep from torch_xpu.dll.
  set(_sycl_srcs_in_torch_xpu_regex "^Sleep$")

  # torch-xpu-ops-sycl-Comm-kernels includes SYCL kernels shared in different other SYCL kernels.
  # SYCL free func: These SYCL kernels are moved from headers into torch-xpu-ops-sycl-Comm-kernels to resolve 'redefinition issue'.
  set(sycl_comm_lib torch-xpu-ops-sycl-Comm-kernels)
  sycl_add_library(
      ${sycl_comm_lib}
      SHARED
      SYCL_SOURCES ${ATen_XPU_SYCL_COMM_SRCS})
  set(_sycl_dlls ${sycl_comm_lib})
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_comm_lib})

  foreach(sycl_src ${ATen_XPU_SYCL_SRCS})
    cmake_path(GET sycl_src STEM LAST_ONLY name)
    set(sycl_lib torch-xpu-ops-sycl-${name})
    if(name MATCHES "${_sycl_srcs_in_torch_xpu_regex}")
      sycl_add_library(
        ${sycl_lib}
        STATIC
        SYCL_SOURCES ${sycl_src})
      target_compile_definitions(${sycl_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
      target_link_libraries(torch_xpu_ops PUBLIC
          "$<LINK_LIBRARY:WHOLE_ARCHIVE,${sycl_lib}>")
    else()
      sycl_add_library(
        ${sycl_lib}
        SHARED
        SYCL_SOURCES ${sycl_src})
      target_link_libraries(torch_xpu_ops PUBLIC ${sycl_lib})
      list(APPEND _sycl_dlls ${sycl_lib})
    endif()
    list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_lib})
  endforeach()
  list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops)

  _torch_xpu_ops_add_pre_implib(_torch_xpu_pre_implib torch_xpu
      "$<TARGET_OBJECTS:torch_xpu>")
  # Only emits the command; nothing depends on this target.
  add_custom_target(torch_xpu_pre_implib DEPENDS ${_torch_xpu_pre_implib})
  set(_pre_implibs ${_torch_xpu_pre_implib})
  foreach(_dll ${_sycl_dlls})
    get_target_property(_dll_srcs ${_dll} SOURCES)
    list(FILTER _dll_srcs INCLUDE REGEX "\\.obj$")
    _torch_xpu_ops_add_pre_implib(_dll_pre_implib ${_dll} ${_dll_srcs})
    # On any other target it would drag in the DLL's SYCL compile commands.
    target_sources(${_dll} PRIVATE ${_dll_pre_implib})
    set(_pre_implib_of_${_dll} ${_dll_pre_implib})
    list(APPEND _pre_implibs ${_dll_pre_implib})
  endforeach()

  foreach(_dll ${_sycl_dlls})
    # Export the TORCH_XPU_API kernel entry points from the DLL.
    target_compile_definitions(${_dll} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
    set(_dll_pre_implibs ${_pre_implibs})
    list(REMOVE_ITEM _dll_pre_implibs ${_pre_implib_of_${_dll}})
    target_link_libraries(${_dll} PRIVATE ${_dll_pre_implibs})

    # Decouple with PyTorch cmake definition.
    install(TARGETS ${_dll} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
    if(MSVC)
      install(FILES $<TARGET_PDB_FILE:${_dll}>
          DESTINATION "${TORCH_INSTALL_LIB_DIR}" OPTIONAL)
    endif()
  endforeach()

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
