if(Codegen_XPU_cmake_included)
  return()
endif()
set(Codegen_XPU_cmake_included true)

set(BUILD_TORCH_XPU_ATEN_GENERATED "${CMAKE_BINARY_DIR}/xpu/ATen")
file(MAKE_DIRECTORY ${BUILD_TORCH_XPU_ATEN_GENERATED})

set(RegisterXPU_GENERATED ${BUILD_TORCH_XPU_ATEN_GENERATED}/RegisterXPU_0.cpp)
set(RegisterSparseXPU_GENERATED ${BUILD_TORCH_XPU_ATEN_GENERATED}/RegisterSparseXPU_0.cpp)
set(RegisterSparseCsrXPU_GENERATED ${BUILD_TORCH_XPU_ATEN_GENERATED}/RegisterSparseCsrXPU_0.cpp)
set(RegisterNestedTensorXPU_GENERATED ${BUILD_TORCH_XPU_ATEN_GENERATED}/RegisterNestedTensorXPU_0.cpp)
set(XPUFallback_TEMPLATE ${TORCH_XPU_OPS_ROOT}/src/ATen/native/xpu/XPUFallback.template)
set(XPU_AOTI_INSTALL_DIR ${TORCH_ROOT}/torch/csrc/inductor/aoti_torch/generated/extend)
set(XPU_AOTI_SHIM_HEADER ${XPU_AOTI_INSTALL_DIR}/c_shim_xpu.h)
set(XPU_AOTI_SHIM_SOURCE ${XPU_AOTI_INSTALL_DIR}/c_shim_xpu.cpp)

if(WIN32)
  set(FILE_DISPLAY_CMD type)
else()
  set(FILE_DISPLAY_CMD cat)
endif()
file(TO_NATIVE_PATH "${RegisterXPU_GENERATED}" RegisterXPU_GENERATED_NATIVE)
file(TO_NATIVE_PATH "${XPUFallback_TEMPLATE}" XPUFallback_TEMPLATE_NATIVE)
set(REGISTER_FALLBACK_CMD ${FILE_DISPLAY_CMD} ${XPUFallback_TEMPLATE_NATIVE} ">>" ${RegisterXPU_GENERATED_NATIVE})

function(GEN_XPU file_yaml)
  set(generated_files "")
  foreach(f ${ARGN})
    list(APPEND generated_files "${f}")
  endforeach()
  set(CODEGEN_XPU_YAML_DIR ${TORCH_XPU_OPS_ROOT}/yaml)

  # Codegen prepare process
  if(WIN32)
    file(TO_NATIVE_PATH "${CODEGEN_XPU_YAML_DIR}/templates" DestPATH)
    file(TO_NATIVE_PATH "${CMAKE_SOURCE_DIR}/aten/src/ATen/templates" SrcPATH)
    execute_process(COMMAND cmd /c xcopy ${SrcPATH} ${DestPATH} /E /H /C /I /Y > nul)
  else()
    execute_process(COMMAND ln -s ${CMAKE_SOURCE_DIR}/aten/src/ATen/templates ${CODEGEN_XPU_YAML_DIR}) # soft link to pytorch templates
  endif()

  set(XPU_CODEGEN_COMMAND
    "${PYTHON_EXECUTABLE}" -m torchgen.gen
    --source-path ${CODEGEN_XPU_YAML_DIR}
    --install-dir ${BUILD_TORCH_XPU_ATEN_GENERATED}
    --per-operator-headers
    --backend-whitelist XPU SparseXPU SparseCsrXPU NestedTensorXPU
    --xpu
  )

  add_custom_command(
    COMMENT "Generating XPU ATen Codegen..."
    OUTPUT ${generated_files}
    COMMAND
    ${XPU_CODEGEN_COMMAND}
    --static-dispatch-backend
    # --update-aoti-c-shim: generate extend/c_shim_xpu.h
    --update-aoti-c-shim
    # --exten-aoti-c-shim: specifiy the extend/c_shim_xpu
    # is out-of-tree extention for in-tree c_shim_xpu
    --extend-aoti-c-shim
    # --aoti-install-dir: generates c_shim_xpu.h and c_shim_xpu.cpp at
    # torch/csrc/inductor/aoti_torch/generated/extend/
    --aoti-install-dir=${XPU_AOTI_INSTALL_DIR}
    COMMAND
    ${REGISTER_FALLBACK_CMD}
    # Codegen post-process
    COMMAND "${PYTHON_EXECUTABLE}" ${TORCH_XPU_OPS_ROOT}/tools/codegen/remove_headers.py --register_xpu_path ${RegisterXPU_GENERATED}
    COMMAND "${PYTHON_EXECUTABLE}" ${TORCH_XPU_OPS_ROOT}/tools/codegen/remove_headers.py --register_xpu_path ${RegisterSparseXPU_GENERATED}
    COMMAND "${PYTHON_EXECUTABLE}" ${TORCH_XPU_OPS_ROOT}/tools/codegen/remove_headers.py --register_xpu_path ${RegisterSparseCsrXPU_GENERATED}
    COMMAND "${PYTHON_EXECUTABLE}" ${TORCH_XPU_OPS_ROOT}/tools/codegen/remove_headers.py --register_xpu_path ${RegisterNestedTensorXPU_GENERATED}
    WORKING_DIRECTORY ${TORCH_ROOT}
    DEPENDS
    ${CODEGEN_XPU_YAML_DIR}/native/${file_yaml}
    ${XPUFallback_TEMPLATE}
  )

  # Post codegen delete the copied templates folder only on Windows.
  if(WIN32)
    add_custom_target(DELETE_TEMPLATES ALL DEPENDS ${generated_files})
    add_custom_command(
      TARGET DELETE_TEMPLATES
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E remove_directory "${DestPATH}"
    )
  endif()
endfunction(GEN_XPU)

GEN_XPU(
  native_functions.yaml
  ${BUILD_TORCH_XPU_ATEN_GENERATED}/XPUFunctions.h
  ${BUILD_TORCH_XPU_ATEN_GENERATED}/XPUFunctions_inl.h
  ${RegisterXPU_GENERATED}
  ${RegisterSparseXPU_GENERATED}
  ${RegisterSparseCsrXPU_GENERATED}
  ${RegisterNestedTensorXPU_GENERATED}
  ${XPU_AOTI_SHIM_HEADER}
  ${XPU_AOTI_SHIM_SOURCE}
)

# The c_shim_xpu.cpp needs include files in ${CMAKE_BINARY_DIR}/xpu/ATen/ops/*.h)
# The include path is auto generated as "#include <ATen/ops/*.h">
# To follow the design of aoti codegen, here ${CMAKE_BINARY_DIR}/xpu is added to
# $TORCH_XPU_OPS_INCLUDE_DIRS, so that "#include <ATen/ops/*.h>" works.
list(APPEND TORCH_XPU_OPS_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/xpu)

list(APPEND xpu_generated_src
  ${RegisterXPU_GENERATED}
  ${RegisterSparseXPU_GENERATED}
  ${RegisterSparseCsrXPU_GENERATED}
  ${RegisterNestedTensorXPU_GENERATED}
  ${XPU_AOTI_SHIM_SOURCE}
)
set(ATen_XPU_GEN_SRCS ${xpu_generated_src})
