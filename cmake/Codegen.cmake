if(Codegen_XPU_cmake_included)
  return()
endif()
set(Codegen_XPU_cmake_included true)

set(BUILD_TORCH_XPU_ATEN_GENERATED "${CMAKE_BINARY_DIR}/xpu/ATen")
set(BUILD_TORCH_ATEN_GENERATED "${CMAKE_BINARY_DIR}/aten/src/ATen")
file(MAKE_DIRECTORY ${BUILD_TORCH_XPU_ATEN_GENERATED})

set(RegisterXPU_GENERATED ${BUILD_TORCH_XPU_ATEN_GENERATED}/RegisterXPU_0.cpp)
set(RegisterSparseXPU_GENERATED ${BUILD_TORCH_XPU_ATEN_GENERATED}/RegisterSparseXPU_0.cpp)
set(RegisterSparseCsrXPU_GENERATED ${BUILD_TORCH_XPU_ATEN_GENERATED}/RegisterSparseCsrXPU_0.cpp)
set(RegisterNestedTensorXPU_GENERATED ${BUILD_TORCH_XPU_ATEN_GENERATED}/RegisterNestedTensorXPU_0.cpp)
set(XPUFallback_TEMPLATE ${TORCH_XPU_OPS_ROOT}/src/ATen/native/xpu/XPUFallback.template)
set(XPU_AOTI_INSTALL_DIR ${TORCH_ROOT}/torch/csrc/inductor/aoti_torch/generated/extend)
set(XPU_AOTI_SHIM_HEADER ${XPU_AOTI_INSTALL_DIR}/c_shim_xpu.h)
set(XPU_AOTI_SHIM_SOURCE ${XPU_AOTI_INSTALL_DIR}/c_shim_xpu.cpp)
set(CODEGEN_XPU_YAML_DIR ${TORCH_XPU_OPS_ROOT}/yaml)

# Codegen prepare process
if(WIN32)
  file(TO_NATIVE_PATH "${CODEGEN_XPU_YAML_DIR}/templates" DestPATH)
  file(TO_NATIVE_PATH "${CMAKE_SOURCE_DIR}/aten/src/ATen/templates" SrcPATH)
  # Copy pytorch templates
  execute_process(COMMAND cmd /c xcopy ${SrcPATH} ${DestPATH} /E /H /C /I /Y > nul)
else()
  # soft link to pytorch templates
  execute_process(COMMAND ln -sf ${CMAKE_SOURCE_DIR}/aten/src/ATen/templates ${CODEGEN_XPU_YAML_DIR})
endif()

set(XPU_CODEGEN_COMMAND
  "${Python_EXECUTABLE}" -m torchgen.gen
  --source-path ${CODEGEN_XPU_YAML_DIR}
  --install-dir ${BUILD_TORCH_XPU_ATEN_GENERATED}
  --per-operator-headers
  --backend-whitelist XPU SparseXPU SparseCsrXPU NestedTensorXPU
  --xpu
)

set(XPU_INSTALL_HEADER_COMMAND
  "${Python_EXECUTABLE}" ${TORCH_XPU_OPS_ROOT}/tools/codegen/install_xpu_headers.py
  --src-header-dir ${BUILD_TORCH_XPU_ATEN_GENERATED}
  --dst-header-dir ${BUILD_TORCH_ATEN_GENERATED}
)

# Generate ops_generated_headers.cmake for torch-xpu-ops
execute_process(
  COMMAND
    ${XPU_CODEGEN_COMMAND}
    --generate headers
    --dry-run
    --output-dependencies ${BUILD_TORCH_XPU_ATEN_GENERATED}/generated_headers.cmake
  RESULT_VARIABLE RETURN_VALUE
  WORKING_DIRECTORY ${TORCH_ROOT}
)

if(NOT RETURN_VALUE EQUAL 0)
  message(FATAL_ERROR "Failed to generate ops_generated_headers.cmake for torch-xpu-ops.")
endif()

# Generate xpu_ops_generated_headers.cmake
execute_process(
  COMMAND
    ${XPU_INSTALL_HEADER_COMMAND}
    --dry-run
  RESULT_VARIABLE RETURN_VALUE
  WORKING_DIRECTORY ${TORCH_ROOT}
)

if(NOT RETURN_VALUE EQUAL 0)
  message(FATAL_ERROR "Failed to generate xpu_ops_generated_headers.cmake.")
endif()

include(${BUILD_TORCH_XPU_ATEN_GENERATED}/xpu_ops_generated_headers.cmake)
include(${BUILD_TORCH_XPU_ATEN_GENERATED}/ops_generated_headers.cmake)

if(WIN32)
  set(FILE_DISPLAY_CMD type)
else()
  set(FILE_DISPLAY_CMD cat)
endif()
file(TO_NATIVE_PATH "${RegisterXPU_GENERATED}" RegisterXPU_GENERATED_NATIVE)
file(TO_NATIVE_PATH "${XPUFallback_TEMPLATE}" XPUFallback_TEMPLATE_NATIVE)
set(REGISTER_FALLBACK_CMD ${FILE_DISPLAY_CMD} ${XPUFallback_TEMPLATE_NATIVE} ">>" ${RegisterXPU_GENERATED_NATIVE})

set(OUTPUT_LIST
  ${BUILD_TORCH_XPU_ATEN_GENERATED}/XPUFunctions.h
  ${BUILD_TORCH_XPU_ATEN_GENERATED}/XPUFunctions_inl.h
  ${RegisterXPU_GENERATED}
  ${RegisterSparseXPU_GENERATED}
  ${RegisterSparseCsrXPU_GENERATED}
  ${RegisterNestedTensorXPU_GENERATED}
  ${XPU_AOTI_SHIM_HEADER}
  ${XPU_AOTI_SHIM_SOURCE}
  ${ops_generated_headers}
)

# Generate torch-xpu-ops codegen
add_custom_command(
  COMMENT "Generating XPU ATen Codegen..."
  OUTPUT ${OUTPUT_LIST}
  COMMAND
    ${XPU_CODEGEN_COMMAND}
    --static-dispatch-backend
    --update-aoti-c-shim
    --extend-aoti-c-shim
    --aoti-install-dir=${XPU_AOTI_INSTALL_DIR}
  COMMAND
    ${REGISTER_FALLBACK_CMD}
  # Codegen post process
  COMMAND
    ${XPU_INSTALL_HEADER_COMMAND}
  DEPENDS
    torch_cpu
    ATEN_CPU_FILES_GEN_TARGET
    ATEN_XPU_FILES_GEN_TARGET
    ${XPUFallback_TEMPLATE}
    ${TORCH_XPU_OPS_ROOT}/tools/codegen/install_xpu_headers.py
    ${BUILD_TORCH_XPU_ATEN_GENERATED}/xpu_ops_generated_headers.cmake
    ${CODEGEN_XPU_YAML_DIR}/native/native_functions.yaml
    ${all_python} ${headers_templates}
    ${TORCH_ROOT}/aten/src/ATen/native/native_functions.yaml
    ${TORCH_ROOT}/aten/src/ATen/native/tags.yaml
  WORKING_DIRECTORY ${TORCH_ROOT}
)

# Codegen post progress
if(WIN32)
  add_custom_target(DELETE_TEMPLATES ALL DEPENDS ${OUTPUT_LIST})
  # Delete the copied templates folder only on Windows.
  add_custom_command(
    TARGET DELETE_TEMPLATES
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${DestPATH}"
  )
endif()

# Ensure that all generated ATen XPU op files are built before compiling the
# torch-xpu-ops library.
add_custom_target(ATEN_XPU_OPS_FILES_GEN_TARGET DEPENDS
  ${ops_generated_headers} ${OUTPUT_LIST})
add_library(ATEN_XPU_OPS_FILES_GEN_LIB INTERFACE)
add_dependencies(ATEN_XPU_OPS_FILES_GEN_LIB ATEN_XPU_OPS_FILES_GEN_TARGET)
if(USE_PER_OPERATOR_HEADERS)
  target_compile_definitions(ATEN_XPU_OPS_FILES_GEN_LIB INTERFACE AT_PER_OPERATOR_HEADERS)
endif()

set(ATen_XPU_GEN_SRCS
  ${RegisterXPU_GENERATED}
  ${RegisterSparseXPU_GENERATED}
  ${RegisterSparseCsrXPU_GENERATED}
  ${RegisterNestedTensorXPU_GENERATED}
  ${XPU_AOTI_SHIM_SOURCE}
)
