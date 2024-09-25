if(Codegen_GPU_cmake_included)
  return()
endif()
set(Codegen_GPU_cmake_included true)

set(BUILD_TORCH_XPU_ATEN_GENERATED "${CMAKE_BINARY_DIR}/xpu/ATen/")
file(MAKE_DIRECTORY ${BUILD_TORCH_XPU_ATEN_GENERATED})

set(RegisterXPU_PATH ${BUILD_TORCH_XPU_ATEN_GENERATED}/RegisterXPU.cpp)
set(XPUFallback_PATH ${TORCH_XPU_OPS_ROOT}/src/ATen/native/xpu/XPUFallback.template)

if(WIN32)
  set(FILE_DISPLAY_CMD type)
  # replace forward slash with back slash for compatibility with 'type' command on Windows
  string(REPLACE "/" "\\" RegisterXPU_PATH_BACKSLASH "${RegisterXPU_PATH}")
  string(REPLACE "/" "\\" XPUFallback_PATH_BACKSLASH "${XPUFallback_PATH}")
  set(REGISTER_FALLBACK_CMD ${FILE_DISPLAY_CMD} ${XPUFallback_PATH_BACKSLASH} ">>" ${RegisterXPU_PATH_BACKSLASH})
else()
  set(FILE_DISPLAY_CMD cat)
  set(REGISTER_FALLBACK_CMD ${FILE_DISPLAY_CMD} ${XPUFallback_PATH} ">>" ${RegisterXPU_PATH})
endif()

function(GEN_BACKEND file_yaml)
  set(generated_files "")
  foreach(f ${ARGN})
    list(APPEND generated_files "${BUILD_TORCH_XPU_ATEN_GENERATED}/${f}")
  endforeach()
  file(GLOB_RECURSE depended_files ${TORCH_XPU_OPS_ROOT}/yaml/${file_yaml})
  add_custom_command(
    OUTPUT ${generated_files}
    COMMAND
    "${PYTHON_EXECUTABLE}" -m torchgen.gen_backend_stubs
    --output_dir ${BUILD_TORCH_XPU_ATEN_GENERATED}
    --source_yaml ${TORCH_XPU_OPS_ROOT}/yaml/${file_yaml}
    COMMAND
    ${REGISTER_FALLBACK_CMD}
    ${SIMPLE_TRACE}
    WORKING_DIRECTORY ${TORCH_ROOT}
    DEPENDS
    ${depended_files}
    ${TORCH_XPU_OPS_ROOT}/yaml/${file_yaml}
    ${XPUFallback_PATH}
    )
endfunction(GEN_BACKEND)


set(RegisterXPU_PATH ${BUILD_TORCH_XPU_ATEN_GENERATED}/RegisterXPU.cpp)
set(XPUFallback_PATH ${TORCH_XPU_OPS_ROOT}/src/ATen/native/xpu/XPUFallback.template)
function(GEN_XPU file_yaml)
  set(generated_files "")
  foreach(f ${ARGN})
    list(APPEND generated_files "${BUILD_TORCH_XPU_ATEN_GENERATED}/${f}")
  endforeach()
  file(GLOB_RECURSE depend_files ${TORCH_XPU_OPS_ROOT}/yaml/${file_yaml})
  set(CODEGEN_TEMPLATE ${TORCH_XPU_OPS_ROOT}/yaml/)

  # Codegen prepare process
  if(WIN32)
    execute_process(COMMAND mklink /d ${CODEGEN_TEMPLATE}/templates ${CMAKE_SOURCE_DIR}/aten/src/ATen/templates)
    string(REPLACE "/" "\\" RegisterXPU_PATH_BACKSLASH "${RegisterXPU_PATH}")
    string(REPLACE "/" "\\" XPUFallback_PATH_BACKSLASH "${XPUFallback_PATH}")
    set(REGISTER_FALLBACK_CMD ${FILE_DISPLAY_CMD} ${XPUFallback_PATH_BACKSLASH} ">>" ${RegisterXPU_PATH_BACKSLASH})
  else()
    execute_process(COMMAND ln -s ${CMAKE_SOURCE_DIR}/aten/src/ATen/templates ${CODEGEN_TEMPLATE}) # soft link to pytorch templates
    set(REGISTER_FALLBACK_CMD ${FILE_DISPLAY_CMD} ${XPUFallback_PATH} ">>" ${RegisterXPU_PATH})
  endif()

  add_custom_command(
    OUTPUT ${generated_files}
    COMMAND
    "${PYTHON_EXECUTABLE}" -m torchgen.gen
    --source-path ${TORCH_XPU_OPS_ROOT}/yaml/
    --install-dir ${BUILD_TORCH_XPU_ATEN_GENERATED}
    --per-operator-headers
    --static-dispatch-backend
    --backend-whitelist=XPU
    COMMAND
    ${REGISTER_FALLBACK_CMD}
    # Codegen post-process
    COMMAND "${PYTHON_EXECUTABLE}" ${TORCH_XPU_OPS_ROOT}/tools/codegen/remove_headers.py --register_xpu_path ${RegisterXPU_PATH}
    ${SIMPLE_TRACE} 
    WORKING_DIRECTORY ${TORCH_ROOT}
    DEPENDS
  ${depended_files}
    ${TORCH_XPU_OPS_ROOT}/yaml/native/${file_yaml}
    ${XPUFallback_PATH}
  )
endfunction(GEN_XPU)

# GEN_BACKEND(
#   xpu_functions.yaml
#   XPUNativeFunctions.h
#   RegisterXPU.cpp)

GEN_XPU(
  native_functions.yaml
  XPUFunctions.h
  RegisterXPU.cpp
)




list(APPEND xpu_generated_src ${RegisterXPU_PATH})
add_custom_target(TORCH_XPU_GEN_TARGET DEPENDS ${xpu_generated_src})
set(ATen_XPU_GEN_SRCS ${xpu_generated_src})
