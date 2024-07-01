if(Codegen_GPU_cmake_included)
  return()
endif()
set(Codegen_GPU_cmake_included true)

set(BUILD_TORCH_XPU_ATEN_GENERATED "${CMAKE_BINARY_DIR}/aten/src/ATen/xpu")
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

GEN_BACKEND(
  xpu_functions.yaml
  XPUNativeFunctions.h
  RegisterXPU.cpp)


list(APPEND xpu_generated_src ${RegisterXPU_PATH})
add_custom_target(TORCH_XPU_GEN_TARGET DEPENDS ${xpu_generated_src})
set(ATen_XPU_GEN_SRCS ${xpu_generated_src})
