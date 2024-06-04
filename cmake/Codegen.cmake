if(Codegen_GPU_cmake_included)
  return()
endif()
set(Codegen_GPU_cmake_included true)

set(BUILD_TORCH_XPU_ATEN_GENERATED "${CMAKE_BINARY_DIR}/aten/src/ATen/xpu")
file(MAKE_DIRECTORY ${BUILD_TORCH_XPU_ATEN_GENERATED})

set(RegisterXPU_PATH ${BUILD_TORCH_XPU_ATEN_GENERATED}/RegisterXPU.cpp)
set(XPUFallback_PATH ${TORCH_XPU_OPS_ROOT}/src/aten/XPUFallback.template)
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
    cat ${XPUFallback_PATH} >> ${RegisterXPU_PATH}
    ${SIMPLE_TRACE}
    WORKING_DIRECTORY ${TORCH_ROOT}
    DEPENDS
    ${depended_files}
    ${TORCH_XPU_OPS_ROOT}/yaml/${file_yaml}
    ${XPUFallback_PATH}
    )
endfunction(GEN_BACKEND)

set(RegisterXPU_PATH ${BUILD_TORCH_XPU_ATEN_GENERATED}/RegisterXPU.cpp)
set(XPUFallback_PATH ${TORCH_XPU_OPS_ROOT}/src/aten/XPUFallback.template)
function(GEN_XPU file_yaml)
  set(generated_files "")
  foreach(f ${ARGN})
    list(APPEND generated_files "${BUILD_TORCH_XPU_ATEN_GENERATED}/${f}")
  endforeach()
  file(GLOB_RECURSE depend_files ${TORCH_XPU_OPS_ROOT}/yaml/${file_yaml})
  set(CODEGEN_TEMPLATE ${TORCH_XPU_OPS_ROOT}/yaml/)
  execute_process(COMMAND ln -s ${CMAKE_SOURCE_DIR}/aten/src/ATen/templates ${CODEGEN_TEMPLATE})
  add_custom_command(
    OUTPUT ${generated_files}
    COMMAND
    "${PYTHON_EXECUTABLE}" -m torchgen.gen
    --source-path ${TORCH_XPU_OPS_ROOT}/yaml/
    --install-dir ${BUILD_TORCH_XPU_ATEN_GENERATED}
    --per-operator-headers
    --static-dispatch-backend
    --backend-whitelist=XPU
    --only_backend
    COMMAND
    cat ${XPUFallback_PATH} >> ${RegisterXPU_PATH}
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
