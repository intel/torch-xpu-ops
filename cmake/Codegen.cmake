if(Codegen_GPU_cmake_included)
  return()
endif()
set(Codegen_GPU_cmake_included true)

set(BUILD_TORCH_XPU_ATEN_GENERATED "${CMAKE_BINARY_DIR}/xpu/ATen/")
file(MAKE_DIRECTORY ${BUILD_TORCH_XPU_ATEN_GENERATED})

set(RegisterXPU_PATH ${BUILD_TORCH_XPU_ATEN_GENERATED}/RegisterXPU.cpp)
set(RegisterSparseXPU_PATH ${BUILD_TORCH_XPU_ATEN_GENERATED}/RegisterSparseXPU.cpp)
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
set(RegisterSparseXPU_PATH ${BUILD_TORCH_XPU_ATEN_GENERATED}/RegisterSparseXPU.cpp)
set(XPUFallback_PATH ${TORCH_XPU_OPS_ROOT}/src/ATen/native/xpu/XPUFallback.template)
set(XPU_AOTI_INSTALL_DIR ${TORCH_ROOT}/torch/csrc/inductor/aoti_torch/generated/extend)
function(GEN_XPU file_yaml)
  set(generated_files "")
  foreach(f ${ARGN})
    list(APPEND generated_files "${f}")
  endforeach()
  file(GLOB_RECURSE depend_files ${TORCH_XPU_OPS_ROOT}/yaml/${file_yaml})
  set(CODEGEN_TEMPLATE ${TORCH_XPU_OPS_ROOT}/yaml/)

  # Codegen prepare process
  if(WIN32)
    string(REPLACE "/" "\\" DestPATH "${CODEGEN_TEMPLATE}templates")
    string(REPLACE "/" "\\" SrcPATH "${CMAKE_SOURCE_DIR}/aten/src/ATen/templates")
    execute_process(COMMAND cmd /c xcopy ${SrcPATH} ${DestPATH} /E /H /C /I /Y > nul)
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
    --backend-whitelist XPU SparseXPU
    # --xpu: generate in-tree RegisterXPU.cpp for in-tree OPs
    --xpu
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
    COMMAND "${PYTHON_EXECUTABLE}" ${TORCH_XPU_OPS_ROOT}/tools/codegen/remove_headers.py --register_xpu_path ${RegisterXPU_PATH}
    COMMAND "${PYTHON_EXECUTABLE}" ${TORCH_XPU_OPS_ROOT}/tools/codegen/remove_headers.py --register_xpu_path ${RegisterSparseXPU_PATH}
    ${SIMPLE_TRACE} 
    WORKING_DIRECTORY ${TORCH_ROOT}
    DEPENDS
    ${depended_files}
    ${TORCH_XPU_OPS_ROOT}/yaml/native/${file_yaml}
    ${XPUFallback_PATH}
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

# GEN_BACKEND(
#   xpu_functions.yaml
#   XPUNativeFunctions.h
#   RegisterXPU.cpp)

GEN_XPU(
  native_functions.yaml
  ${BUILD_TORCH_XPU_ATEN_GENERATED}/XPUFunctions.h
  ${BUILD_TORCH_XPU_ATEN_GENERATED}/RegisterXPU.cpp
  ${BUILD_TORCH_XPU_ATEN_GENERATED}/RegisterSparseXPU.cpp
  ${XPU_AOTI_INSTALL_DIR}/c_shim_xpu.h
  ${XPU_AOTI_INSTALL_DIR}/c_shim_xpu.cpp
)


# The c_shim_xpu.cpp needs include files in ${CMAKE_BINARY_DIR}/xpu/ATen/ops/*.h)
# The include path is auto generated as "#include <ATen/ops/*.h">
# To follow the design of aoti codegen, here ${CMAKE_BINARY_DIR}/xpu is added to
# $TORCH_XPU_OPS_INCLUDE_DIRS, so that "#include <ATen/ops/*.h>" works.
list(APPEND TORCH_XPU_OPS_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/xpu)

list(APPEND xpu_generated_src ${RegisterXPU_PATH} ${RegisterSparseXPU_PATH})
list(APPEND xpu_generated_src ${XPU_AOTI_INSTALL_DIR}/c_shim_xpu.cpp)
add_custom_target(TORCH_XPU_GEN_TARGET DEPENDS ${xpu_generated_src})
set(ATen_XPU_GEN_SRCS ${xpu_generated_src})
