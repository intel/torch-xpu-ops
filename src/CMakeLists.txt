# Define archive static library target - torch_xpu_ops
include(${TORCH_XPU_OPS_ROOT}/cmake/Codegen.cmake)

set(ATen_XPU_CPP_SRCS)
set(ATen_XPU_NATIVE_CPP_SRCS)
set(ATen_XPU_SYCL_SRCS)

set(ATen_XPU_INCLUDE_DIRS ${TORCH_XPU_OPS_ROOT}/src CACHE STRING "ATen XPU Include directory")

add_subdirectory(ATen)

if(WIN32)
  include(BuildOnWindows.cmake)
else()
  include(BuildOnLinux.cmake)
endif()

include(${TORCH_XPU_OPS_ROOT}/cmake/ClangFormat.cmake)
if(CLANG_FORMAT)
  file(GLOB_RECURSE ALL_CSRCS ${TORCH_XPU_OPS_ROOT}/**.[ch] ${TORCH_XPU_OPS_ROOT}/**.[ch]pp)
  add_custom_target(CL_FORMAT_CSRCS COMMAND ${CLANG_FORMAT_EXEC} -i -style=file ${ALL_CSRCS})
  add_dependencies(torch_xpu_ops CL_FORMAT_CSRCS)
endif()
