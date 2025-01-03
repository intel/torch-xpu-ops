# Define archive static library target - torch_xpu_ops
include(${TORCH_XPU_OPS_ROOT}/cmake/Codegen.cmake)

set(ATen_XPU_CPP_SRCS)
set(ATen_XPU_NATIVE_CPP_SRCS)
set(ATen_XPU_SYCL_SRCS)
set(ATen_XPU_XCCL_SRCS)

set(ATen_XPU_INCLUDE_DIRS ${TORCH_XPU_OPS_ROOT}/src CACHE STRING "ATen XPU Include directory")

add_subdirectory(ATen)
if(USE_C10D_XCCL)
  add_subdirectory(xccl)
endif()
# With the increasement of bin size, we have to split libtorch_xpu.so into
# multiple libraries. Because of strict linkage requirements on Windows,
# we add extra logics to resolve, 1) Cyclic dependence, 2) Make symbols visible.
# To isolate impact and keep both maintainable, there are separate code pieces
# to setup CMAKE targets. Will revert splitting library and unify these once
# compression of device code is supported by SYCL/backend compiler.
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
