# Define archive static library target - torch_xpu_ops
include(${TORCH_XPU_OPS_ROOT}/cmake/Codegen.cmake)

set(ATen_XPU_CPP_SRCS)
set(ATen_XPU_SYCL_SRCS)

set(ATen_XPU_INCLUDE_DIRS ${TORCH_XPU_OPS_ROOT}/src)

add_subdirectory(aten)

set(SYCL_LINK_LIBRARIES_KEYWORD PRIVATE)
sycl_add_library(
  torch_xpu_ops
  STATIC
  SYCL_SOURCES ${ATen_XPU_SYCL_SRCS}
  CXX_SOURCES ${ATen_XPU_CPP_SRCS} ${ATen_XPU_GEN_SRCS})
set(SYCL_LINK_LIBRARIES_KEYWORD)

# Align with PyTorch compile options
# 1. submodule - PYTORCH_SRC_DIR/cmake/public/utils.cmake
# 2. standalone - PYTORCH_INSTALL_DIR/share/cmake/Caffe2/public/utils.cmake
torch_compile_options(torch_xpu_ops)
target_compile_options_if_supported(torch_xpu_ops "-Wno-deprecated-copy")
target_compile_options(torch_xpu_ops PRIVATE ${TORCH_XPU_OPS_FLAGS})

target_include_directories(torch_xpu_ops PUBLIC ${TORCH_XPU_OPS_INCLUDE_DIRS})
target_include_directories(torch_xpu_ops PUBLIC ${ATen_XPU_INCLUDE_DIRS})
target_include_directories(torch_xpu_ops PUBLIC ${SYCL_INCLUDE_DIR})

target_link_libraries(torch_xpu_ops PUBLIC ${SYCL_LIBRARIES})
