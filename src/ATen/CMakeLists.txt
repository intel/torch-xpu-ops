# ATen XPU sources

file(GLOB xpu_h "xpu/*.h")
file(GLOB xpu_cpp "xpu/*.cpp")
file(GLOB xpu_mkl "native/xpu/mkl/*.cpp")
file(GLOB xpu_native_cpp "native/xpu/*.cpp" "native/sparse/*.cpp" "native/sparse/xpu/*.cpp" "native/nested/*.cpp" "native/nested/xpu/*.cpp" "native/transformers/*.cpp" "native/quantized/*.cpp" "native/cutlass/*.cpp")
file(GLOB xpu_sycl "native/xpu/sycl/*.cpp" "native/sparse/xpu/sycl/*.cpp" "native/nested/xpu/sycl/*.cpp" "native/transformers/sycl/*.cpp" "native/quantized/sycl/*.cpp")

list(APPEND ATen_XPU_CPP_SRCS ${xpu_cpp})
if(USE_ONEMKL_XPU)
  list(APPEND ATen_XPU_MKL_SRCS ${xpu_mkl})
endif()
list(APPEND ATen_XPU_NATIVE_CPP_SRCS ${xpu_native_cpp})
list(APPEND ATen_XPU_SYCL_SRCS ${xpu_sycl})

set(ATen_XPU_CPP_SRCS ${ATen_XPU_CPP_SRCS} PARENT_SCOPE)
set(ATen_XPU_MKL_SRCS ${ATen_XPU_MKL_SRCS} PARENT_SCOPE)
set(ATen_XPU_NATIVE_CPP_SRCS ${ATen_XPU_NATIVE_CPP_SRCS} PARENT_SCOPE)
set(ATen_XPU_SYCL_SRCS ${ATen_XPU_SYCL_SRCS} PARENT_SCOPE)

foreach(HEADER  ${xpu_h})
  install(FILES ${HEADER} DESTINATION "${AT_INSTALL_INCLUDE_DIR}/ATen/xpu")
endforeach()

foreach(HEADER  ${xpu_ops_generated_headers})
  install(FILES ${HEADER} DESTINATION ${AT_INSTALL_INCLUDE_DIR}/ATen/ops)
endforeach()

include(ExternalProject)
ExternalProject_Add(
  cutlass_sycl_kernels_proj
  SOURCE_DIR ${TORCH_XPU_OPS_ROOT}/src/ATen/native/cutlass/sycl
  CMAKE_ARGS
    -DCMAKE_C_COMPILER=icx
    -DCMAKE_CXX_COMPILER=icpx
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DATen_XPU_INCLUDE_STR=${ATen_XPU_INCLUDE_STR}
    -DTORCH_XPU_OPS_INCLUDE_STR=${TORCH_XPU_OPS_INCLUDE_STR}
  BUILD_ALWAYS TRUE
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS "cutlass_sycl_kernels_proj-prefix/src/cutlass_sycl_kernels_proj-build/libcutlass_kernels.so"
)

ExternalProject_Get_Property(cutlass_sycl_kernels_proj SOURCE_DIR BINARY_DIR)
set(CUTLASS_SYCL_KERNELS_LIBRARIES ${BINARY_DIR}/libcutlass_kernels.so)

add_library(cutlass_sycl_kernels INTERFACE)
add_dependencies(cutlass_sycl_kernels cutlass_sycl_kernels_proj)
target_link_libraries(cutlass_sycl_kernels INTERFACE ${CUTLASS_SYCL_KERNELS_LIBRARIES})
install(FILES ${CUTLASS_SYCL_KERNELS_LIBRARIES} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
