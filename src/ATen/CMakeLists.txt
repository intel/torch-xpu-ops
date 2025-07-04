# ATen XPU sources

file(GLOB xpu_h "xpu/*.h")
file(GLOB xpu_cpp "xpu/*.cpp")
file(GLOB xpu_mkl "native/xpu/mkl/*.cpp")
file(GLOB xpu_native_cpp "native/xpu/*.cpp" "native/sparse/*.cpp" "native/sparse/xpu/*.cpp" "native/nested/*.cpp" "native/nested/xpu/*.cpp" "native/transformers/*.cpp" "native/quantized/*.cpp")
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
