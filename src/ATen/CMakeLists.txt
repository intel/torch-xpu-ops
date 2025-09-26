# ATen XPU sources

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

# ATen XPU headers

macro(install_xpu_headers glob_pattern dest_subdir)
  file(GLOB headers ${glob_pattern})
  if(headers)
    install(FILES ${headers} DESTINATION "${AT_INSTALL_INCLUDE_DIR}/${dest_subdir}")
  endif()
endmacro()

install_xpu_headers("xpu/*.h" "ATen/xpu")
install_xpu_headers("native/xpu/*.h" "ATen/native/xpu")
install_xpu_headers("native/xpu/sycl/*.h" "ATen/native/xpu/sycl")
install_xpu_headers("native/xpu/mkl/*.h" "ATen/native/xpu/mkl")
install_xpu_headers("native/nested/xpu/*.h" "ATen/native/nested/xpu")
install_xpu_headers("native/nested/xpu/sycl/*.h" "ATen/native/nested/xpu/sycl")
install_xpu_headers("native/quantized/*.h" "ATen/native/quantized/xpu")
install_xpu_headers("native/quantized/sycl/*.h" "ATen/native/quantized/xpu/sycl")
install_xpu_headers("native/sparse/xpu/*.h" "ATen/native/sparse/xpu")
install_xpu_headers("native/sparse/xpu/sycl/*.h" "ATen/native/sparse/xpu/sycl")
install_xpu_headers("native/transformers/*.h" "ATen/native/transformers/xpu")
install_xpu_headers("native/transformers/sycl/*.h" "ATen/native/transformers/xpu/sycl")

if(xpu_ops_generated_headers)
  install(FILES ${xpu_ops_generated_headers} DESTINATION ${AT_INSTALL_INCLUDE_DIR}/ATen/ops)
endif()
