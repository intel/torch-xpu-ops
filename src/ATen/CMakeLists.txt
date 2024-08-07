# ATen XPU sources

file(GLOB xpu_cpp "xpu/*.cpp" "native/xpu/*.cpp" "native/sparse/*.cpp")
file(GLOB xpu_sycl "native/xpu/sycl/*.cpp")

list(APPEND ATen_XPU_CPP_SRCS ${xpu_cpp})
list(APPEND ATen_XPU_SYCL_SRCS ${xpu_sycl})

set(ATen_XPU_CPP_SRCS ${ATen_XPU_CPP_SRCS} PARENT_SCOPE)
set(ATen_XPU_SYCL_SRCS ${ATen_XPU_SYCL_SRCS} PARENT_SCOPE)
