# XCCL sources

file(GLOB xccl_cpp "*.cpp")

list(APPEND ATen_XPU_XCCL_SRCS ${xccl_cpp})

set(ATen_XPU_XCCL_SRCS ${ATen_XPU_XCCL_SRCS} PARENT_SCOPE)
