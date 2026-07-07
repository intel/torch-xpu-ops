# ISHMEM SymmetricMemory sources
# Only compiled when USE_ISHMEM is enabled

file(GLOB ishmem_cpp "*.cpp")

list(APPEND ATen_XPU_XCCL_SRCS ${ishmem_cpp})

set(ATen_XPU_XCCL_SRCS ${ATen_XPU_XCCL_SRCS} PARENT_SCOPE)
