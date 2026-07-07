# oneCCL LSA SymmetricMemory sources.
# Only compiled when USE_ONECCL is enabled (see top-level CMakeLists.txt).

file(GLOB oneccl_cpp "*.cpp")

list(APPEND ATen_XPU_XCCL_SRCS ${oneccl_cpp})

set(ATen_XPU_XCCL_SRCS ${ATen_XPU_XCCL_SRCS} PARENT_SCOPE)
