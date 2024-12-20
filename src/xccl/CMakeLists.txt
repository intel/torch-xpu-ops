# XCCL sources

file(GLOB xccl_h "*.hpp")
file(GLOB xccl_cpp "*.cpp")

list(APPEND ATen_XPU_XCCL_SRCS ${xccl_cpp})

set(ATen_XPU_XCCL_SRCS ${ATen_XPU_XCCL_SRCS} PARENT_SCOPE)

# Why copy the header file to the build directory?
# We want register XCCL backend to PyTorch c10d in torch/csrc/distributed/c10d/init.cpp#L27-L29.
# To align with other backends, we need to copy the header file to the build torch/csrc/distributed/c10d directory.
# Further solution is add find path for torch/csrc/distributed/c10d/init.cpp#L27-L29.
foreach(HEADER  ${xccl_h})
  file(COPY ${HEADER} DESTINATION "${CMAKE_BINARY_DIR}/torch/csrc/distributed/c10d")
endforeach()
