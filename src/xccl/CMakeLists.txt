# XCCL sources

file(GLOB xccl_h "*.hpp")
file(GLOB xccl_cpp "*.cpp")

list(APPEND ATen_XPU_XCCL_SRCS ${xccl_cpp})

set(ATen_XPU_XCCL_SRCS ${ATen_XPU_XCCL_SRCS} PARENT_SCOPE)

# Copy the header file to the build directory so that the PyTorch registration file can locate it.
foreach(HEADER  ${xccl_h})
  file(COPY ${HEADER} DESTINATION "${CMAKE_BINARY_DIR}/torch/csrc/distributed/c10d")
endforeach()
