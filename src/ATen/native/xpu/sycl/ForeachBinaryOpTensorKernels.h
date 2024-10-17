#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

#define FOREACH_BINARY_TENSOR_INPLACE_KERNEL_NAME(NAME) \
  foreach_binary_##NAME##_tensor__kernel

#define FOREACH_BINARY_TENSOR_KERNEL_NAME(NAME) \
  foreach_binary_##NAME##_tensor_kernel

#define FOREACH_BINARY_TENSOR_INPLACE_KERNEL(NAME)      \
  void FOREACH_BINARY_TENSOR_INPLACE_KERNEL_NAME(NAME)( \
      TensorList tensors, const Tensor& scalar)

#define FOREACH_BINARY_TENSOR_KERNEL(NAME)                     \
  std::vector<Tensor> FOREACH_BINARY_TENSOR_KERNEL_NAME(NAME)( \
      TensorList tensors, const Tensor& scalar)

TORCH_XPU_API FOREACH_BINARY_TENSOR_INPLACE_KERNEL(mul);
TORCH_XPU_API FOREACH_BINARY_TENSOR_KERNEL(mul);

} // namespace at::native::xpu
