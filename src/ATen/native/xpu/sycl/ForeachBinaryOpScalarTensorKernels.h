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

#define FOREACH_BINARY_TENSOR_ALPHA_INPLACE_KERNEL(NAME) \
  void FOREACH_BINARY_TENSOR_INPLACE_KERNEL_NAME(NAME)(  \
      TensorList tensors, const Tensor& scalar, const Scalar& alpha)

#define FOREACH_BINARY_TENSOR_ALPHA_KERNEL(NAME)               \
  std::vector<Tensor> FOREACH_BINARY_TENSOR_KERNEL_NAME(NAME)( \
      TensorList tensors, const Tensor& scalar, const Scalar& alpha)

TORCH_XPU_API FOREACH_BINARY_TENSOR_INPLACE_KERNEL(mul);
TORCH_XPU_API FOREACH_BINARY_TENSOR_KERNEL(mul);
TORCH_XPU_API FOREACH_BINARY_TENSOR_INPLACE_KERNEL(div);
TORCH_XPU_API FOREACH_BINARY_TENSOR_KERNEL(div);

TORCH_XPU_API FOREACH_BINARY_TENSOR_ALPHA_INPLACE_KERNEL(add);
TORCH_XPU_API FOREACH_BINARY_TENSOR_ALPHA_KERNEL(add);

} // namespace at::native::xpu
