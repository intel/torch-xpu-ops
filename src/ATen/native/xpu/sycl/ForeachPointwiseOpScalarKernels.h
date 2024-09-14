#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

#define FOREACH_POINTWISE_OP_SCALAR_KERNEL(NAME) \
  std::vector<Tensor> foreach_##NAME##_kernel(   \
      TensorList input,                          \
      TensorList tensors1,                       \
      TensorList tensors2,                       \
      const Scalar& scalar)

#define FOREACH_POINTWISE_OP_SCALAR_INPLACE_KERNEL(NAME) \
  void foreach_##NAME##_kernel_(                         \
      TensorList input,                                  \
      TensorList tensors1,                               \
      TensorList tensors2,                               \
      const Scalar& scalar)

TORCH_XPU_API FOREACH_POINTWISE_OP_SCALAR_KERNEL(addcmul);
TORCH_XPU_API FOREACH_POINTWISE_OP_SCALAR_INPLACE_KERNEL(addcmul);
TORCH_XPU_API FOREACH_POINTWISE_OP_SCALAR_KERNEL(addcdiv);
TORCH_XPU_API FOREACH_POINTWISE_OP_SCALAR_INPLACE_KERNEL(addcdiv);

} // namespace at::native::xpu
