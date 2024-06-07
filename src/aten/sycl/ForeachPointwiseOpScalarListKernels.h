#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

#define FOREACH_POINTWISE_OP_SCALARLIST_KERNEL(NAME) \
  std::vector<Tensor> foreach_##NAME##_kernel(       \
      TensorList input,                              \
      TensorList tensors1,                           \
      TensorList tensors2,                           \
      at::ArrayRef<Scalar> scalars)

#define FOREACH_POINTWISE_OP_SCALARLIST_INPLACE_KERNEL(NAME) \
  void foreach_##NAME##_kernel_(                             \
      TensorList input,                                      \
      TensorList tensors1,                                   \
      TensorList tensors2,                                   \
      at::ArrayRef<Scalar> scalars)

FOREACH_POINTWISE_OP_SCALARLIST_KERNEL(addcmul);
FOREACH_POINTWISE_OP_SCALARLIST_INPLACE_KERNEL(addcmul);
FOREACH_POINTWISE_OP_SCALARLIST_KERNEL(addcdiv);
FOREACH_POINTWISE_OP_SCALARLIST_INPLACE_KERNEL(addcdiv);

} // namespace at::native::xpu
