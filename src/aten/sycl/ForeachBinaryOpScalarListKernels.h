#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

#define FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL_NAME(NAME) \
  foreach_binary_##NAME##_scalarlist__kernel

#define FOREACH_BINARY_SCALARLIST_KERNEL_NAME(NAME) \
  foreach_binary_##NAME##_scalarlist_kernel

#define FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL(NAME)      \
  void FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL_NAME(NAME)( \
      TensorList tensors, at::ArrayRef<Scalar> scalars)

#define FOREACH_BINARY_SCALARLIST_KERNEL(NAME)                     \
  std::vector<Tensor> FOREACH_BINARY_SCALARLIST_KERNEL_NAME(NAME)( \
      TensorList tensors, at::ArrayRef<Scalar> scalars)

FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL(add);
FOREACH_BINARY_SCALARLIST_KERNEL(add);

} // namespace at::native::xpu
