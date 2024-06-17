#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

#define FOREACH_BINARY_SCALAR_INPLACE_KERNEL_NAME(NAME) \
  foreach_binary_##NAME##_scalar__kernel

#define FOREACH_BINARY_SCALAR_KERNEL_NAME(NAME) \
  foreach_binary_##NAME##_scalar_kernel

#define FOREACH_BINARY_SCALAR_INPLACE_KERNEL(NAME)      \
  void FOREACH_BINARY_SCALAR_INPLACE_KERNEL_NAME(NAME)( \
      TensorList tensors, const Scalar& scalar)

#define FOREACH_BINARY_SCALAR_KERNEL(NAME)                     \
  std::vector<Tensor> FOREACH_BINARY_SCALAR_KERNEL_NAME(NAME)( \
      TensorList tensors, const Scalar& scalar)

FOREACH_BINARY_SCALAR_INPLACE_KERNEL(add);
FOREACH_BINARY_SCALAR_KERNEL(add);
FOREACH_BINARY_SCALAR_INPLACE_KERNEL(mul);
FOREACH_BINARY_SCALAR_KERNEL(mul);
FOREACH_BINARY_SCALAR_INPLACE_KERNEL(div);
FOREACH_BINARY_SCALAR_KERNEL(div);

} // namespace at::native::xpu
