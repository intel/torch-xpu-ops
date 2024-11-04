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

TORCH_XPU_API FOREACH_BINARY_SCALAR_INPLACE_KERNEL(add);
TORCH_XPU_API FOREACH_BINARY_SCALAR_KERNEL(add);
TORCH_XPU_API FOREACH_BINARY_SCALAR_INPLACE_KERNEL(sub);
TORCH_XPU_API FOREACH_BINARY_SCALAR_KERNEL(sub);
TORCH_XPU_API FOREACH_BINARY_SCALAR_INPLACE_KERNEL(mul);
TORCH_XPU_API FOREACH_BINARY_SCALAR_KERNEL(mul);
TORCH_XPU_API FOREACH_BINARY_SCALAR_INPLACE_KERNEL(div);
TORCH_XPU_API FOREACH_BINARY_SCALAR_KERNEL(div);
TORCH_XPU_API FOREACH_BINARY_SCALAR_INPLACE_KERNEL(clamp_max);
TORCH_XPU_API FOREACH_BINARY_SCALAR_KERNEL(clamp_max);
TORCH_XPU_API FOREACH_BINARY_SCALAR_INPLACE_KERNEL(clamp_min);
TORCH_XPU_API FOREACH_BINARY_SCALAR_KERNEL(clamp_min);
TORCH_XPU_API FOREACH_BINARY_SCALAR_INPLACE_KERNEL(pow);
TORCH_XPU_API FOREACH_BINARY_SCALAR_KERNEL(pow);
TORCH_XPU_API FOREACH_BINARY_SCALAR_KERNEL(pow_list);

} // namespace at::native::xpu
