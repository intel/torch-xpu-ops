#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

#define FOREACH_BINARY_LIST_INPLACE_KERNEL_NAME(NAME) \
  foreach_binary_##NAME##_list__kernel

#define FOREACH_BINARY_LIST_KERNEL_NAME(NAME) \
  foreach_binary_##NAME##_list_kernel

#define FOREACH_BINARY_LIST_INPLACE_KERNEL(NAME)      \
  void FOREACH_BINARY_LIST_INPLACE_KERNEL_NAME(NAME)( \
      TensorList tensor1, TensorList tensor2)

#define FOREACH_BINARY_LIST_KERNEL(NAME)                     \
  std::vector<Tensor> FOREACH_BINARY_LIST_KERNEL_NAME(NAME)( \
      TensorList tensor1, TensorList tensor2)

#define FOREACH_BINARY_LIST_ALPHA_INPLACE_KERNEL_NAME(NAME) \
  foreach_binary_##NAME##_list_alpha__kernel

#define FOREACH_BINARY_LIST_ALPHA_KERNEL_NAME(NAME) \
  foreach_binary_##NAME##_list_alpha_kernel

#define FOREACH_BINARY_LIST_ALPHA_INPLACE_KERNEL(NAME)      \
  void FOREACH_BINARY_LIST_ALPHA_INPLACE_KERNEL_NAME(NAME)( \
      TensorList tensor1, TensorList tensor2, const Scalar& alpha)

#define FOREACH_BINARY_LIST_ALPHA_KERNEL(NAME)                     \
  std::vector<Tensor> FOREACH_BINARY_LIST_ALPHA_KERNEL_NAME(NAME)( \
      TensorList tensor1, TensorList tensor2, const Scalar& alpha)

FOREACH_BINARY_LIST_ALPHA_INPLACE_KERNEL(add);
FOREACH_BINARY_LIST_ALPHA_KERNEL(add);
FOREACH_BINARY_LIST_INPLACE_KERNEL(mul);
FOREACH_BINARY_LIST_KERNEL(mul);
FOREACH_BINARY_LIST_INPLACE_KERNEL(div);
FOREACH_BINARY_LIST_KERNEL(div);

} // namespace at::native::xpu
