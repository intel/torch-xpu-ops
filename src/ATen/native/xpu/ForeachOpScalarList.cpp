#include <ATen/native/BinaryOps.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_addcdiv_native.h>
#include <ATen/ops/_foreach_addcmul_native.h>
#include <ATen/ops/_foreach_clamp_max_native.h>
#include <ATen/ops/_foreach_clamp_min_native.h>
#include <ATen/ops/_foreach_div_native.h>
#include <ATen/ops/_foreach_mul_native.h>
#include <ATen/ops/_foreach_pow_native.h>
#include <ATen/ops/_foreach_sub_native.h>

#include <ATen/native/xpu/sycl/ForeachBinaryOpScalarListKernels.h>
#include <ATen/native/xpu/sycl/ForeachPointwiseOpScalarListKernels.h>

#include <xpu/ATen/ops/_foreach_add_native.h>
#include <xpu/ATen/ops/_foreach_mul_native.h>

namespace at {
namespace native {

#define FOREACH_BINARY_OP_SCALARLIST(NAME, DIV_OP)                             \
  void foreach_tensor_##NAME##_scalarlist_kernel_xpu_(                         \
      TensorList tensors, at::ArrayRef<Scalar> scalars) {                      \
    check_foreach_api_restrictions(tensors, scalars);                          \
    if (!can_use_fast_route(tensors, scalars, DIV_OP)) {                       \
      return foreach_tensor_##NAME##_scalarlist_kernel_slow_(                  \
          tensors, scalars);                                                   \
    }                                                                          \
                                                                               \
    xpu::FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL_NAME(NAME)(                  \
        tensors, scalars);                                                     \
  }                                                                            \
                                                                               \
  std::vector<Tensor> foreach_tensor_##NAME##_scalarlist_kernel_xpu(           \
      TensorList tensors, at::ArrayRef<Scalar> scalars) {                      \
    check_foreach_api_restrictions(tensors, scalars);                          \
    if (!can_use_fast_route(tensors, scalars, DIV_OP)) {                       \
      return foreach_tensor_##NAME##_scalarlist_kernel_slow(tensors, scalars); \
    }                                                                          \
                                                                               \
    return xpu::FOREACH_BINARY_SCALARLIST_KERNEL_NAME(NAME)(tensors, scalars); \
  }

// This does not use FOREACH_BINARY_OP_SCALARLIST because
// In the case of subtraction, we dont allow scalar to be boolean following the
// torch.sub logic
void foreach_tensor_sub_scalarlist_kernel_xpu_(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  check_foreach_api_restrictions(tensors, scalars);
  for (const auto i : c10::irange(tensors.size())) {
    sub_check(tensors[i], scalars[i]);
  }

  if (!can_use_fast_route({tensors}, scalars, false)) {
    return foreach_tensor_sub_scalarlist_kernel_slow_(tensors, scalars);
  }

  xpu::FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL_NAME(sub)(tensors, scalars);
}

std::vector<Tensor> foreach_tensor_sub_scalarlist_kernel_xpu(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  check_foreach_api_restrictions(tensors, scalars);
  for (const auto i : c10::irange(tensors.size())) {
    sub_check(tensors[i], scalars[i]);
  }

  if (!can_use_fast_route({tensors}, scalars, false)) {
    return foreach_tensor_sub_scalarlist_kernel_slow(tensors, scalars);
  }

  return xpu::FOREACH_BINARY_SCALARLIST_KERNEL_NAME(sub)(tensors, scalars);
}

FOREACH_BINARY_OP_SCALARLIST(add, /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST(mul, /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST(div, /*div_op*/ true);
FOREACH_BINARY_OP_SCALARLIST(clamp_max, /*div_op*/ true);
FOREACH_BINARY_OP_SCALARLIST(clamp_min, /*div_op*/ true);
FOREACH_BINARY_OP_SCALARLIST(pow, true);

#define FOREACH_POINTWISE_OP_SCALARLIST(NAME)                                \
  std::vector<Tensor> foreach_tensor_##NAME##_scalarlist_xpu(                \
      TensorList input,                                                      \
      TensorList tensors1,                                                   \
      TensorList tensors2,                                                   \
      at::ArrayRef<Scalar> scalars) {                                        \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);      \
                                                                             \
    if (!can_use_fast_route({input, tensors1, tensors2}, scalars) ||         \
        has_integral_tensor(input, /* includeBool */ true)) {                \
      return foreach_tensor_##NAME##_scalarlist_slow(                        \
          input, tensors1, tensors2, scalars);                               \
    }                                                                        \
                                                                             \
    return xpu::foreach_##NAME##_kernel(input, tensors1, tensors2, scalars); \
  }                                                                          \
                                                                             \
  void foreach_tensor_##NAME##_scalarlist_xpu_(                              \
      TensorList input,                                                      \
      TensorList tensors1,                                                   \
      TensorList tensors2,                                                   \
      at::ArrayRef<Scalar> scalars) {                                        \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);      \
                                                                             \
    if (!can_use_fast_route({input, tensors1, tensors2}, scalars) ||         \
        has_integral_tensor(input, /* includeBool */ true)) {                \
      return foreach_tensor_##NAME##_scalarlist_slow_(                       \
          input, tensors1, tensors2, scalars);                               \
    }                                                                        \
                                                                             \
    xpu::foreach_##NAME##_kernel_(input, tensors1, tensors2, scalars);       \
  }

FOREACH_POINTWISE_OP_SCALARLIST(addcmul)
FOREACH_POINTWISE_OP_SCALARLIST(addcdiv)

}; // namespace native
} // namespace at
