#include <ATen/native/BinaryOps.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_addcdiv_native.h>
#include <ATen/ops/_foreach_addcmul_native.h>
#include <ATen/ops/_foreach_clamp_max_native.h>
#include <ATen/ops/_foreach_clamp_min_native.h>
#include <ATen/ops/_foreach_div_native.h>
#include <ATen/ops/_foreach_lerp_native.h>
#include <ATen/ops/_foreach_mul_native.h>
#include <ATen/ops/_foreach_pow_native.h>
#include <ATen/ops/_foreach_sub_native.h>

#include <ATen/native/xpu/sycl/ForeachBinaryOpScalarKernels.h>
#include <ATen/native/xpu/sycl/ForeachPointwiseOpScalarKernels.h>
#include <ATen/native/xpu/sycl/ForeachTernaryOpScalarKernels.h>

namespace at {
namespace native {

#define FOREACH_BINARY_OP_SCALAR(NAME, DIV_OP)                             \
  void foreach_tensor_##NAME##_scalar_kernel_xpu_(                         \
      TensorList tensors, const Scalar& scalar) {                          \
    check_foreach_api_restrictions(tensors);                               \
    if (!can_use_fast_route(tensors, scalar, DIV_OP)) {                    \
      return foreach_tensor_##NAME##_scalar_kernel_slow_(tensors, scalar); \
    }                                                                      \
                                                                           \
    xpu::FOREACH_BINARY_SCALAR_INPLACE_KERNEL_NAME(NAME)(tensors, scalar); \
  }                                                                        \
                                                                           \
  std::vector<Tensor> foreach_tensor_##NAME##_scalar_kernel_xpu(           \
      TensorList tensors, const Scalar& scalar) {                          \
    check_foreach_api_restrictions(tensors);                               \
    if (!can_use_fast_route(tensors, scalar, DIV_OP)) {                    \
      return foreach_tensor_##NAME##_scalar_kernel_slow(tensors, scalar);  \
    }                                                                      \
                                                                           \
    return xpu::FOREACH_BINARY_SCALAR_KERNEL_NAME(NAME)(tensors, scalar);  \
  }

// In the case of subtraction, we dont allow scalar to be boolean following the
// torch.sub logic
#define FOREACH_BINARY_OP_SCALAR_NO_BOOLEAN(NAME, DIV_OP)                  \
  void foreach_tensor_##NAME##_scalar_kernel_xpu_(                         \
      TensorList tensors, const Scalar& scalar) {                          \
    check_foreach_api_restrictions(tensors);                               \
    sub_check(tensors[0], scalar);                                         \
    if (!can_use_fast_route(tensors, scalar, DIV_OP)) {                    \
      return foreach_tensor_##NAME##_scalar_kernel_slow_(tensors, scalar); \
    }                                                                      \
                                                                           \
    xpu::FOREACH_BINARY_SCALAR_INPLACE_KERNEL_NAME(NAME)(tensors, scalar); \
  }                                                                        \
                                                                           \
  std::vector<Tensor> foreach_tensor_##NAME##_scalar_kernel_xpu(           \
      TensorList tensors, const Scalar& scalar) {                          \
    check_foreach_api_restrictions(tensors);                               \
    sub_check(tensors[0], scalar);                                         \
    if (!can_use_fast_route(tensors, scalar, DIV_OP)) {                    \
      return foreach_tensor_##NAME##_scalar_kernel_slow(tensors, scalar);  \
    }                                                                      \
                                                                           \
    return xpu::FOREACH_BINARY_SCALAR_KERNEL_NAME(NAME)(tensors, scalar);  \
  }

FOREACH_BINARY_OP_SCALAR(add, /*div_op*/ false);
FOREACH_BINARY_OP_SCALAR_NO_BOOLEAN(sub, /*div_op*/ false);
FOREACH_BINARY_OP_SCALAR(mul, /*div_op*/ false);
FOREACH_BINARY_OP_SCALAR(div, /*div_op*/ true);
FOREACH_BINARY_OP_SCALAR(clamp_max, /*div_op*/ true);
FOREACH_BINARY_OP_SCALAR(clamp_min, /*div_op*/ true);
FOREACH_BINARY_OP_SCALAR(pow, /*div_op*/ true);

std::vector<Tensor> foreach_scalar_pow_list_kernel_xpu(
    const Scalar& scalar,
    TensorList exponent) {
  check_foreach_api_restrictions(exponent);
  if (!can_use_fast_route(exponent)) {
    return foreach_scalar_pow_list_kernel_slow(scalar, exponent);
  }
  return xpu::foreach_binary_pow_list_scalar_kernel(exponent, scalar);
}

#define FOREACH_POINTWISE_OP_SCALAR(NAME)                                   \
  std::vector<Tensor> foreach_tensor_##NAME##_scalar_xpu(                   \
      TensorList input,                                                     \
      TensorList tensors1,                                                  \
      TensorList tensors2,                                                  \
      const Scalar& scalar) {                                               \
    check_foreach_api_restrictions(input, tensors1, tensors2);              \
                                                                            \
    if (!can_use_fast_route({input, tensors1, tensors2}, scalar) ||         \
        has_integral_tensor(input, /* includeBool */ true)) {               \
      return foreach_tensor_##NAME##_scalar_slow(                           \
          input, tensors1, tensors2, scalar);                               \
    }                                                                       \
                                                                            \
    return xpu::foreach_##NAME##_kernel(input, tensors1, tensors2, scalar); \
  }                                                                         \
                                                                            \
  void foreach_tensor_##NAME##_scalar_xpu_(                                 \
      TensorList input,                                                     \
      TensorList tensors1,                                                  \
      TensorList tensors2,                                                  \
      const Scalar& scalar) {                                               \
    check_foreach_api_restrictions(input, tensors1, tensors2);              \
                                                                            \
    if (!can_use_fast_route({input, tensors1, tensors2}, scalar) ||         \
        has_integral_tensor(input, /* includeBool */ true)) {               \
      return foreach_tensor_##NAME##_scalar_slow_(                          \
          input, tensors1, tensors2, scalar);                               \
    }                                                                       \
                                                                            \
    xpu::foreach_##NAME##_kernel_(input, tensors1, tensors2, scalar);       \
  }

FOREACH_POINTWISE_OP_SCALAR(addcmul)
FOREACH_POINTWISE_OP_SCALAR(addcdiv)

std::vector<at::Tensor> foreach_tensor_lerp_list_xpu(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& weight) {
  check_foreach_api_restrictions(tensors1, tensors2);
  if (!can_use_fast_route({tensors1, tensors2}, {}, true)) {
    return foreach_tensor_lerp_list_kernel_slow(tensors1, tensors2, weight);
  }

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::empty_like(t));
  }

  xpu::foreach_lerp_scalar_kernel(tensors1, tensors2, weight, vec_res);

  return vec_res;
}

void foreach_tensor_lerp_list_xpu_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& weight) {
  check_foreach_api_restrictions(tensors1, tensors2);
  if (!can_use_fast_route({tensors1, tensors2}, {}, true)) {
    return foreach_tensor_lerp_list_kernel_slow_(tensors1, tensors2, weight);
  }

  xpu::foreach_lerp_scalar_kernel_(tensors1, tensors2, weight);
}

} // namespace native
} // namespace at
