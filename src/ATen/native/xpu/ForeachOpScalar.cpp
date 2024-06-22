#include <ATen/native/ForeachUtils.h>

#include <ATen/xpu/XPUNativeFunctions.h>
#include <ATen/native/xpu/sycl/ForeachBinaryOpScalarKernels.h>
#include <ATen/native/xpu/sycl/ForeachPointwiseOpScalarKernels.h>
#include <ATen/native/xpu/sycl/ForeachTernaryOpScalarKernels.h>

namespace at {

#define FOREACH_BINARY_OP_SCALAR(NAME, DIV_OP)                        \
  void XPUNativeFunctions::_foreach_##NAME##_(                        \
      TensorList tensors, const Scalar& scalar) {                     \
    at::native::check_foreach_api_restrictions(tensors);              \
    if (!at::native::can_use_fast_route(tensors, scalar, DIV_OP)) {   \
      return at::native::foreach_tensor_##NAME##_scalar_kernel_slow_( \
          tensors, scalar);                                           \
    }                                                                 \
                                                                      \
    at::native::xpu::FOREACH_BINARY_SCALAR_INPLACE_KERNEL_NAME(NAME)( \
        tensors, scalar);                                             \
  }                                                                   \
                                                                      \
  std::vector<Tensor> XPUNativeFunctions::_foreach_##NAME(            \
      TensorList tensors, const Scalar& scalar) {                     \
    at::native::check_foreach_api_restrictions(tensors);              \
    if (!at::native::can_use_fast_route(tensors, scalar, DIV_OP)) {   \
      return at::native::foreach_tensor_##NAME##_scalar_kernel_slow(  \
          tensors, scalar);                                           \
    }                                                                 \
                                                                      \
    return at::native::xpu::FOREACH_BINARY_SCALAR_KERNEL_NAME(NAME)(  \
        tensors, scalar);                                             \
  }

FOREACH_BINARY_OP_SCALAR(add, /*div_op*/ false);
FOREACH_BINARY_OP_SCALAR(mul, /*div_op*/ false);
FOREACH_BINARY_OP_SCALAR(div, /*div_op*/ true);

#define FOREACH_POINTWISE_OP_SCALAR(NAME)                                     \
  std::vector<Tensor> XPUNativeFunctions::_foreach_##NAME(                    \
      TensorList input,                                                       \
      TensorList tensors1,                                                    \
      TensorList tensors2,                                                    \
      const Scalar& scalar) {                                                 \
    at::native::check_foreach_api_restrictions(input, tensors1, tensors2);    \
                                                                              \
    if (!at::native::can_use_fast_route(                                      \
            {input, tensors1, tensors2}, scalar) ||                           \
        at::native::has_integral_tensor(input, /* includeBool */ true)) {     \
      return at::native::foreach_tensor_##NAME##_scalar_slow(                 \
          input, tensors1, tensors2, scalar);                                 \
    }                                                                         \
                                                                              \
    return native::xpu::foreach_##NAME##_kernel(                              \
        input, tensors1, tensors2, scalar);                                   \
  }                                                                           \
                                                                              \
  void XPUNativeFunctions::_foreach_##NAME##_(                                \
      TensorList input,                                                       \
      TensorList tensors1,                                                    \
      TensorList tensors2,                                                    \
      const Scalar& scalar) {                                                 \
    at::native::check_foreach_api_restrictions(input, tensors1, tensors2);    \
                                                                              \
    if (!at::native::can_use_fast_route(                                      \
            {input, tensors1, tensors2}, scalar) ||                           \
        at::native::has_integral_tensor(input, /* includeBool */ true)) {     \
      return at::native::foreach_tensor_##NAME##_scalar_slow_(                \
          input, tensors1, tensors2, scalar);                                 \
    }                                                                         \
                                                                              \
    native::xpu::foreach_##NAME##_kernel_(input, tensors1, tensors2, scalar); \
  }

FOREACH_POINTWISE_OP_SCALAR(addcmul)
FOREACH_POINTWISE_OP_SCALAR(addcdiv)

std::vector<at::Tensor> XPUNativeFunctions::_foreach_lerp(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& weight) {
  at::native::check_foreach_api_restrictions(tensors1, tensors2);
  if (!at::native::can_use_fast_route({tensors1, tensors2}, {}, true)) {
    return at::native::foreach_tensor_lerp_list_kernel_slow(
        tensors1, tensors2, weight);
  }

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  native::xpu::foreach_lerp_scalar_kernel(tensors1, tensors2, weight, vec_res);

  return vec_res;
}

void XPUNativeFunctions::_foreach_lerp_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& weight) {
  at::native::check_foreach_api_restrictions(tensors1, tensors2);
  if (!at::native::can_use_fast_route({tensors1, tensors2}, {}, true)) {
    return at::native::foreach_tensor_lerp_list_kernel_slow_(
        tensors1, tensors2, weight);
  }

  native::xpu::foreach_lerp_scalar_kernel_(tensors1, tensors2, weight);
}
} // namespace at
