#include <ATen/native/ForeachUtils.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/ForeachBinaryOpListKernels.h>
#include <ATen/native/xpu/sycl/ForeachPointwiseOpListKernels.h>
#include <ATen/native/xpu/sycl/ForeachTernaryOpListKernels.h>

namespace at {

#define FOREACH_BINARY_OP_LIST(NAME, DIVISION_OP)                           \
  void XPUNativeFunctions::_foreach_##NAME##_(                              \
      TensorList tensors1, TensorList tensors2) {                           \
    at::native::check_foreach_api_restrictions(tensors1, tensors2);         \
    if (!at::native::can_use_fast_route(tensors1, tensors2, DIVISION_OP)) { \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow_(         \
          tensors1, tensors2);                                              \
    }                                                                       \
                                                                            \
    at::native::xpu::FOREACH_BINARY_LIST_INPLACE_KERNEL_NAME(NAME)(         \
        tensors1, tensors2);                                                \
  }                                                                         \
                                                                            \
  std::vector<Tensor> XPUNativeFunctions::_foreach_##NAME(                  \
      TensorList tensors1, TensorList tensors2) {                           \
    at::native::check_foreach_api_restrictions(tensors1, tensors2);         \
    if (!at::native::can_use_fast_route(tensors1, tensors2, DIVISION_OP)) { \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow(          \
          tensors1, tensors2);                                              \
    }                                                                       \
                                                                            \
    return at::native::xpu::FOREACH_BINARY_LIST_KERNEL_NAME(NAME)(          \
        tensors1, tensors2);                                                \
  }

#define FOREACH_BINARY_OP_LIST_ALPHA(NAME)                                \
  void XPUNativeFunctions::_foreach_##NAME##_(                            \
      TensorList tensors1, TensorList tensors2, const Scalar& alpha) {    \
    at::native::check_foreach_api_restrictions(tensors1, tensors2);       \
    if (!at::native::can_use_fast_route({tensors1, tensors2}, alpha)) {   \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow_(       \
          tensors1, tensors2, alpha);                                     \
    }                                                                     \
                                                                          \
    at::native::xpu::FOREACH_BINARY_LIST_ALPHA_INPLACE_KERNEL_NAME(NAME)( \
        tensors1, tensors2, alpha);                                       \
  }                                                                       \
                                                                          \
  std::vector<Tensor> XPUNativeFunctions::_foreach_##NAME(                \
      TensorList tensors1, TensorList tensors2, const Scalar& alpha) {    \
    at::native::check_foreach_api_restrictions(tensors1, tensors2);       \
    if (!at::native::can_use_fast_route({tensors1, tensors2}, alpha)) {   \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow(        \
          tensors1, tensors2, alpha);                                     \
    }                                                                     \
                                                                          \
    return at::native::xpu::FOREACH_BINARY_LIST_ALPHA_KERNEL_NAME(NAME)(  \
        tensors1, tensors2, alpha);                                       \
  }

FOREACH_BINARY_OP_LIST_ALPHA(add);
FOREACH_BINARY_OP_LIST(mul, false);
FOREACH_BINARY_OP_LIST(div, true);

#define FOREACH_POINTWISE_OP_TENSOR(NAME)                                      \
  std::vector<Tensor> XPUNativeFunctions::_foreach_##NAME(                     \
      TensorList input,                                                        \
      TensorList tensors1,                                                     \
      TensorList tensors2,                                                     \
      const Tensor& scalars_) {                                                \
    auto scalars =                                                             \
        at::native::convert_tensor_to_scalar_list(scalars_, input.size());     \
    at::native::check_foreach_api_restrictions(                                \
        input, tensors1, tensors2, scalars);                                   \
    if (!at::native::can_use_fast_route({input, tensors1, tensors2}) ||        \
        at::native::has_integral_tensor(input, /* includeBool */ true)) {      \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow(              \
          input, tensors1, tensors2, scalars);                                 \
    }                                                                          \
                                                                               \
    return native::xpu::foreach_##NAME##_kernel(                               \
        input, tensors1, tensors2, scalars);                                   \
  }                                                                            \
                                                                               \
  void XPUNativeFunctions::_foreach_##NAME##_(                                 \
      TensorList input,                                                        \
      TensorList tensors1,                                                     \
      TensorList tensors2,                                                     \
      const Tensor& scalars_) {                                                \
    auto scalars =                                                             \
        at::native::convert_tensor_to_scalar_list(scalars_, input.size());     \
    at::native::check_foreach_api_restrictions(                                \
        input, tensors1, tensors2, scalars);                                   \
    if (!at::native::can_use_fast_route(                                       \
            {input, tensors1, tensors2}, scalars) ||                           \
        at::native::has_integral_tensor(input, /* includeBool */ true)) {      \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow_(             \
          input, tensors1, tensors2, scalars);                                 \
    }                                                                          \
                                                                               \
    native::xpu::foreach_##NAME##_kernel_(input, tensors1, tensors2, scalars); \
  }

FOREACH_POINTWISE_OP_TENSOR(addcmul)
FOREACH_POINTWISE_OP_TENSOR(addcdiv)

std::vector<at::Tensor> XPUNativeFunctions::_foreach_lerp(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3) {
  at::native::check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  if (!at::native::can_use_fast_route(
          {tensors1, tensors2, tensors3}, {}, true)) {
    return at::native::foreach_tensor_ternary_lerp_slow(
        tensors1, tensors2, tensors3);
  }

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  native::xpu::foreach_lerp_list_kernel(tensors1, tensors2, tensors3, vec_res);
  return vec_res;
}

void XPUNativeFunctions::_foreach_lerp_(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3) {
  at::native::check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  if (!at::native::can_use_fast_route(
          {tensors1, tensors2, tensors3}, {}, true)) {
    return at::native::foreach_tensor_ternary_lerp_slow_(
        tensors1, tensors2, tensors3);
  }

  native::xpu::foreach_lerp_list_kernel_(tensors1, tensors2, tensors3);

  // TODO: Handle version bump in codegen.
  // increment_version
  for (const auto& t : tensors1) {
    t.unsafeGetTensorImpl()->bump_version();
  }
}

} // namespace at
