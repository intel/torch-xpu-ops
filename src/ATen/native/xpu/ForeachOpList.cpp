#include <ATen/native/ForeachUtils.h>
#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_addcdiv_native.h>
#include <ATen/ops/_foreach_addcmul_native.h>
#include <ATen/ops/_foreach_clamp_max_native.h>
#include <ATen/ops/_foreach_clamp_min_native.h>
#include <ATen/ops/_foreach_copy_native.h>
#include <ATen/ops/_foreach_div_native.h>
#include <ATen/ops/_foreach_lerp_native.h>
#include <ATen/ops/_foreach_mul_native.h>
#include <ATen/ops/_foreach_pow_native.h>
#include <ATen/ops/_foreach_sub_native.h>

#include <ATen/native/xpu/sycl/ForeachBinaryOpListKernels.h>
#include <ATen/native/xpu/sycl/ForeachCopyKernels.h>
#include <ATen/native/xpu/sycl/ForeachPointwiseOpListKernels.h>
#include <ATen/native/xpu/sycl/ForeachTernaryOpListKernels.h>

#include <ATen/ops/empty_like.h>

namespace at {
namespace native {

#define FOREACH_BINARY_OP_LIST(NAME, DIVISION_OP)                           \
  void foreach_tensor_##NAME##_list_kernel_xpu_(                            \
      TensorList tensors1, TensorList tensors2) {                           \
    check_foreach_api_restrictions(tensors1, tensors2);                     \
    if (!can_use_fast_route(tensors1, tensors2, DIVISION_OP)) {             \
      return foreach_tensor_##NAME##_list_kernel_slow_(tensors1, tensors2); \
    }                                                                       \
                                                                            \
    xpu::FOREACH_BINARY_LIST_INPLACE_KERNEL_NAME(NAME)(tensors1, tensors2); \
  }                                                                         \
                                                                            \
  std::vector<Tensor> foreach_tensor_##NAME##_list_kernel_xpu(              \
      TensorList tensors1, TensorList tensors2) {                           \
    check_foreach_api_restrictions(tensors1, tensors2);                     \
    if (!can_use_fast_route(tensors1, tensors2, DIVISION_OP)) {             \
      return foreach_tensor_##NAME##_list_kernel_slow(tensors1, tensors2);  \
    }                                                                       \
                                                                            \
    return xpu::FOREACH_BINARY_LIST_KERNEL_NAME(NAME)(tensors1, tensors2);  \
  }

#define FOREACH_BINARY_OP_LIST_ALPHA(NAME)                             \
  void foreach_tensor_##NAME##_list_kernel_xpu_(                       \
      TensorList tensors1, TensorList tensors2, const Scalar& alpha) { \
    check_foreach_api_restrictions(tensors1, tensors2);                \
    if (!can_use_fast_route({tensors1, tensors2}, alpha)) {            \
      return foreach_tensor_##NAME##_list_kernel_slow_(                \
          tensors1, tensors2, alpha);                                  \
    }                                                                  \
                                                                       \
    xpu::FOREACH_BINARY_LIST_ALPHA_INPLACE_KERNEL_NAME(NAME)(          \
        tensors1, tensors2, alpha);                                    \
  }                                                                    \
                                                                       \
  std::vector<Tensor> foreach_tensor_##NAME##_list_kernel_xpu(         \
      TensorList tensors1, TensorList tensors2, const Scalar& alpha) { \
    check_foreach_api_restrictions(tensors1, tensors2);                \
    if (!can_use_fast_route({tensors1, tensors2}, alpha)) {            \
      return foreach_tensor_##NAME##_list_kernel_slow(                 \
          tensors1, tensors2, alpha);                                  \
    }                                                                  \
                                                                       \
    return xpu::FOREACH_BINARY_LIST_ALPHA_KERNEL_NAME(NAME)(           \
        tensors1, tensors2, alpha);                                    \
  }

FOREACH_BINARY_OP_LIST_ALPHA(add);
FOREACH_BINARY_OP_LIST_ALPHA(sub);
FOREACH_BINARY_OP_LIST(mul, false);
FOREACH_BINARY_OP_LIST(div, true);
FOREACH_BINARY_OP_LIST(clamp_max, true);
FOREACH_BINARY_OP_LIST(clamp_min, true);
FOREACH_BINARY_OP_LIST(pow, true);

#define FOREACH_POINTWISE_OP_TENSOR(NAME)                                  \
  std::vector<Tensor> foreach_tensor_##NAME##_tensor_xpu(                  \
      TensorList input,                                                    \
      TensorList tensors1,                                                 \
      TensorList tensors2,                                                 \
      const Tensor& scalars_) {                                            \
    auto scalars =                                                         \
        at::native::convert_tensor_to_scalar_list(scalars_, input.size()); \
    at::native::check_foreach_api_restrictions(                            \
        input, tensors1, tensors2, scalars);                               \
    if (!at::native::can_use_fast_route({input, tensors1, tensors2}) ||    \
        at::native::has_integral_tensor(input, /* includeBool */ true)) {  \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow(          \
          input, tensors1, tensors2, scalars);                             \
    }                                                                      \
                                                                           \
    return native::xpu::foreach_##NAME##_kernel(                           \
        input, tensors1, tensors2, scalars);                               \
  }                                                                        \
                                                                           \
  void foreach_tensor_##NAME##_tensor_xpu_(                                \
      TensorList input,                                                    \
      TensorList tensors1,                                                 \
      TensorList tensors2,                                                 \
      const Tensor& scalars_) {                                            \
    auto scalars = convert_tensor_to_scalar_list(scalars_, input.size());  \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);    \
    if (!can_use_fast_route({input, tensors1, tensors2}, scalars) ||       \
        has_integral_tensor(input, /* includeBool */ true)) {              \
      return foreach_tensor_##NAME##_scalarlist_slow_(                     \
          input, tensors1, tensors2, scalars);                             \
    }                                                                      \
                                                                           \
    xpu::foreach_##NAME##_kernel_(input, tensors1, tensors2, scalars);     \
  }

FOREACH_POINTWISE_OP_TENSOR(addcmul)
FOREACH_POINTWISE_OP_TENSOR(addcdiv)

std::vector<at::Tensor> foreach_tensor_lerp_ternary_xpu(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3) {
  check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  if (!can_use_fast_route({tensors1, tensors2, tensors3}, {}, true)) {
    return foreach_tensor_ternary_lerp_slow(tensors1, tensors2, tensors3);
  }

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::empty_like(t));
  }

  xpu::foreach_lerp_list_kernel(tensors1, tensors2, tensors3, vec_res);
  return vec_res;
}

void foreach_tensor_lerp_ternary_xpu_(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3) {
  check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  if (!can_use_fast_route({tensors1, tensors2, tensors3}, {}, true)) {
    return foreach_tensor_ternary_lerp_slow_(tensors1, tensors2, tensors3);
  }

  xpu::foreach_lerp_list_kernel_(tensors1, tensors2, tensors3);

  // TODO: Handle version bump in codegen.
  // increment_version
  for (const auto& t : tensors1) {
    t.unsafeGetTensorImpl()->bump_version();
  }
}

void foreach_tensor_copy_list_kernel_xpu_(
    TensorList self,
    TensorList src,
    bool non_blocking) {
  check_foreach_api_restrictions(self, src);
  if (!can_use_fast_route(
          self, src, /* does_op_promote_integer_inputs_to_float */ false)) {
    return foreach_tensor_copy_list_kernel_slow_(self, src, non_blocking);
  }

  xpu::foreach_copy_list_kernel_(self, src);

  // increment_version
  for (const auto& t : self) {
    t.unsafeGetTensorImpl()->bump_version();
  }
}

} // namespace native
} // namespace at
