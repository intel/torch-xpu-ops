#include <ATen/native/ForeachUtils.h>
#include <ATen/native/xpu/sycl/ForeachBinaryOpListKernels.h>
#include <ATen/native/xpu/sycl/ForeachPointwiseOpListKernels.h>
#include <ATen/native/xpu/sycl/ForeachTernaryOpListKernels.h>

#include <ATen/ops/empty_like.h>

namespace at {
namespace native {

::std::vector<at::Tensor> foreach_tensor_mul_list_kernel_slow(
    at::TensorList self,
    at::TensorList other);
void foreach_tensor_mul_list_kernel_slow_(
    at::TensorList self,
    at::TensorList other);

::std::vector<at::Tensor> foreach_tensor_div_list_kernel_slow(
    at::TensorList self,
    at::TensorList other);
void foreach_tensor_div_list_kernel_slow_(
    at::TensorList self,
    at::TensorList other);

::std::vector<at::Tensor> foreach_tensor_add_list_kernel_slow(
    at::TensorList self,
    at::TensorList other,
    const at::Scalar& alpha);
void foreach_tensor_add_list_kernel_slow_(
    at::TensorList self,
    at::TensorList other,
    const at::Scalar& alpha);

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
FOREACH_BINARY_OP_LIST(mul, false);
FOREACH_BINARY_OP_LIST(div, true);

::std::vector<at::Tensor> foreach_tensor_addcmul_scalarlist_slow(
    at::TensorList self,
    at::TensorList tensor1,
    at::TensorList tensor2,
    at::ArrayRef<at::Scalar> scalars);
void foreach_tensor_addcmul_scalarlist_slow_(
    at::TensorList self,
    at::TensorList tensor1,
    at::TensorList tensor2,
    at::ArrayRef<at::Scalar> scalars);

::std::vector<at::Tensor> foreach_tensor_addcdiv_scalarlist_slow(
    at::TensorList self,
    at::TensorList tensor1,
    at::TensorList tensor2,
    at::ArrayRef<at::Scalar> scalars);
void foreach_tensor_addcdiv_scalarlist_slow_(
    at::TensorList self,
    at::TensorList tensor1,
    at::TensorList tensor2,
    at::ArrayRef<at::Scalar> scalars);

#define FOREACH_POINTWISE_OP_TENSOR(NAME)                                  \
  std::vector<Tensor> foreach_tensor_##NAME##_list_kernel_xpu(             \
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
  void foreach_tensor_##NAME##_list_kernel_xpu_(                           \
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

::std::vector<at::Tensor> foreach_tensor_ternary_lerp_slow(
    at::TensorList self,
    at::TensorList tensors1,
    at::TensorList weights);

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

void foreach_tensor_ternary_lerp_slow_(
    at::TensorList self,
    at::TensorList tensors1,
    at::TensorList weights);

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

} // namespace native
} // namespace at
