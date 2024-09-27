#include <ATen/native/ForeachUtils.h>

#include <ATen/native/xpu/sycl/ForeachBinaryOpScalarKernels.h>
#include <ATen/native/xpu/sycl/ForeachPointwiseOpScalarKernels.h>
#include <ATen/native/xpu/sycl/ForeachTernaryOpScalarKernels.h>

namespace at {

namespace native {

::std::vector<at::Tensor> foreach_tensor_add_scalar_kernel_slow(
    at::TensorList self,
    const at::Scalar& scalar);
void foreach_tensor_add_scalar_kernel_slow_(
    at::TensorList self,
    const at::Scalar& scalar);

::std::vector<at::Tensor> foreach_tensor_mul_scalar_kernel_slow(
    at::TensorList self,
    const at::Scalar& scalar);
void foreach_tensor_mul_scalar_kernel_slow_(
    at::TensorList self,
    const at::Scalar& scalar);

::std::vector<at::Tensor> foreach_tensor_div_scalar_kernel_slow(
    at::TensorList self,
    const at::Scalar& scalar);
void foreach_tensor_div_scalar_kernel_slow_(
    at::TensorList self,
    const at::Scalar& scalar);

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

FOREACH_BINARY_OP_SCALAR(add, /*div_op*/ false);
FOREACH_BINARY_OP_SCALAR(mul, /*div_op*/ false);
FOREACH_BINARY_OP_SCALAR(div, /*div_op*/ true);

::std::vector<at::Tensor> foreach_tensor_addcmul_scalar_slow(
    at::TensorList self,
    at::TensorList tensor1,
    at::TensorList tensor2,
    const at::Scalar& value);
void foreach_tensor_addcmul_scalar_slow_(
    at::TensorList self,
    at::TensorList tensor1,
    at::TensorList tensor2,
    const at::Scalar& value);

::std::vector<at::Tensor> foreach_tensor_addcdiv_scalar_slow(
    at::TensorList self,
    at::TensorList tensor1,
    at::TensorList tensor2,
    const at::Scalar& value);
void foreach_tensor_addcdiv_scalar_slow_(
    at::TensorList self,
    at::TensorList tensor1,
    at::TensorList tensor2,
    const at::Scalar& value);

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

::std::vector<at::Tensor> foreach_tensor_lerp_list_kernel_slow(
    at::TensorList self,
    at::TensorList tensors1,
    const at::Scalar& weight);
void foreach_tensor_lerp_list_kernel_slow_(
    at::TensorList self,
    at::TensorList tensors1,
    const at::Scalar& weight);

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
