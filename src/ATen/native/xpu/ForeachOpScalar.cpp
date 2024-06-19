#include <ATen/native/ForeachUtils.h>

#include <ATen/native/xpu/sycl/ForeachBinaryOpScalarKernels.h>
#include <ATen/native/xpu/sycl/ForeachPointwiseOpScalarKernels.h>

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
    const at::Scalar& value = 1);
void foreach_tensor_addcmul_scalar_slow_(
    at::TensorList self,
    at::TensorList tensor1,
    at::TensorList tensor2,
    const at::Scalar& value = 1);

::std::vector<at::Tensor> foreach_tensor_addcdiv_scalar_slow(
    at::TensorList self,
    at::TensorList tensor1,
    at::TensorList tensor2,
    const at::Scalar& value = 1);
void foreach_tensor_addcdiv_scalar_slow_(
    at::TensorList self,
    at::TensorList tensor1,
    at::TensorList tensor2,
    const at::Scalar& value = 1);

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
} // namespace native

} // namespace at
