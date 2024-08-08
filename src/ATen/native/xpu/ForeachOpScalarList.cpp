#include <ATen/native/ForeachUtils.h>

#include <ATen/native/xpu/sycl/ForeachBinaryOpScalarListKernels.h>
#include <ATen/native/xpu/sycl/ForeachPointwiseOpScalarListKernels.h>

#include <xpu/ATen/ops/_foreach_add_native.h>
#include <xpu/ATen/ops/_foreach_mul_native.h>

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

::std::vector<at::Tensor> foreach_tensor_add_scalarlist_kernel_slow(
    at::TensorList self,
    at::ArrayRef<at::Scalar> scalars);
void foreach_tensor_add_scalarlist_kernel_slow_(
    at::TensorList self,
    at::ArrayRef<at::Scalar> scalars);
::std::vector<at::Tensor> foreach_tensor_mul_scalarlist_kernel_slow(
    at::TensorList self,
    at::ArrayRef<at::Scalar> scalars);
void foreach_tensor_mul_scalarlist_kernel_slow_(
    at::TensorList self,
    at::ArrayRef<at::Scalar> scalars);

::std::vector<at::Tensor> foreach_tensor_div_scalar_kernel_slow(
    at::TensorList self,
    const at::Scalar& scalar);
void foreach_tensor_div_scalar_kernel_slow_(
    at::TensorList self,
    const at::Scalar& scalar);
::std::vector<at::Tensor> foreach_tensor_div_scalarlist_kernel_slow(
    at::TensorList self,
    at::ArrayRef<at::Scalar> scalars);
void foreach_tensor_div_scalarlist_kernel_slow_(
    at::TensorList self,
    at::ArrayRef<at::Scalar> scalars);

#define FOREACH_BINARY_OP_SCALARLIST(NAME, DIV_OP)                             \
  void foreach_tensor_##NAME##_scalar_kernel_xpu_(                             \
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
  std::vector<Tensor> foreach_tensor_##NAME##_scalar_kernel_xpu(               \
      TensorList tensors, at::ArrayRef<Scalar> scalars) {                      \
    check_foreach_api_restrictions(tensors, scalars);                          \
    if (!can_use_fast_route(tensors, scalars, DIV_OP)) {                       \
      return foreach_tensor_##NAME##_scalarlist_kernel_slow(tensors, scalars); \
    }                                                                          \
                                                                               \
    return xpu::FOREACH_BINARY_SCALARLIST_KERNEL_NAME(NAME)(tensors, scalars); \
  }

FOREACH_BINARY_OP_SCALARLIST(add, /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST(mul, /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST(div, /*div_op*/ true);

void foreach_tensor_addcmul_scalar_slow_(
    at::TensorList self,
    at::TensorList tensor1,
    at::TensorList tensor2,
    const at::Scalar& value = 1);
::std::vector<at::Tensor> foreach_tensor_addcmul_scalar_slow(
    at::TensorList self,
    at::TensorList tensor1,
    at::TensorList tensor2,
    const at::Scalar& value = 1);
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
void foreach_tensor_addcdiv_scalar_slow_(
    at::TensorList self,
    at::TensorList tensor1,
    at::TensorList tensor2,
    const at::Scalar& value = 1);
::std::vector<at::Tensor> foreach_tensor_addcdiv_scalar_slow(
    at::TensorList self,
    at::TensorList tensor1,
    at::TensorList tensor2,
    const at::Scalar& value = 1);
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
