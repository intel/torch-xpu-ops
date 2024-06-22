#include <ATen/native/ForeachUtils.h>

#include <ATen/XPUNativeFunctions.h>
#include <ATen/native/xpu/sycl/ForeachBinaryOpScalarListKernels.h>
#include <ATen/native/xpu/sycl/ForeachPointwiseOpScalarListKernels.h>

namespace at {

#define FOREACH_BINARY_OP_SCALARLIST(NAME, DIV_OP)                        \
  void XPUNativeFunctions::_foreach_##NAME##_(                            \
      TensorList tensors, at::ArrayRef<Scalar> scalars) {                 \
    at::native::check_foreach_api_restrictions(tensors, scalars);         \
    if (!at::native::can_use_fast_route(tensors, scalars, DIV_OP)) {      \
      return at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow_( \
          tensors, scalars);                                              \
    }                                                                     \
                                                                          \
    at::native::xpu::FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL_NAME(NAME)( \
        tensors, scalars);                                                \
  }                                                                       \
                                                                          \
  std::vector<Tensor> XPUNativeFunctions::_foreach_##NAME(                \
      TensorList tensors, at::ArrayRef<Scalar> scalars) {                 \
    at::native::check_foreach_api_restrictions(tensors, scalars);         \
    if (!at::native::can_use_fast_route(tensors, scalars, DIV_OP)) {      \
      return at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow(  \
          tensors, scalars);                                              \
    }                                                                     \
                                                                          \
    return at::native::xpu::FOREACH_BINARY_SCALARLIST_KERNEL_NAME(NAME)(  \
        tensors, scalars);                                                \
  }

FOREACH_BINARY_OP_SCALARLIST(add, /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST(mul, /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST(div, /*div_op*/ true);

#define FOREACH_POINTWISE_OP_SCALARLIST(NAME)                                  \
  std::vector<Tensor> XPUNativeFunctions::_foreach_##NAME(                     \
      TensorList input,                                                        \
      TensorList tensors1,                                                     \
      TensorList tensors2,                                                     \
      at::ArrayRef<Scalar> scalars) {                                          \
    at::native::check_foreach_api_restrictions(                                \
        input, tensors1, tensors2, scalars);                                   \
                                                                               \
    if (!at::native::can_use_fast_route(                                       \
            {input, tensors1, tensors2}, scalars) ||                           \
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
      at::ArrayRef<Scalar> scalars) {                                          \
    at::native::check_foreach_api_restrictions(                                \
        input, tensors1, tensors2, scalars);                                   \
                                                                               \
    if (!at::native::can_use_fast_route(                                       \
            {input, tensors1, tensors2}, scalars) ||                           \
        at::native::has_integral_tensor(input, /* includeBool */ true)) {      \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow_(             \
          input, tensors1, tensors2, scalars);                                 \
    }                                                                          \
                                                                               \
    native::xpu::foreach_##NAME##_kernel_(input, tensors1, tensors2, scalars); \
  }

FOREACH_POINTWISE_OP_SCALARLIST(addcmul)
FOREACH_POINTWISE_OP_SCALARLIST(addcdiv)

} // namespace at
