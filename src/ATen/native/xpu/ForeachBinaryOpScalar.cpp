#include <ATen/native/ForeachUtils.h>

#include <ATen/native/xpu/sycl/ForeachBinaryOpScalarKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>

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

} // namespace at
