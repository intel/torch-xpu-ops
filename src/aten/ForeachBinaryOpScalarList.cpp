#include <ATen/native/ForeachUtils.h>

#include <ATen/XPUNativeFunctions.h>
#include <aten/sycl/ForeachBinaryOpScalarListKernels.h>

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

} // namespace at
