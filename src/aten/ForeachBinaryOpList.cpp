#include <ATen/native/ForeachUtils.h>

#include <ATen/XPUNativeFunctions.h>
#include <aten/sycl/ForeachBinaryOpListKernels.h>

namespace at {

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

} // namespace at
