#include <ATen/XPUNativeFunctions.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/xpu/sycl/ForeachUnaryKernels.h>

namespace at {

// given a functor and a "dispatch function", creates the outplace and inplace
// operations
#define FOREACH_UNARY_OP(op_name)                                       \
  std::vector<Tensor> XPUNativeFunctions::_foreach_##op_name(           \
      TensorList tensors) {                                             \
    native::check_foreach_api_restrictions(tensors);                    \
    if (!native::can_use_fast_route(tensors) ||                         \
        native::has_integral_tensor(tensors, /* includeBool */ true)) { \
      return at::native::foreach_tensor_##op_name##_slow(tensors);      \
    }                                                                   \
    return native::xpu::foreach_##op_name##_kernel(tensors);            \
  }                                                                     \
  void XPUNativeFunctions::_foreach_##op_name##_(TensorList tensors) {  \
    native::check_foreach_api_restrictions(tensors);                    \
    if (!native::can_use_fast_route(tensors) ||                         \
        native::has_integral_tensor(tensors, /* includeBool */ true)) { \
      return at::native::foreach_tensor_##op_name##_slow_(tensors);     \
    }                                                                   \
                                                                        \
    native::xpu::foreach_##op_name##_kernel_(tensors);                  \
  }

FOREACH_UNARY_OP(sqrt);

} // namespace at
