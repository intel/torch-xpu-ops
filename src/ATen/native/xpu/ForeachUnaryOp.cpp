#include <ATen/native/ForeachUtils.h>
#include <ATen/native/xpu/sycl/ForeachUnaryKernels.h>

namespace at {
namespace native {
// given a functor and a "dispatch function", creates the outplace and inplace
// operations

::std::vector<at::Tensor> foreach_tensor_sqrt_slow(at::TensorList self);
void foreach_tensor_sqrt_slow_(at::TensorList self);

#define FOREACH_UNARY_OP(op_name)                                          \
  std::vector<Tensor> foreach_tensor_##op_name##_xpu(TensorList tensors) { \
    check_foreach_api_restrictions(tensors);                               \
    if (!can_use_fast_route(tensors) ||                                    \
        has_integral_tensor(tensors, /* includeBool */ true)) {            \
      return foreach_tensor_##op_name##_slow(tensors);                     \
    }                                                                      \
    return xpu::foreach_##op_name##_kernel(tensors);                       \
  }                                                                        \
  void foreach_tensor_##op_name##_xpu_(TensorList tensors) {               \
    check_foreach_api_restrictions(tensors);                               \
    if (!can_use_fast_route(tensors) ||                                    \
        has_integral_tensor(tensors, /* includeBool */ true)) {            \
      return foreach_tensor_##op_name##_slow_(tensors);                    \
    }                                                                      \
                                                                           \
    xpu::foreach_##op_name##_kernel_(tensors);                             \
  }

FOREACH_UNARY_OP(sqrt);
} // namespace native
} // namespace at
