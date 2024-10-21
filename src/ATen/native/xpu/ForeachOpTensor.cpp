#include <ATen/native/ForeachUtils.h>
#include <ATen/ops/_foreach_div_native.h>
#include <ATen/ops/_foreach_mul_native.h>

#include <ATen/native/xpu/sycl/ForeachBinaryOpScalarKernels.h>
#include <ATen/native/xpu/sycl/ForeachBinaryOpTensorKernels.h>

namespace at {
namespace native {

#define FOREACH_BINARY_OP_TENSOR(NAME, DIV_OP)                             \
  void foreach_tensor_##NAME##_tensor_kernel_xpu_(                         \
      TensorList tensors, const Tensor& scalar) {                          \
    if (scalar.device().type() == DeviceType::CPU) {                       \
      return xpu::FOREACH_BINARY_SCALAR_INPLACE_KERNEL_NAME(NAME)(         \
          tensors, scalar.item());                                         \
    }                                                                      \
    check_foreach_api_restrictions(tensors);                               \
    if (!(can_use_fast_route(ArrayRef<TensorList>{tensors}, {}, DIV_OP) && \
          tensors[0].scalar_type() == scalar.scalar_type())) {             \
      return foreach_tensor_##NAME##_tensor_kernel_slow_(tensors, scalar); \
    }                                                                      \
                                                                           \
    xpu::FOREACH_BINARY_TENSOR_INPLACE_KERNEL_NAME(NAME)(tensors, scalar); \
  }                                                                        \
                                                                           \
  std::vector<Tensor> foreach_tensor_##NAME##_tensor_kernel_xpu(           \
      TensorList tensors, const Tensor& scalar) {                          \
    if (scalar.device().type() == DeviceType::CPU) {                       \
      return xpu::FOREACH_BINARY_SCALAR_KERNEL_NAME(NAME)(                 \
          tensors, scalar.item());                                         \
    }                                                                      \
    check_foreach_api_restrictions(tensors);                               \
    if (!(can_use_fast_route(ArrayRef<TensorList>{tensors}, {}, DIV_OP) && \
          tensors[0].scalar_type() == scalar.scalar_type())) {             \
      return foreach_tensor_##NAME##_tensor_kernel_slow(tensors, scalar);  \
    }                                                                      \
                                                                           \
    return xpu::FOREACH_BINARY_TENSOR_KERNEL_NAME(NAME)(tensors, scalar);  \
  }

FOREACH_BINARY_OP_TENSOR(mul, /*div_op*/ false);
FOREACH_BINARY_OP_TENSOR(div, /*div_op*/ true);

} // namespace native
} // namespace at
