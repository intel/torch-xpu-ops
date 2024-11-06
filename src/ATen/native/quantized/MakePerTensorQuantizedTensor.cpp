#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_empty_per_channel_affine_quantized.h>
#endif

#include <ATen/native/quantized/sycl/MakePerTensorQuantizedTensorKernels.h>

namespace at {
namespace native {

Tensor make_per_tensor_quantized_tensor_xpu(
    const Tensor& self,
    double scale,
    int64_t zero_point) {
  Tensor dst = at::_empty_affine_quantized(
      self.sizes(),
      self.options().dtype(toQIntType(self.scalar_type())),
      scale,
      zero_point);
  xpu::assign_quantized_tensor_kernel(self, dst);
  return dst;
}

Tensor make_per_channel_quantized_tensor_xpu(
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  Tensor dst = at::_empty_per_channel_affine_quantized(
      self.sizes(),
      scales,
      zero_points,
      axis,
      self.options().dtype(toQIntType(self.scalar_type())));
  xpu::assign_quantized_tensor_kernel(self, dst);
  return dst;
}

} // namespace native
} // namespace at
