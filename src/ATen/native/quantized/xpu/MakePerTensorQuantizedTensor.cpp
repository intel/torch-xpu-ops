#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>

#include <ATen/xpu/XPUNativeFunctions.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_empty_per_channel_affine_quantized.h>
#endif

#include <ATen/native/quantized/xpu/sycl/MakePerTensorQuantizedTensorKernel.h>

namespace at {

Tensor XPUNativeFunctions::_make_per_tensor_quantized_tensor(
    const Tensor& self,
    double scale,
    int64_t zero_point) {
  Tensor dst = at::_empty_affine_quantized(
      self.sizes(),
      self.options().dtype(toQIntType(self.scalar_type())),
      scale,
      zero_point);

  auto iter = TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(dst)
                  .add_input(self)
                  .build();
  native::xpu::assign_quantized_tensor_kernel(iter);
  return dst;
}

Tensor XPUNativeFunctions::_make_per_channel_quantized_tensor(
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

  auto iter = TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(dst)
                  .add_input(self)
                  .build();
  native::xpu::assign_quantized_tensor_kernel(iter);
  return dst;
}

} // namespace at
