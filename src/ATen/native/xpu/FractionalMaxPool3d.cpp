#include <ATen/ATen.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <ATen/native/xpu/sycl/FractionalMaxPool3dKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>

namespace at {
std::tuple<Tensor&, Tensor&> XPUNativeFunctions::fractional_max_pool3d_out(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef output_size,
    const Tensor& random_samples,
    Tensor& output,
    Tensor& indices) {
  native::xpu::fractional_max_pool3d_out_kernel(
      output, indices, self, kernel_size, output_size, random_samples);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

Tensor& XPUNativeFunctions::fractional_max_pool3d_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef output_size,
    const Tensor& indices,
    Tensor& grad_input) {
  native::xpu::fractional_max_pool3d_backward_out_kernel(
      grad_input, grad_output, self, kernel_size, output_size, indices);
  return grad_input;
}

Tensor XPUNativeFunctions::fractional_max_pool3d_backward(
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef output_size,
    const Tensor& indices) {
  Tensor grad_input = at::empty({0}, self.options());
  native::xpu::fractional_max_pool3d_backward_out_kernel(
      grad_input, grad_output, self, kernel_size, output_size, indices);
  return grad_input;
}

} // namespace at