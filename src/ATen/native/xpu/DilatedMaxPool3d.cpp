#include <ATen/core/Tensor.h>
#include <ATen/native/Pool.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/xpu/sycl/DilatedMaxPool3d.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <comm/RegisterUtils.h>

namespace at {

using namespace at::native;

void max_pool3d_with_indices_meta(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    Tensor& output,
    Tensor& indices) {
  if (output.defined()) {
    at::xpu::resize_out(output, {0}, {}, input.options());
  } else {
    output = at::xpu::create_out({0}, {}, input.options());
  }

  /* indices will contain the locations for each output point */
  if (indices.defined()) {
    at::xpu::resize_out(indices, {0}, {}, input.options().dtype(kLong));
  } else {
    indices = at::xpu::create_out({0}, {}, input.options().dtype(kLong));
  }
}

Tensor& max_pool3d_with_indices_backward_meta(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices,
    Tensor& gradInput) {
  if (gradInput.defined()) {
    at::xpu::resize_out(gradInput, input.sizes(), {}, input.options());
  } else {
    gradInput = at::xpu::create_out(input.sizes(), {}, input.options());
  }
  return gradInput;
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::max_pool3d_with_indices(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  Tensor output;
  Tensor indices;
  max_pool3d_with_indices_meta(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      output,
      indices);

  at::native::xpu::max_pool3d_with_indices_kernel(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      output,
      indices);

  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<Tensor&, Tensor&> XPUNativeFunctions::max_pool3d_with_indices_out(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    Tensor& output,
    Tensor& indices) {
  max_pool3d_with_indices_meta(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      output,
      indices);

  at::native::xpu::max_pool3d_with_indices_kernel(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      output,
      indices);

  return std::tuple<Tensor&, Tensor&>(output, indices);
}

Tensor& XPUNativeFunctions::max_pool3d_with_indices_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices,
    Tensor& grad_input) {
  grad_input = max_pool3d_with_indices_backward_meta(
      grad_output,
      self,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      indices,
      grad_input);

  at::native::xpu::max_pool3d_with_indices_backward_kernel(
      grad_input,
      grad_output,
      self,
      indices,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);

  return grad_input;
}

Tensor XPUNativeFunctions::max_pool3d_with_indices_backward(
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices) {
  Tensor grad_input;
  max_pool3d_with_indices_backward_out(
      grad_output,
      self,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      indices,
      grad_input);

  return grad_input;
}

} // namespace at
