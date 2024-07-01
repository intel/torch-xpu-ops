#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Padding.h>
#include <ATen/native/xpu/sycl/ReflectionPadKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <comm/RegisterUtils.h>

namespace at {

void reflection_pad1d_meta(
    Tensor& output,
    const Tensor& input,
    IntArrayRef padding) {
  int64_t dim_plane = 0;
  int64_t dim_w = 1;
  int64_t nbatch = 1;

  if (input.ndimension() == 3) {
    nbatch = input.size(0);
    dim_w++;
    dim_plane++;
  }

  at::native::padding::check_valid_input<1>(input, padding);

  /* sizes */
  auto pad_l = padding[0];
  auto pad_r = padding[1];

  int64_t nplane = input.size(dim_plane);
  int64_t input_w = input.size(dim_w);
  int64_t output_w = input_w + pad_l + pad_r;

  TORCH_CHECK(
      pad_l < input_w && pad_r < input_w,
      "Argument #4: Padding size "
      "should be less than the corresponding input dimension, but got: padding (",
      pad_l,
      ", ",
      pad_r,
      ") at dimension ",
      dim_w,
      " of input ",
      input.sizes());

  TORCH_CHECK(
      output_w >= 1,
      "input (W: ",
      input_w,
      ") is too small. Calculated output W: ",
      output_w);

  if (output.defined()) {
    if (input.ndimension() == 2) {
      xpu::resize_out(output, {nplane, output_w}, {}, input.options());
    } else {
      xpu::resize_out(output, {nbatch, nplane, output_w}, {}, input.options());
    }
  } else {
    if (input.ndimension() == 2) {
      output = xpu::create_out({nplane, output_w}, {}, input.options());
    } else {
      output = xpu::create_out({nbatch, nplane, output_w}, {}, input.options());
    }
  }
}

void reflection_pad1d_backward_meta(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  int64_t dim_w = 1;
  if (input.ndimension() == 3) {
    dim_w++;
  }

  /* sizes */
  auto pad_l = padding[0];
  auto pad_r = padding[1];
  int64_t input_w = input.size(dim_w);
  int64_t output_w = input_w + pad_l + pad_r;

  TORCH_CHECK(
      pad_l < input_w && pad_r < input_w,
      "Argument #4: Padding size "
      "should be less than the corresponding input dimension, but got: padding (",
      pad_l,
      ", ",
      pad_r,
      ") at dimension ",
      dim_w,
      " of input ",
      input.sizes());

  TORCH_CHECK(
      output_w == grad_output.size(dim_w),
      "grad_output width unexpected."
      " Expected: ",
      output_w,
      ", Got: ",
      grad_output.size(dim_w));

  if (grad_input.defined()) {
    xpu::resize_out(grad_input, input.sizes(), {}, input.options());
  } else {
    xpu::create_out(input.sizes(), {}, input.options());
  }
}

void reflection_pad3d_meta(
    Tensor& output,
    const Tensor& input,
    IntArrayRef padding) {
  int64_t pad_left = padding[0];
  int64_t pad_right = padding[1];
  int64_t pad_top = padding[2];
  int64_t pad_bottom = padding[3];
  int64_t pad_front = padding[4];
  int64_t pad_back = padding[5];
  int64_t dim_w = 3;
  int64_t dim_h = 2;
  int64_t dim_d = 1;
  int64_t dim_plane = 0;

  at::native::padding::check_valid_input<3>(input, padding);

  bool batch_mode = (input.dim() == 5);
  if (batch_mode) {
    dim_w++;
    dim_h++;
    dim_d++;
    dim_plane++;
  }

  int64_t nplane = input.size(dim_plane);
  int64_t input_d = input.size(dim_d);
  int64_t input_h = input.size(dim_h);
  int64_t input_w = input.size(dim_w);
  int64_t output_d = input_d + pad_front + pad_back;
  int64_t output_h = input_h + pad_top + pad_bottom;
  int64_t output_w = input_w + pad_left + pad_right;

  TORCH_CHECK(
      pad_left < input_w && pad_right < input_w,
      "Argument #4: Padding size "
      "should be less than the corresponding input dimension, but got: padding (",
      pad_left,
      ", ",
      pad_right,
      ") at dimension ",
      dim_w,
      " of input ",
      input.sizes());
  TORCH_CHECK(
      pad_top < input_h && pad_bottom < input_h,
      "Argument #6: Padding size "
      "should be less than the corresponding input dimension, but got: padding (",
      pad_top,
      ", ",
      pad_bottom,
      ") at dimension ",
      dim_h,
      " of input ",
      input.sizes());
  TORCH_CHECK(
      pad_front < input_d && pad_back < input_d,
      "Argument #8: Padding size "
      "should be less than the corresponding input dimension, but got: padding (",
      pad_front,
      ", ",
      pad_back,
      ") at dimension ",
      dim_d,
      " of input ",
      input.sizes());

  TORCH_CHECK(
      output_w >= 1 || output_h >= 1 || output_d >= 1,
      "input (D: ",
      input_d,
      " H: ",
      input_h,
      ", W: ",
      input_w,
      ") is too small."
      " Calculated output D: ",
      output_d,
      " H: ",
      output_h,
      " W: ",
      output_w);

  if (output.defined()) {
    if (batch_mode) {
      xpu::resize_out(
          output,
          {input.size(0), nplane, output_d, output_h, output_w},
          {},
          input.options());
    } else {
      xpu::resize_out(
          output, {nplane, output_d, output_h, output_w}, {}, input.options());
    }
  } else {
    if (batch_mode) {
      xpu::create_out(
          {input.size(0), nplane, output_d, output_h, output_w},
          {},
          input.options());
    } else {
      xpu::create_out(
          {nplane, output_d, output_h, output_w}, {}, input.options());
    }
  }
}

void reflection_pad3d_backward_meta(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  TORCH_CHECK(padding.size() == 6, "padding size is expected to be 6");
  TORCH_CHECK(input.dim() > 3);
  TORCH_CHECK(grad_output.dim() == input.dim());

  int64_t pad_left = padding[0];
  int64_t pad_right = padding[1];
  int64_t pad_top = padding[2];
  int64_t pad_bottom = padding[3];
  int64_t pad_front = padding[4];
  int64_t pad_back = padding[5];
  int64_t dim_w = 3;
  int64_t dim_h = 2;
  int64_t dim_d = 1;

  if (input.dim() == 5) {
    // batch mode
    dim_w++;
    dim_h++;
    dim_d++;
  }

  int64_t input_d = input.size(dim_d);
  int64_t input_h = input.size(dim_h);
  int64_t input_w = input.size(dim_w);
  int64_t output_d = input_d + pad_front + pad_back;
  int64_t output_h = input_h + pad_top + pad_bottom;
  int64_t output_w = input_w + pad_left + pad_right;

  TORCH_CHECK(
      output_w == grad_output.size(dim_w),
      "grad_output width unexpected."
      " Expected: ",
      output_w,
      ", Got: ",
      grad_output.size(dim_w));
  TORCH_CHECK(
      output_h == grad_output.size(dim_h),
      "grad_output height unexpected."
      " Expected: ",
      output_h,
      ", Got: ",
      grad_output.size(dim_h));
  TORCH_CHECK(
      output_d == grad_output.size(dim_d),
      "grad_output depth unexpected."
      " Expected: ",
      output_d,
      ", Got: ",
      grad_output.size(dim_d));

  if (grad_input.defined()) {
    xpu::resize_out(grad_input, input.sizes(), {}, input.options());
  } else {
    xpu::create_out(input.sizes(), {}, input.options());
  }
}

Tensor XPUNativeFunctions::reflection_pad1d(
    const Tensor& input,
    IntArrayRef padding) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::reflection_pad1d", "input");

  Tensor output;
  reflection_pad1d_meta(output, input, padding);
  native::xpu::reflection_pad1d_kernel(output, input, padding);
  return output;
}

Tensor& XPUNativeFunctions::reflection_pad1d_out(
    const Tensor& input,
    IntArrayRef padding,
    Tensor& output) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, output, "xpu::reflection_pad1d_out", "output");
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::reflection_pad1d_out", "input");

  reflection_pad1d_meta(output, input, padding);
  native::xpu::reflection_pad1d_kernel(output, input, padding);
  return output;
}

Tensor XPUNativeFunctions::reflection_pad1d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "xpu::reflection_pad1d_backward",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu:reflection_pad1d_backward", "input");

  Tensor grad_input;
  reflection_pad1d_backward_meta(grad_input, grad_output, input, padding);
  native::xpu::reflection_pad1d_backward_kernel(
      grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor& XPUNativeFunctions::reflection_pad1d_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    Tensor& grad_input) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device,
      grad_input,
      "xpu::reflection_pad1d_backward_out",
      "grad_input");
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "xpu::reflection_pad1d_backward_out",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::reflection_pad1d_backward_out", "input");

  native::xpu::reflection_pad1d_backward_kernel(
      grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor& XPUNativeFunctions::reflection_pad2d_out(
    const Tensor& input,
    IntArrayRef padding,
    Tensor& output) {
  native::xpu::reflection_pad2d_kernel(output, input, padding);
  return output;
}

Tensor XPUNativeFunctions::reflection_pad2d(
    const Tensor& input,
    IntArrayRef padding) {
  auto output = at::empty({0}, input.options());
  native::xpu::reflection_pad2d_kernel(output, input, padding);
  return output;
}

Tensor& XPUNativeFunctions::reflection_pad2d_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    Tensor& grad_input) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("reflection_pad2d_backward_out_xpu");
  grad_input.resize_as_(input);
  grad_input.zero_();
  native::xpu::reflection_pad2d_backward_kernel(
      grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor XPUNativeFunctions::reflection_pad2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("reflection_pad2d_backward_xpu");
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  native::xpu::reflection_pad2d_backward_kernel(
      grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor XPUNativeFunctions::reflection_pad3d(
    const Tensor& input,
    IntArrayRef padding) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::reflection_pad3d", "input");

  Tensor output;
  reflection_pad3d_meta(output, input, padding);
  native::xpu::reflection_pad3d_kernel(output, input, padding);
  return output;
}

Tensor& XPUNativeFunctions::reflection_pad3d_out(
    const Tensor& input,
    IntArrayRef padding,
    Tensor& output) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, output, "xpu::reflection_pad3d_out", "output");
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::reflection_pad3d_out", "input");

  reflection_pad3d_meta(output, input, padding);
  native::xpu::reflection_pad3d_kernel(output, input, padding);
  return output;
}

Tensor XPUNativeFunctions::reflection_pad3d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    at::IntArrayRef padding) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "xpu::reflection_pad3d_backward",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::reflection_pad3d_backward", "input");

  Tensor grad_input;
  reflection_pad3d_backward_meta(grad_input, grad_output, input, padding);
  native::xpu::reflection_pad3d_backward_kernel(
      grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor& XPUNativeFunctions::reflection_pad3d_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    Tensor& grad_input) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device,
      grad_input,
      "xpu::reflection_pad3d_backward_out",
      "grad_input");
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "xpu::reflection_pad3d_backward_out",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::reflection_pad3d_backward_out", "input");

  reflection_pad3d_backward_meta(grad_input, grad_output, input, padding);
  native::xpu::reflection_pad3d_backward_kernel(
      grad_input, grad_output, input, padding);
  return grad_input;
}

} // namespace at