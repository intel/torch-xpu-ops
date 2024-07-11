#include <ATen/ATen.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/AdaptiveMaxPooling2dKernels.h>
#include <comm/RegisterUtils.h>

namespace at {

void adaptive_max_pool2d_meta(
    const Tensor& input,
    IntArrayRef output_size,
    Tensor& output,
    Tensor& indices) {
  int ndim = input.ndimension();
  TORCH_CHECK(
      ndim == 3 || ndim == 4,
      "adaptive_max_pool2d(): Expected 3D or 4D tensor, but got: ",
      input.sizes());
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_max_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being empty");
  }

  TORCH_CHECK(
      output_size.size() == 2,
      "adaptive_max_pool2d(): internal error: output_size.size() must be 2");

  int dimH = 1;
  int64_t sizeB = 1;
  int64_t sizeD = 0;

  if (input.ndimension() == 4) {
    sizeB = input.size(0);
    dimH++;
  }

  sizeD = input.size(dimH - 1);

  int64_t osizeH = output_size[0];
  int64_t osizeW = output_size[1];

  /* resize output */
  if (input.ndimension() == 3) {
    if (output.defined()) {
      at::xpu::resize_out(output, {sizeD, osizeH, osizeW}, {}, input.options());
    } else {
      output =
          at::xpu::create_out({sizeD, osizeH, osizeW}, {}, input.options());
    }
    if (indices.defined()) {
      at::xpu::resize_out(
          indices, {sizeD, osizeH, osizeW}, {}, input.options());
    } else {
      indices = at::xpu::create_out(
          {sizeD, osizeH, osizeW}, {}, input.options().dtype(kLong));
    }
  } else {
    if (output.defined()) {
      at::xpu::resize_out(
          output,
          {sizeB, sizeD, osizeH, osizeW},
          {},
          input.options().memory_format(input.suggest_memory_format()));
    } else {
      output = at::xpu::create_out(
          {sizeB, sizeD, osizeH, osizeW},
          {},
          input.options().memory_format(input.suggest_memory_format()));
    }
    if (indices.defined()) {
      at::xpu::resize_out(
          indices,
          {sizeB, sizeD, osizeH, osizeW},
          {},
          input.options()
              .memory_format(input.suggest_memory_format())
              .dtype(kLong));
    } else {
      indices = at::xpu::create_out(
          {sizeB, sizeD, osizeH, osizeW},
          {},
          input.options()
              .memory_format(input.suggest_memory_format())
              .dtype(kLong));
    }
  }
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::adaptive_max_pool2d(
    const Tensor& input,
    IntArrayRef output_size) {
  TensorArg input_arg{input, "input", 1};
  checkAllSameGPU(__func__, {input_arg});

  Tensor output, indices;
  adaptive_max_pool2d_meta(input, output_size, output, indices);

  if (input.numel() == 0) {
    return {output, indices};
  }

  native::xpu::adaptive_max_pool2d_kernel(input, output_size, output, indices);
  return {output, indices};
}

std::tuple<Tensor&, Tensor&> XPUNativeFunctions::adaptive_max_pool2d_out(
    const Tensor& input,
    IntArrayRef output_size,
    Tensor& output,
    Tensor& indices) {
  TensorArg output_arg{output, "output", 1};
  TensorArg indices_arg{indices, "indices", 2};
  TensorArg input_arg{input, "input", 3};
  checkAllSameGPU(__func__, {output_arg, indices_arg, input_arg});

  adaptive_max_pool2d_meta(input, output_size, output, indices);

  if (input.numel() == 0) {
    return {output, indices};
  }

  native::xpu::adaptive_max_pool2d_kernel(input, output_size, output, indices);
  return {output, indices};
}

void adaptive_max_pool2d_backward_meta(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& indices,
    Tensor& grad_input) {
  int64_t ndim = grad_output.ndimension();
  TORCH_CHECK(
      ndim == 3 || ndim == 4,
      "adaptive_max_pooling2d_backward(): Expected 3D or 4D grad_output, but got: ",
      grad_output.sizes());

  at::native::adaptive_pool_empty_output_check(
      grad_output, "adaptive_max_pool2d_backward");

  TORCH_CHECK(
      input.dtype() == grad_output.dtype(),
      "expected dtype ",
      input.dtype(),
      " for `grad_output` but got dtype ",
      grad_output.dtype());

  if (grad_input.defined()) {
    at::xpu::resize_out(
        grad_input,
        input.sizes(),
        {},
        input.options().memory_format(input.suggest_memory_format()));
  } else {
    grad_input = at::xpu::create_out(
        input.sizes(),
        {},
        input.options().memory_format(input.suggest_memory_format()));
  }
}

Tensor XPUNativeFunctions::adaptive_max_pool2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& indices) {
  TensorArg grad_output_arg{grad_output, "grad_output", 1};
  TensorArg input_arg{input, "input", 2};
  TensorArg indices_arg{indices, "indices", 3};

  checkAllSameGPU(__func__, {grad_output_arg, input_arg, indices_arg});

  Tensor grad_input;
  adaptive_max_pool2d_backward_meta(grad_output, input, indices, grad_input);

  if (grad_output.numel() == 0) {
    return grad_input;
  }

  native::xpu::adaptive_max_pool2d_backward_kernel(
      grad_output, input, indices, grad_input);
  return grad_input;
}

Tensor& XPUNativeFunctions::adaptive_max_pool2d_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& indices,
    Tensor& grad_input) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1};
  TensorArg grad_output_arg{grad_output, "grad_output", 2};
  TensorArg input_arg{input, "input", 3};
  TensorArg indices_arg{indices, "indices", 4};

  checkAllSameGPU(
      __func__, {grad_input_arg, grad_output_arg, input_arg, indices_arg});

  adaptive_max_pool2d_backward_meta(grad_output, input, indices, grad_input);

  if (grad_output.numel() == 0) {
    return grad_input;
  }

  native::xpu::adaptive_max_pool2d_backward_kernel(
      grad_output, input, indices, grad_input);
  return grad_input;
}

} // namespace at
