
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/AdaptivePooling.h>

#include <comm/xpu_aten.h>

#include <ATen/ops/mean.h>
#include <ATen/ops/zeros_like.h>
#include <xpu/ATen/ops/_adaptive_avg_pool2d_backward_native.h>
#include <xpu/ATen/ops/_adaptive_avg_pool2d_native.h>

#include <ATen/native/xpu/sycl/AdaptiveAveragePooling2dKernels.h>

namespace at {
namespace native {
Tensor adaptive_avg_pool2d_backward_xpu(
    const Tensor& grad_output_,
    const Tensor& input_) {
  Tensor grad_input;
  if (input_.numel() != 0) {
    Tensor input, grad_output;
    if (input_.ndimension() == 3) {
      input = input_.contiguous();
      grad_output = grad_output_.contiguous();
      grad_input = at::empty_like(input);
    } else {
      auto smf = input_.suggest_memory_format();
      input = input_.contiguous(smf);
      grad_output = grad_output_.contiguous(smf);
      grad_input = at::empty_like(input_, smf);
    }
    xpu::adaptive_avg_pool2d_backward_kernel(grad_input, grad_output, input);
  } else {
    grad_input = at::zeros_like(input_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  return grad_input;
}

Tensor& adaptive_avg_pool2d_out_xpu(
    const Tensor& input,
    IntArrayRef output_size,
    Tensor& output) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});

  TORCH_CHECK(
      output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");
  int64_t ndim = input.dim();
  TORCH_CHECK(
      (ndim == 3 || ndim == 4),
      "adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got ",
      input.sizes());
  for (const auto i : {-2, -1}) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_avg_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i + ndim,
        " being "
        "empty");
  }

  if (output_size[0] == 1 && output_size[1] == 1) {
    if (output.numel() == 0) {
      output = input.mean({-1, -2}, /* keepdim = */ true);
    } else {
      at::mean_out(output, input, {-1, -2}, true, std::nullopt);
    }
    if (input.suggest_memory_format() == at::MemoryFormat::ChannelsLast) {
      // assert ndim == 4, since ndim = 3 doesn't give channels_last
      const auto n = input.sym_size(0);
      const auto c = input.sym_size(1);
      output.as_strided__symint({n, c, 1, 1}, {c, 1, c, c});
    }
  } else {
    xpu::adaptive_avg_pool2d_kernel(output, input, output_size);
  }
  return output;
}

Tensor adaptive_avg_pool2d_xpu(
    at::Tensor const& input,
    IntArrayRef output_size) {
  auto output = at::empty({0}, input.options());
  adaptive_avg_pool2d_out_xpu(input, output_size, output);
  return output;
}

} // namespace native
} // namespace at
