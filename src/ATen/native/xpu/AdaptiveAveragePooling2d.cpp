#include <ATen/ATen.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <aten/src/ATen/ops/mean_ops.h>
#endif

#include <ATen/native/xpu/sycl/AdaptiveAveragePooling2dKernels.h>

namespace at {

namespace {

static c10::SymInt _safe_size(c10::SymIntArrayRef sizes, c10::IntArrayRef dim) {
  c10::SymInt size = 1;
  if (sizes.empty()) {
    return 1;
  }
  for (auto d : dim) {
    d = at::maybe_wrap_dim(d, static_cast<int64_t>(sizes.size()));
    size *= sizes[d];
  }
  return size;
}

Tensor unsqueeze_multiple(
    const Tensor& t,
    OptionalIntArrayRef opt_dim,
    size_t n_dims) {
  if (opt_dim.has_value()) {
    IntArrayRef dim = opt_dim.value();
    auto dim_size = dim.size();
    // Optimisation for two common cases
    if (dim_size == 0) {
      return t;
    } else if (dim_size == 1) {
      return t.unsqueeze(dim[0]);
    }
  }
  auto dims_to_unsqueeze = at::dim_list_to_bitset(opt_dim, n_dims);
  Tensor res = t;
  for (const auto i : c10::irange(n_dims)) {
    if (dims_to_unsqueeze[i]) {
      res = res.unsqueeze(static_cast<int64_t>(i));
    }
  }
  return res;
}

Tensor sum_backward(
    const Tensor& grad,
    c10::SymIntArrayRef sizes,
    OptionalIntArrayRef opt_dims,
    bool keepdim) {
  if (!keepdim && !sizes.empty()) {
    if (opt_dims.has_value() && !opt_dims.value().empty()) {
      return unsqueeze_multiple(grad, opt_dims, sizes.size())
          .expand_symint(sizes);
    }
  }
  return grad.expand_symint(sizes);
}

Tensor mean_backward(
    const Tensor& grad,
    c10::SymIntArrayRef shape,
    OptionalIntArrayRef opt_dim,
    c10::SymInt numel,
    bool keepdim,
    const Tensor& input) {
  bool is_all_reduce = !opt_dim.has_value() || opt_dim.value().empty();
  auto n =
      is_all_reduce ? std::move(numel) : _safe_size(shape, opt_dim.value());

  Tensor grad_input_ =
      sum_backward(grad, shape, opt_dim, keepdim) / std::move(n);

  if (input.suggest_memory_format() == at::MemoryFormat::ChannelsLast) {
    grad_input_ = grad_input_.contiguous(input.suggest_memory_format());
  }

  return grad_input_;
}
} // namespace

Tensor XPUNativeFunctions::_adaptive_avg_pool2d_backward(
    const Tensor& grad_output,
    const Tensor& input) {
  TensorArg grad_output_arg{grad_output, "grad_output", 1},
      input_arg{input, "input", 2};

  native::adaptive_pool_empty_output_check(
      grad_output, "adaptive_avg_pool2d_backward");

  checkAllSameGPU(__func__, {grad_output_arg, input_arg});

  TORCH_CHECK(
      (input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  if (grad_output.size(-1) == 1 && grad_output.size(-2) == 1) {
    return mean_backward(
        grad_output,
        input.sym_sizes().vec(),
        {-1, -2},
        input.sym_numel(),
        true,
        input);
  }

  globalContext().alertNotDeterministic("_adaptive_avg_pool2d_backward");

  Tensor grad_input;
  if (input.numel() != 0) {
    native::xpu::adaptive_avg_pool2d_backward_kernel(
        grad_input, grad_output, input);
  } else {
    grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  return grad_input;
}

Tensor& XPUNativeFunctions::adaptive_avg_pool2d_out(
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
    native::xpu::adaptive_avg_pool2d_kernel(output, input, output_size);
  }
  return output;
}

Tensor XPUNativeFunctions::_adaptive_avg_pool2d(
    at::Tensor const& input,
    IntArrayRef output_size) {
  auto output = at::empty({0}, input.options());
  adaptive_avg_pool2d_out(input, output_size, output);
  return output;
}

} // namespace at
