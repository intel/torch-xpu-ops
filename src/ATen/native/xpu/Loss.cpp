#include <ATen/ATen.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/BinaryMiscOpsKernels.h>
#include <ATen/native/xpu/sycl/PointwiseOpsKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <comm/RegisterUtils.h>

namespace at {

static inline at::Tensor apply_loss_reduction(
    const at::Tensor& unreduced,
    int64_t reduction) {
  if (reduction == at::Reduction::Mean) {
    return unreduced.mean();
  } else if (reduction == at::Reduction::Sum) {
    return unreduced.sum();
  }
  return unreduced;
}

Tensor& XPUNativeFunctions::mse_loss_out(
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    Tensor& result) {
  if (reduction != Reduction::None) {
    TORCH_INTERNAL_ASSERT(
        reduction == Reduction::Mean || reduction == Reduction::Sum);
    result.resize_({});
    Tensor loss;
    auto iter = TensorIterator::borrowing_binary_op(loss, input, target);
    native::xpu::mse_kernel(iter);
    if (reduction == Reduction::Mean) {
      at::mean_out(const_cast<Tensor&>(result), iter.output(), IntArrayRef{});
    } else {
      at::sum_out(const_cast<Tensor&>(result), iter.output(), IntArrayRef{});
    }
  } else {
    auto iter = TensorIterator::borrowing_binary_op(result, input, target);
    native::xpu::mse_kernel(iter);
  }
  return result;
}

Tensor XPUNativeFunctions::mse_loss(
    const Tensor& input,
    const Tensor& target,
    int64_t reduction) {
  Tensor result = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  result = XPUNativeFunctions::mse_loss_out(input, target, reduction, result);
  return result;
}

Tensor XPUNativeFunctions::mse_loss_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction) {
  Tensor grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return at::mse_loss_backward_out(
      grad_input, grad_output, input, target, reduction);
}

Tensor& XPUNativeFunctions::mse_loss_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    Tensor& grad_input) {
  auto norm = reduction == Reduction::Mean ? 2. / input.numel() : 2.;
  auto iter = at::TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_const_input(input)
                  .add_const_input(target)
                  .add_const_input(grad_output)
                  .build();
  native::xpu::mse_backward_kernel(iter, norm);
  return grad_input;
}

Tensor& XPUNativeFunctions::smooth_l1_loss_out(
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    double beta,
    Tensor& result) {
  if (reduction != Reduction::None) {
    TORCH_INTERNAL_ASSERT(
        reduction == Reduction::Mean || reduction == Reduction::Sum);
    result.resize_({});
    Tensor loss;
    auto iter = TensorIterator::borrowing_binary_op(loss, input, target);
    native::xpu::smooth_l1_kernel(iter, beta);
    if (reduction == Reduction::Mean) {
      at::mean_out(const_cast<Tensor&>(result), iter.output(), IntArrayRef{});
    } else {
      at::sum_out(const_cast<Tensor&>(result), iter.output(), IntArrayRef{});
    }
  } else {
    auto iter = TensorIterator::borrowing_binary_op(result, input, target);
    native::xpu::smooth_l1_kernel(iter, beta);
  }
  return result;
}

Tensor XPUNativeFunctions::smooth_l1_loss(
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    double beta) {
  Tensor result = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  result = XPUNativeFunctions::smooth_l1_loss_out(
      input, target, reduction, beta, result);
  return result;
}

Tensor& XPUNativeFunctions::smooth_l1_loss_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    double beta,
    Tensor& grad_input) {
  auto norm = reduction == Reduction::Mean ? 1. / input.numel() : 1.;
  auto iter = at::TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_const_input(input)
                  .add_const_input(target)
                  .add_const_input(grad_output)
                  .promote_inputs_to_common_dtype(true)
                  .cast_common_dtype_to_outputs(true)
                  .enforce_safe_casting_to_output(true)
                  .build();
  native::xpu::smooth_l1_backward_kernel(iter, norm, beta);
  return grad_input;
}

Tensor XPUNativeFunctions::huber_loss(
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    double delta) {
  TORCH_CHECK(
      delta > 0, "huber_loss does not support non-positive values for delta.")
  Tensor loss = at::empty_like(input);
  auto iter = TensorIterator::borrowing_binary_op(loss, input, target);
  native::xpu::huber_kernel(iter, delta);
  return apply_loss_reduction(loss, reduction);
}

Tensor& XPUNativeFunctions::huber_loss_out(
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    double delta,
    Tensor& result) {
  TORCH_CHECK(
      delta > 0, "huber_loss does not support non-positive values for delta.")
  auto iter = TensorIterator::borrowing_binary_op(result, input, target);
  native::xpu::huber_kernel(iter, delta);
  if (reduction != Reduction::None) {
    auto reduced = apply_loss_reduction(result, reduction);
    result.resize_({});
    result.copy_(reduced);
  }
  return result;
}

Tensor& XPUNativeFunctions::huber_loss_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    double delta,
    Tensor& grad_input) {
  auto norm = (reduction == Reduction::Mean) ? (1. / input.numel()) : 1.;
  auto iter = at::TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_const_input(input)
                  .add_const_input(target)
                  .add_const_input(grad_output)
                  .build();
  native::xpu::huber_backward_kernel(iter, norm, delta);
  return grad_input;
}

} // namespace at
