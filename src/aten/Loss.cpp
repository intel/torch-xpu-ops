#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <aten/sycl/BinaryMiscOpsKernels.h>
#include <aten/sycl/PointwiseOpsKernels.h>
#include <comm/RegisterUtils.h>

namespace at {

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

} // namespace at
