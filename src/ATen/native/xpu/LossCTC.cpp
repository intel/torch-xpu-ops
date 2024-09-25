#include <ATen/ATen.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/LossCTCKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <comm/RegisterUtils.h>

namespace at {

std::tuple<Tensor, Tensor> XPUNativeFunctions::_ctc_loss(
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    int64_t blank,
    bool zero_infinity) {
  return native::xpu::ctc_loss_kernel(
      log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::_ctc_loss(
    const Tensor& log_probs,
    const Tensor& targets,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    int64_t blank,
    bool zero_infinity) {
  return at::native::ctc_loss_tensor(
      log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
}

Tensor XPUNativeFunctions::_ctc_loss_backward(
    const Tensor& grad,
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    const Tensor& neg_log_likelihood,
    const Tensor& log_alpha,
    int64_t blank,
    bool zero_infinity) {
  return native::xpu::ctc_loss_backward_kernel(
      grad,
      log_probs,
      targets,
      input_lengths,
      target_lengths,
      neg_log_likelihood,
      log_alpha,
      blank,
      zero_infinity);
}

Tensor XPUNativeFunctions::_ctc_loss_backward(
    const Tensor& grad,
    const Tensor& log_probs,
    const Tensor& targets,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    const Tensor& neg_log_likelihood,
    const Tensor& log_alpha,
    int64_t blank,
    bool zero_infinity) {
  return at::native::ctc_loss_backward_tensor(
      grad,
      log_probs,
      targets,
      input_lengths,
      target_lengths,
      neg_log_likelihood,
      log_alpha,
      blank,
      zero_infinity);
}

} // namespace at
