#pragma once
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API std::tuple<Tensor, Tensor> ctc_loss_kernel(
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    int64_t BLANK,
    bool zero_infinity);

TORCH_XPU_API Tensor ctc_loss_backward_kernel(
    const Tensor& grad,
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    const Tensor& neg_log_likelihood,
    const Tensor& log_alpha,
    int64_t BLANK,
    bool zero_infinity);

} // namespace at::native::xpu
