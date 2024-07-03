#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {
void nll_loss2d_forward_out_kernel(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index);

void nll_loss2d_backward_out_kernel(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight);
} // namespace at::native::xpu