#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API Tensor& multi_margin_loss_kernel(
    const Tensor& input,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin,
    const c10::optional<Tensor>& weight,
    int64_t reduction,
    Tensor& out);

TORCH_XPU_API Tensor& multi_margin_loss_backward_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin,
    const c10::optional<Tensor>& weight,
    int64_t reduction,
    Tensor& grad_input);

} // namespace at::native::xpu
