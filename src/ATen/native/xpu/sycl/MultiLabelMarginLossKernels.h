#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API void multilabel_margin_loss_kernel(
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    Tensor& output,
    Tensor& is_target);

// TORCH_XPU_API void multilabel_margin_loss_backward_kernel(
//     const Tensor& grad_output,
//     const Tensor& input,
//     const Tensor& target,
//     int64_t reduction,
//     const Tensor& is_target,
//     Tensor& grad_input);

} // namespace at::native::xpu
