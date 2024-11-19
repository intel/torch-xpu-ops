#pragma once
#include <ATen/core/Tensor.h>

namespace at::native::xpu {

TORCH_XPU_API void fused_sgd_kernel(
    at::TensorList params,
    at::TensorList grads,
    const double weight_decay,
    const double momentum,
    const float* lr_ptr,
    const double lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const float* grad_scale_ptr,
    const float* found_inf_ptr);

TORCH_XPU_API void fused_sgd_with_momentum_kernel(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const double weight_decay,
    const double momentum,
    const float* lr_ptr,
    const double lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const float* grad_scale_ptr,
    const float* found_inf_ptr);

} // namespace at::native::xpu
