/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/ForeachUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_adagrad.h>
#include <ATen/ops/_fused_adagrad_native.h>
#endif

#include <ATen/native/xpu/sycl/FusedAdagradKernels.h>

namespace at {
namespace native {

void _fused_adagrad_kernel_xpu_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList state_sums,
    at::TensorList state_steps,
    const double lr,
    const double lr_decay,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  TORCH_CHECK(
      at::native::check_fast_path_restrictions({params, grads, state_sums}),
      "params, grads, and state_sums must have same dtype, device, and layout");

  const float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  const float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;

  xpu::fused_adagrad_kernel(
      params,
      grads,
      state_sums,
      state_steps,
      lr,
      lr_decay,
      weight_decay,
      eps,
      maximize,
      grad_scale_ptr,
      found_inf_ptr);
}

void _fused_adagrad_kernel_xpu_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList state_sums,
    at::TensorList state_steps,
    const at::Tensor& lr,
    const double lr_decay,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (lr.is_cpu()) {
    _fused_adagrad_kernel_xpu_(
        params,
        grads,
        state_sums,
        state_steps,
        lr.item<double>(),
        lr_decay,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
    return;
  }

  // Manually check devices since we specify no device check in
  // native_functions.yaml
  Device param_device = params[0].device();
  if (grad_scale.has_value()) {
    TORCH_CHECK(
        grad_scale->device() == param_device,
        "grad_scale must be on the same GPU device as the params");
  }
  if (found_inf.has_value()) {
    TORCH_CHECK(
        found_inf->device() == param_device,
        "found_inf must be on the same GPU device as the params");
  }
  TORCH_CHECK(
      lr.device() == param_device,
      "lr must be on the same GPU device as the params");

  TORCH_CHECK(
      at::native::check_fast_path_restrictions({params, grads, state_sums}),
      "params, grads, and state_sums must have same dtype, device, and layout");

  const float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  const float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;

  xpu::fused_adagrad_kernel(
      params,
      grads,
      state_sums,
      state_steps,
      lr,
      lr_decay,
      weight_decay,
      eps,
      maximize,
      grad_scale_ptr,
      found_inf_ptr);
}

} // namespace native
} // namespace at
