/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>

#include <ATen/native/xpu/sycl/FusedAdagradKernels.h>
#include <ATen/native/xpu/sycl/FusedAdagradUtils.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>

namespace at::native::xpu {

void fused_adagrad_kernel(
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
  std::vector<std::vector<at::Tensor>> tensor_lists{
      params.vec(), grads.vec(), state_sums.vec()};

  const float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  const float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  const float* lr_ptr = nullptr;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      params[0].scalar_type(),
      "fused_adagrad_kernel_xpu",
      [&]() {
        multi_tensor_apply_for_fused_optimizer<3>(
            tensor_lists,
            state_steps,
            FusedAdagradMathFunctor<scalar_t>(),
            lr_ptr, // unused
            lr,
            lr_decay,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr);
      });
}

void fused_adagrad_kernel(
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
  std::vector<std::vector<at::Tensor>> tensor_lists{
      params.vec(), grads.vec(), state_sums.vec()};

  const float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->const_data_ptr<float>() : nullptr;
  const float* found_inf_ptr =
      found_inf.has_value() ? found_inf->const_data_ptr<float>() : nullptr;
  const float* lr_ptr = lr.const_data_ptr<float>();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      params[0].scalar_type(),
      "fused_adagrad_kernel_xpu",
      [&]() {
        multi_tensor_apply_for_fused_optimizer<3>(
            tensor_lists,
            state_steps,
            FusedAdagradMathFunctor<scalar_t>(),
            lr_ptr,
            1.0, // unused when lr_ptr is set
            lr_decay,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr);
      });
}

} // namespace at::native::xpu
