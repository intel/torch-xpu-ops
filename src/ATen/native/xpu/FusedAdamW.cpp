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
#include <ATen/native/xpu/FusedAdamMixedPrecisionUtils.h>

#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_adamw.h>
#include <ATen/ops/_fused_adamw_native.h>
#endif

#include <ATen/native/xpu/sycl/FusedAdamWKernels.h>

namespace at {
namespace native {

namespace {

template <typename LrT>
void run_mixed_precision_fused_adamw_xpu_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const LrT& lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (params.empty()) {
    return;
  }

  if (amsgrad) {
    at::native::validate_fused_mixed_precision_dtypes(
        params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, "_fused_adamw");
  } else {
    at::native::validate_fused_mixed_precision_dtypes(
        params, grads, exp_avgs, exp_avg_sqs, "_fused_adamw");
  }

  const auto param_dtype = params[0].scalar_type();

  std::vector<at::Tensor> exp_avgs_fp32;
  std::vector<at::Tensor> exp_avg_sqs_fp32;
  std::vector<at::Tensor> max_exp_avg_sqs_fp32;
  exp_avgs_fp32.reserve(exp_avgs.size());
  exp_avg_sqs_fp32.reserve(exp_avg_sqs.size());
  if (amsgrad) {
    max_exp_avg_sqs_fp32.reserve(max_exp_avg_sqs.size());
  }

  for (const auto i : c10::irange(params.size())) {
    exp_avgs_fp32.emplace_back(exp_avgs[i].to(param_dtype));
    exp_avg_sqs_fp32.emplace_back(exp_avg_sqs[i].to(param_dtype));
    if (amsgrad) {
      max_exp_avg_sqs_fp32.emplace_back(max_exp_avg_sqs[i].to(param_dtype));
    }
  }

  const bool upcast_fast_path = amsgrad
      ? at::native::check_fast_path_restrictions(
            {params,
             grads,
             exp_avgs_fp32,
             exp_avg_sqs_fp32,
             max_exp_avg_sqs_fp32})
      : at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs_fp32, exp_avg_sqs_fp32});
  TORCH_CHECK(
      upcast_fast_path,
      "_fused_adamw: params, grads, and (upcast) states must share device and "
      "layout for the mixed-precision fused path");

  if (amsgrad) {
    xpu::fused_adamw_amsgrad_kernel(
        params,
        grads,
        exp_avgs_fp32,
        exp_avg_sqs_fp32,
        max_exp_avg_sqs_fp32,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  } else {
    xpu::fused_adamw_kernel(
        params,
        grads,
        exp_avgs_fp32,
        exp_avg_sqs_fp32,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  }

  for (const auto i : c10::irange(params.size())) {
    exp_avgs[i].copy_(exp_avgs_fp32[i]);
    exp_avg_sqs[i].copy_(exp_avg_sqs_fp32[i]);
    if (amsgrad) {
      max_exp_avg_sqs[i].copy_(max_exp_avg_sqs_fp32[i]);
    }
  }
}

} // namespace

void _fused_adamw_kernel_xpu_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  const bool fast_path = amsgrad
      ? at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs})
      : at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs});

  if (fast_path) {
    if (amsgrad) {
      xpu::fused_adamw_amsgrad_kernel(
          params,
          grads,
          exp_avgs,
          exp_avg_sqs,
          max_exp_avg_sqs,
          state_steps,
          lr,
          beta1,
          beta2,
          weight_decay,
          eps,
          maximize,
          grad_scale,
          found_inf);
    } else {
      xpu::fused_adamw_kernel(
          params,
          grads,
          exp_avgs,
          exp_avg_sqs,
          state_steps,
          lr,
          beta1,
          beta2,
          weight_decay,
          eps,
          maximize,
          grad_scale,
          found_inf);
    }
  } else {
    run_mixed_precision_fused_adamw_xpu_(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        amsgrad,
        maximize,
        grad_scale,
        found_inf);
  }
}

// overload with tensor lr(single element tensor) input
void _fused_adamw_kernel_xpu_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const Tensor& lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (lr.is_cpu()) {
    _fused_adamw_kernel_xpu_(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr.item<double>(),
        beta1,
        beta2,
        weight_decay,
        eps,
        amsgrad,
        maximize,
        grad_scale,
        found_inf);
    return;
  }

  Device param_device = params[0].device();
  TORCH_CHECK(
      lr.device() == param_device,
      "lr must be on the same GPU device as the params");

  if (grad_scale != std::nullopt) {
    TORCH_CHECK(
        grad_scale->device() == param_device,
        "grad_scale must be on the same GPU device as the params");
  }
  if (found_inf != std::nullopt) {
    TORCH_CHECK(
        found_inf->device() == param_device,
        "found_inf must be on the same GPU device as the params");
  }

  const bool fast_path = amsgrad
      ? at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs})
      : at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs});

  if (fast_path) {
    if (amsgrad) {
      xpu::fused_adamw_amsgrad_kernel(
          params,
          grads,
          exp_avgs,
          exp_avg_sqs,
          max_exp_avg_sqs,
          state_steps,
          lr,
          beta1,
          beta2,
          weight_decay,
          eps,
          maximize,
          grad_scale,
          found_inf);
    } else {
      xpu::fused_adamw_kernel(
          params,
          grads,
          exp_avgs,
          exp_avg_sqs,
          state_steps,
          lr,
          beta1,
          beta2,
          weight_decay,
          eps,
          maximize,
          grad_scale,
          found_inf);
    }
  } else {
    run_mixed_precision_fused_adamw_xpu_(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        amsgrad,
        maximize,
        grad_scale,
        found_inf);
  }
}

} // namespace native
} // namespace at
