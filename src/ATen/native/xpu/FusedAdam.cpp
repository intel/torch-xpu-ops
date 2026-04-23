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

#include <cmath>
#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_adam.h>
#include <ATen/ops/_fused_adam_native.h>
#endif

#include <ATen/native/xpu/sycl/FusedAdamKernels.h>

namespace at {
namespace native {

namespace {

bool found_inf_nonzero(const std::optional<at::Tensor>& found_inf) {
  return found_inf.has_value() && found_inf->is_cpu() &&
      found_inf->item<double>() != 0.0;
}

double grad_scale_value(const std::optional<at::Tensor>& grad_scale) {
  return grad_scale.has_value() && grad_scale->is_cpu()
      ? grad_scale->item<double>()
      : 1.0;
}

template <typename LrT>
bool try_mixed_precision_fused_adam_xpu_(
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
  if (params.empty() || found_inf_nonzero(found_inf)) {
    return true;
  }

  if (params.size() != grads.size() || params.size() != exp_avgs.size() ||
      params.size() != exp_avg_sqs.size() ||
      params.size() != state_steps.size()) {
    return false;
  }

  if (amsgrad && params.size() != max_exp_avg_sqs.size()) {
    return false;
  }

  const auto param_dtype = params[0].scalar_type();

  std::vector<at::Tensor> grads_mixed;
  std::vector<at::Tensor> exp_avgs_mixed;
  std::vector<at::Tensor> exp_avg_sqs_mixed;
  std::vector<at::Tensor> max_exp_avg_sqs_mixed;

  grads_mixed.reserve(grads.size());
  exp_avgs_mixed.reserve(exp_avgs.size());
  exp_avg_sqs_mixed.reserve(exp_avg_sqs.size());
  if (amsgrad) {
    max_exp_avg_sqs_mixed.reserve(max_exp_avg_sqs.size());
  }

  for (int64_t i = 0; i < params.size(); ++i) {
    if (!grads[i].defined()) {
      return false;
    }
    grads_mixed.emplace_back(
        grads[i].scalar_type() == param_dtype ? grads[i]
                                              : grads[i].to(param_dtype));
    exp_avgs_mixed.emplace_back(
        exp_avgs[i].scalar_type() == param_dtype ? exp_avgs[i]
                                                 : exp_avgs[i].to(param_dtype));
    exp_avg_sqs_mixed.emplace_back(
        exp_avg_sqs[i].scalar_type() == param_dtype
            ? exp_avg_sqs[i]
            : exp_avg_sqs[i].to(param_dtype));
    if (amsgrad) {
      max_exp_avg_sqs_mixed.emplace_back(
          max_exp_avg_sqs[i].scalar_type() == param_dtype
              ? max_exp_avg_sqs[i]
              : max_exp_avg_sqs[i].to(param_dtype));
    }
  }

  const bool mixed_fast_path = amsgrad
      ? at::native::check_fast_path_restrictions(
            {params,
             grads_mixed,
             exp_avgs_mixed,
             exp_avg_sqs_mixed,
             max_exp_avg_sqs_mixed})
      : at::native::check_fast_path_restrictions(
            {params, grads_mixed, exp_avgs_mixed, exp_avg_sqs_mixed});

  if (!mixed_fast_path) {
    return false;
  }

  if (amsgrad) {
    xpu::fused_adam_amsgrad_kernel(
        params,
        grads_mixed,
        exp_avgs_mixed,
        exp_avg_sqs_mixed,
        max_exp_avg_sqs_mixed,
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
    xpu::fused_adam_kernel(
        params,
        grads_mixed,
        exp_avgs_mixed,
        exp_avg_sqs_mixed,
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

  for (int64_t i = 0; i < params.size(); ++i) {
    if (exp_avgs[i].scalar_type() != param_dtype) {
      exp_avgs[i].copy_(exp_avgs_mixed[i].to(exp_avgs[i].scalar_type()));
    }
    if (exp_avg_sqs[i].scalar_type() != param_dtype) {
      exp_avg_sqs[i].copy_(
          exp_avg_sqs_mixed[i].to(exp_avg_sqs[i].scalar_type()));
    }
    if (amsgrad && max_exp_avg_sqs[i].scalar_type() != param_dtype) {
      max_exp_avg_sqs[i].copy_(
          max_exp_avg_sqs_mixed[i].to(max_exp_avg_sqs[i].scalar_type()));
    }
    if (grad_scale.has_value() && grads[i].scalar_type() != param_dtype) {
      grads[i].copy_(grads_mixed[i].to(grads[i].scalar_type()));
    }
  }

  return true;
}

void fused_adam_fallback_xpu_(
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
  if (params.empty() || found_inf_nonzero(found_inf)) {
    return;
  }

  if (found_inf.has_value() && !found_inf->is_cpu() &&
      found_inf->item<double>() != 0.0) {
    return;
  }

  TORCH_CHECK(
      params.size() == grads.size() && params.size() == exp_avgs.size() &&
          params.size() == exp_avg_sqs.size() &&
          params.size() == state_steps.size(),
      "inconsistent fused Adam tensor list sizes");

  if (amsgrad) {
    TORCH_CHECK(
        params.size() == max_exp_avg_sqs.size(),
        "inconsistent fused Adam max_exp_avg_sqs size");
  }

  const bool grad_scale_on_cpu = grad_scale.has_value() && grad_scale->is_cpu();
  const double grad_scale_v = grad_scale_value(grad_scale);

  for (int64_t i = 0; i < params.size(); ++i) {
    at::Tensor grad = grads[i];
    if (!grad.defined()) {
      continue;
    }

    at::Tensor param = params[i];
    at::Tensor exp_avg = exp_avgs[i];
    at::Tensor exp_avg_sq = exp_avg_sqs[i];

    if (grad_scale.has_value()) {
      grad = grad_scale_on_cpu ? grad.div(grad_scale_v) : grad.div(*grad_scale);
    }
    if (maximize) {
      grad = grad.neg();
    }
    if (weight_decay != 0.0) {
      grad = grad.add(param, weight_decay);
    }

    const auto math_dtype = param.scalar_type();
    at::Tensor grad_for_math =
        grad.scalar_type() == math_dtype ? grad : grad.to(math_dtype);
    at::Tensor exp_avg_math =
        exp_avg.scalar_type() == math_dtype ? exp_avg : exp_avg.to(math_dtype);
    at::Tensor exp_avg_sq_math = exp_avg_sq.scalar_type() == math_dtype
        ? exp_avg_sq
        : exp_avg_sq.to(math_dtype);

    exp_avg_math.mul_(beta1).add_(grad_for_math, 1.0 - beta1);
    exp_avg_sq_math.mul_(beta2).add_(
        grad_for_math.mul(grad_for_math), 1.0 - beta2);

    const double step = state_steps[i].item<double>();
    const double bias_correction1 = 1.0 - std::pow(beta1, step);
    const double bias_correction2_sqrt = std::sqrt(1.0 - std::pow(beta2, step));

    at::Tensor denom =
        exp_avg_sq_math.sqrt().div_(bias_correction2_sqrt).add_(eps);
    if (amsgrad) {
      at::Tensor max_exp_avg_sq = max_exp_avg_sqs[i];
      at::Tensor max_exp_avg_sq_math =
          max_exp_avg_sq.scalar_type() == math_dtype
          ? max_exp_avg_sq
          : max_exp_avg_sq.to(math_dtype);
      max_exp_avg_sq_math.copy_(max_exp_avg_sq_math.maximum(exp_avg_sq_math));
      denom = max_exp_avg_sq_math.sqrt().div_(bias_correction2_sqrt).add_(eps);
      if (max_exp_avg_sq.scalar_type() != math_dtype) {
        max_exp_avg_sq.copy_(
            max_exp_avg_sq_math.to(max_exp_avg_sq.scalar_type()));
      }
    }

    const double step_size = lr / bias_correction1;
    param.addcdiv_(exp_avg_math, denom, -step_size);

    if (exp_avg.scalar_type() != math_dtype) {
      exp_avg.copy_(exp_avg_math.to(exp_avg.scalar_type()));
    }
    if (exp_avg_sq.scalar_type() != math_dtype) {
      exp_avg_sq.copy_(exp_avg_sq_math.to(exp_avg_sq.scalar_type()));
    }
  }
}

} // namespace

void _fused_adam_kernel_xpu_(
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

  const bool mixed_path = !fast_path &&
      try_mixed_precision_fused_adam_xpu_(
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

  if (fast_path) {
    if (amsgrad) {
      xpu::fused_adam_amsgrad_kernel(
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
      xpu::fused_adam_kernel(
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
  } else if (!mixed_path) {
    fused_adam_fallback_xpu_(
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
void _fused_adam_kernel_xpu_(
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
    _fused_adam_kernel_xpu_(
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
  } else {
    // Manually check devices since we specify no device check in
    // native_functions.yaml
    Device param_device = params[0].device();
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
    TORCH_CHECK(
        lr.device() == param_device,
        "lr must be on the same GPU device as the params");

    const bool fast_path = amsgrad
        ? at::native::check_fast_path_restrictions(
              {params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs})
        : at::native::check_fast_path_restrictions(
              {params, grads, exp_avgs, exp_avg_sqs});

    const bool mixed_path = !fast_path &&
        try_mixed_precision_fused_adam_xpu_(
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

    if (fast_path) {
      if (amsgrad) {
        xpu::fused_adam_amsgrad_kernel(
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
        xpu::fused_adam_kernel(
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
    } else if (!mixed_path) {
      fused_adam_fallback_xpu_(
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
    }
  }
}

} // namespace native
} // namespace at
