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
#include <ATen/OpMathType.h>
#include <ATen/native/ForeachUtils.h>

#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/FusedAdagradKernels.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>

#include <comm/SYCLHelpers.h>

namespace at::native::xpu {

constexpr uint8_t kParamIdx = 0;
constexpr uint8_t kGradIdx = 1;
constexpr uint8_t kStateSumIdx = 2;

// ---------------------------------------------------------------------------
// Adagrad math: operates on kILP elements at a time, matching SGD/Adam style
//
// Math (matches CPU/CUDA):
//   clr = lr / (1 + (step - 1) * lr_decay)
//   if grad_scale:   grad /= grad_scale
//   if maximize:     grad = -grad
//   if weight_decay: grad += param * weight_decay
//   state_sum += grad * grad
//   param -= clr * grad / (sqrt(state_sum) + eps)
// ---------------------------------------------------------------------------
template <typename scalar_t, typename opmath_t>
void adagrad_math(
    scalar_t r_args[3][kILP],
    const double clr,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const float* grad_scale_ptr) {
#pragma unroll
  for (int ii = 0; ii < kILP; ii++) {
    opmath_t param = static_cast<opmath_t>(r_args[kParamIdx][ii]);
    opmath_t grad = static_cast<opmath_t>(r_args[kGradIdx][ii]);
    opmath_t state_sum = static_cast<opmath_t>(r_args[kStateSumIdx][ii]);

    if (grad_scale_ptr) {
      grad /= static_cast<opmath_t>(*grad_scale_ptr);
    }
    const opmath_t grad_to_store = grad;
    if (maximize) {
      grad = -grad;
    }
    if (weight_decay != 0.0) {
      grad += param * static_cast<opmath_t>(weight_decay);
    }

    state_sum += grad * grad;
    param -= static_cast<opmath_t>(clr) * grad /
        (sycl::sqrt(state_sum) + static_cast<opmath_t>(eps));

    r_args[kParamIdx][ii] = static_cast<scalar_t>(param);
    if (grad_scale_ptr) {
      r_args[kGradIdx][ii] = static_cast<scalar_t>(grad_to_store);
    }
    r_args[kStateSumIdx][ii] = static_cast<scalar_t>(state_sum);
  }
}

// depth = 3: [param, grad, state_sum]
// state_steps stored separately in TLFusedMetaForAddress::state_steps_addresses
template <typename scalar_t>
struct FusedAdagradMathFunctor {

  template <typename TLA, typename TLW>
  void operator()(
      const int chunk_size,
      TLA tlAddress,
      TLW tlWGMeta,
      sycl::nd_item<1> item,
      const float* lr_ptr,
      const double lr,
      const double lr_decay,
      const double weight_decay,
      const double eps,
      const bool maximize,
      const float* grad_scale_ptr,
      const float* found_inf_ptr) const {
    if (found_inf_ptr && *found_inf_ptr == 1) {
      return;
    }

    auto workgroup_id = item.get_group(0);
    auto item_id = item.get_local_id(0);
    auto local_range = item.get_local_range(0);

    const auto tensor_loc = tlWGMeta[workgroup_id].wg_to_tensor;
    const auto chunk_idx = tlWGMeta[workgroup_id].wg_to_chunk;

    // Read step from state_steps (float scalar stored per-tensor)
    const auto* step_ptr = reinterpret_cast<const float*>(
        tlAddress[tensor_loc].state_steps_addresses);
    const double step = static_cast<double>(*step_ptr);
    const double lr_val = lr_ptr ? static_cast<double>(*lr_ptr) : lr;
    using opmath_t = at::opmath_type<scalar_t>;
    const double clr = lr_val / (1.0 + (step - 1.0) * lr_decay);

    scalar_t* args[3];
    scalar_t r_args[3][kILP];
    const auto all_aligned{
        init_args<3>(args, tlAddress, chunk_idx, chunk_size, tensor_loc)};
    const auto n =
        tlAddress[tensor_loc].numel_to_tensor - chunk_idx * chunk_size;

    if ((n % kILP == 0) && (chunk_size % kILP == 0) && all_aligned) {
      for (auto i_start = item_id;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += local_range) {
#pragma unroll
        for (int i = 0; i < 3; i++) {
          load_store(r_args[i], args[i], 0, i_start);
        }
        adagrad_math<scalar_t, opmath_t>(
            r_args, clr, weight_decay, eps, maximize, grad_scale_ptr);
        // store param
        load_store(args[kParamIdx], r_args[kParamIdx], i_start, 0);
        // store grad (only when grad_scale applied)
        if (grad_scale_ptr) {
          load_store(args[kGradIdx], r_args[kGradIdx], i_start, 0);
        }
        // store state_sum
        load_store(args[kStateSumIdx], r_args[kStateSumIdx], i_start, 0);
      }
    } else {
      for (auto i_start = 0; i_start < n && i_start < chunk_size;
           i_start += local_range * kILP) {
        load_args<3>(
            r_args, args, i_start, chunk_size, n, item_id, local_range);
        adagrad_math<scalar_t, opmath_t>(
            r_args, clr, weight_decay, eps, maximize, grad_scale_ptr);
        store_args(
            args[kParamIdx], r_args[kParamIdx], i_start, chunk_size, n, item_id, local_range);
        if (grad_scale_ptr) {
          store_args(
              args[kGradIdx], r_args[kGradIdx], i_start, chunk_size, n, item_id, local_range);
        }
        store_args(
            args[kStateSumIdx], r_args[kStateSumIdx], i_start, chunk_size, n, item_id, local_range);
      }
    }
  }
};

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
    const float* grad_scale_ptr,
    const float* found_inf_ptr) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      params.vec(), grads.vec(), state_sums.vec()};

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      params[0].scalar_type(),
      "fused_adagrad_kernel_xpu",
      [&]() {
        const float* lr_ptr_null = nullptr;
        multi_tensor_apply_for_fused_optimizer<3>(
            tensor_lists,
            state_steps,
            FusedAdagradMathFunctor<scalar_t>(),
            lr_ptr_null,
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
    const float* grad_scale_ptr,
    const float* found_inf_ptr) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      params.vec(), grads.vec(), state_sums.vec()};
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
            /*lr=*/1.0,
            lr_decay,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr);
      });
}

} // namespace at::native::xpu
