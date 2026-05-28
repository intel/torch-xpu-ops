/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>

#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>

#include <comm/SYCLHelpers.h>

namespace at::native::xpu {

// index in TensorList for adagrad params
constexpr uint8_t kAdagradParamIdx = 0;
constexpr uint8_t kAdagradGradIdx = 1;
constexpr uint8_t kAdagradStateSumIdx = 2;

template <typename scalar_type, typename opmath_t>
inline void adagrad_math(
    scalar_type r_args[3][kILP],
    const double& corrected_lr,
    const double& weight_decay,
    const double& eps,
    const bool& maximize,
    const float* grad_scale_ptr,
    const float* found_inf_ptr) {
#pragma unroll
  for (int ii = 0; ii < kILP; ii++) {
    opmath_t param = static_cast<opmath_t>(r_args[kAdagradParamIdx][ii]);
    opmath_t grad = static_cast<opmath_t>(r_args[kAdagradGradIdx][ii]);
    opmath_t state_sum = static_cast<opmath_t>(r_args[kAdagradStateSumIdx][ii]);

    if (grad_scale_ptr) {
      grad /= (static_cast<double>(*grad_scale_ptr));
    }
    const opmath_t grad_to_store = grad;
    if (maximize) {
      grad = -grad;
    }
    if (weight_decay != 0) {
      grad += param * weight_decay;
    }
    state_sum += grad * grad;
    param = param - corrected_lr * grad / (std::sqrt(state_sum) + eps);

    r_args[kAdagradParamIdx][ii] = param;
    if (grad_scale_ptr) {
      r_args[kAdagradGradIdx][ii] = grad_to_store;
    }
    r_args[kAdagradStateSumIdx][ii] = state_sum;
  }
}

template <typename scalar_type>
struct FusedAdagradMathFunctor {
  using opmath_t = at::opmath_type<scalar_type>;
  static constexpr int depth = 3;

  template <typename TLA, typename TLW>
  inline void operator()(
      int chunk_size,
      TLA tlAddress,
      TLW tlWGMeta,
      sycl::nd_item<1> item,
      const float* lr_ptr,
      const double& lr,
      const double& lr_decay,
      const double& weight_decay,
      const double& eps,
      const bool& maximize,
      const float* grad_scale_ptr,
      const float* found_inf_ptr) const {
    auto group_id = item.get_group(0);
    auto item_id = item.get_local_id(0);
    auto local_range = item.get_local_range(0);

    const auto tensor_loc = tlWGMeta[group_id].wg_to_tensor;
    const auto chunk_idx = tlWGMeta[group_id].wg_to_chunk;
    const double lr_double = lr_ptr ? *lr_ptr : lr;

    if (found_inf_ptr && *found_inf_ptr == 1) {
      return;
    }

    const auto corrected_lr = [&]() -> double {
      auto* step_count = reinterpret_cast<const float*>(
          tlAddress[tensor_loc].state_steps_addresses);
      const auto denom = 1 + (*step_count - 1) * lr_decay;
      return lr_double / denom;
    }();

    scalar_type* args[depth];
    scalar_type r_args[depth][kILP];
    const auto n =
        tlAddress[tensor_loc].numel_to_tensor - chunk_idx * chunk_size;

    const bool all_aligned{
        init_args<depth>(args, tlAddress, chunk_idx, chunk_size, tensor_loc)};
    if ((n % kILP == 0) && (chunk_size % kILP == 0) && all_aligned) {
      for (int64_t i_start = item_id;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += local_range) {
#pragma unroll
        for (int i = 0; i < depth; i++) {
          load_store(r_args[i], args[i], 0, i_start);
        }
        adagrad_math<scalar_type, opmath_t>(
            r_args,
            corrected_lr,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr);
#pragma unroll
        for (int i = 0; i < depth; i++) {
          if (i != kAdagradGradIdx || grad_scale_ptr) {
            load_store(args[i], r_args[i], i_start, 0);
          }
        }
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += local_range * kILP) {
        load_args<depth>(
            r_args, args, i_start, chunk_size, n, item_id, local_range);
        adagrad_math<scalar_type, opmath_t>(
            r_args,
            corrected_lr,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr);
#pragma unroll
        for (int i = 0; i < depth; i++) {
          if (i != kAdagradGradIdx || grad_scale_ptr) {
            store_args(
                args[i],
                r_args[i],
                i_start,
                chunk_size,
                n,
                item_id,
                local_range);
          }
        }
      }
    }
  }
};

} // namespace at::native::xpu
