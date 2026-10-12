/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>
#include <comm/SYCLHelpers.h>

#include <ATen/native/xpu/sycl/AmpKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct AmpNonFiniteCheckUnscaleFunctor {
  using opmath_t = at::opmath_type<scalar_t>;

  scalar_t operator()(scalar_t val_in) const {
    auto val = static_cast<opmath_t>(val_in);
    if (sycl::isinf(val) || sycl::isnan(val)) {
      *found_inf_ptr_ = 1.f;
    }
    const auto inv_scale_val = *inv_scale_ptr_;
    return static_cast<scalar_t>(
        inv_scale_val == 1.f ? val : val * inv_scale_val);
  }

  AmpNonFiniteCheckUnscaleFunctor(
      float* found_inf_ptr,
      const float* inv_scale_ptr)
      : found_inf_ptr_(found_inf_ptr), inv_scale_ptr_(inv_scale_ptr) {}

 private:
  float* found_inf_ptr_;
  const float* inv_scale_ptr_;
};

void amp_non_finite_check_and_unscale_kernel(
    Tensor& scaled_grad,
    Tensor& found_inf,
    const Tensor& inv_scale) {
  auto iter = TensorIterator::unary_op(scaled_grad, scaled_grad);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "amp_non_finite_check_and_unscale_xpu",
      [&iter, &found_inf, &inv_scale] {
        auto* found_inf_ptr = found_inf.data_ptr<float>();
        auto* inv_scale_ptr = inv_scale.const_data_ptr<float>();

        AmpNonFiniteCheckUnscaleFunctor<scalar_t> f(
            found_inf_ptr, inv_scale_ptr);
        gpu_kernel(iter, f);
      });
}

template <typename opmath_t>
struct AmpForeachNonFiniteCheckUnscaleFunctor {
  opmath_t operator()(opmath_t val) const {
    if (sycl::isinf(val) || sycl::isnan(val)) {
      *found_inf_ptr_ = 1.f;
    }
    const auto inv_scale_val = *inv_scale_ptr_;
    return static_cast<opmath_t>(
        inv_scale_val == 1.f ? val : val * inv_scale_val);
  }

  AmpForeachNonFiniteCheckUnscaleFunctor(
      float* found_inf_ptr,
      const float* inv_scale_ptr)
      : found_inf_ptr_(found_inf_ptr), inv_scale_ptr_(inv_scale_ptr) {}

 private:
  float* found_inf_ptr_;
  const float* inv_scale_ptr_;
};

void amp_foreach_non_finite_check_and_unscale_kernel(
    std::vector<std::vector<at::Tensor>> scaled_grads,
    Tensor& found_inf,
    const Tensor& inv_scale) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scaled_grads[0][0].scalar_type(),
      "amp_foreach_non_finite_check_and_unscale_xpu",
      [&scaled_grads, &found_inf, &inv_scale] {
        auto* found_inf_ptr = found_inf.data_ptr<float>();
        auto* inv_scale_ptr = inv_scale.const_data_ptr<float>();

        using opmath_t = at::opmath_type<scalar_t>;

        AmpForeachNonFiniteCheckUnscaleFunctor<opmath_t> f(
            found_inf_ptr, inv_scale_ptr);
        multi_tensor_apply<1>(
            scaled_grads,
            UnaryOpFunctor<
                scalar_t,
                /* depth */ 1,
                /* r_args_depth */ 1,
                /* res_arg_index */ 0>(),
            f);
      });
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void amp_update_scale_ff_kernel(
    float* current_scale,
    int* growth_tracker,
    const float* found_inf,
    double growth_factor,
    double backoff_factor,
    int growth_interval) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  // There is only single item/task scheduled.
  if (item.get_global_linear_id() != 0)
    return;

  if (*found_inf) {
    *current_scale *= backoff_factor;
    *growth_tracker = 0;
  } else {
    // Entering this branch means we just carried out a successful step,
    // so growth_tracker is incremented before comparing to growth_interval.
    auto successful = (*growth_tracker) + 1;
    if (successful == growth_interval) {
      auto new_scale = static_cast<float>((*current_scale) * growth_factor);
      if (!std::isinf(new_scale)) {
        *current_scale = new_scale;
      }
      *growth_tracker = 0;
    } else {
      *growth_tracker = successful;
    }
  }
}

Tensor& amp_update_scale_kernel(
    Tensor& current_scale,
    Tensor& growth_tracker,
    const Tensor& found_inf,
    double growth_factor,
    double backoff_factor,
    int64_t growth_interval) {
  sycl_kernel_submit<amp_update_scale_ff_kernel>(
      sycl::range<1>(1),
      sycl::range<1>(1),
      getCurrentSYCLQueue(),
      0,
      current_scale.mutable_data_ptr<float>(),
      growth_tracker.mutable_data_ptr<int>(),
      found_inf.const_data_ptr<float>(),
      growth_factor,
      backoff_factor,
      growth_interval);
  return current_scale;
}

} // namespace at::native::xpu
