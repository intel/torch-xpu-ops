/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <ATen/native/xpu/sycl/Reduce.h>
#include <ATen/native/xpu/sycl/SharedReduceOps.h>

#include <ATen/native/xpu/sycl/ReduceOpsKernels.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t, typename out_t = scalar_t>
void std_var_template(
    TensorIterator& iter,
    double correction_opt,
    bool take_sqrt) {
  // reducing unrolling factor to 2 for welford kernel
  // This is necessary to lower register usage that leads to register spills.
  using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
  using ops_t =
      WelfordOps<scalar_t, accscalar_t, int32_t, at::xpu::pair<out_t, out_t>>;
  ops_t ops(static_cast<accscalar_t>(correction_opt), take_sqrt);
  gpu_reduce_kernel<scalar_t, out_t, 2>(iter, ops, typename ops_t::acc_t{});
}

void std_var_kernel(
    TensorIterator& iter,
    double correction_opt,
    bool take_sqrt) {
  const auto input_dtype = iter.input_dtype();
  if (input_dtype == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    std_var_template<at::Half, float>(iter, correction_opt, take_sqrt);
  } else if (input_dtype == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    std_var_template<at::BFloat16, float>(iter, correction_opt, take_sqrt);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "std_var_xpu",
        [&]() { std_var_template<scalar_t>(iter, correction_opt, take_sqrt); });
  }
}

template <
    typename scalar_t,
    typename acc_t = scalar_t,
    typename out_t = scalar_t>
void mean_kernel_impl(TensorIterator& iter) {
  //  returns acc_t for all non-complex dtypes and returns T for c10::complex<T>
  using factor_t = typename c10::scalar_value_type<acc_t>::type;
  factor_t factor =
      static_cast<factor_t>(iter.num_output_elements()) / iter.numel();
  gpu_reduce_kernel<scalar_t, out_t>(
      iter, MeanOps<scalar_t, acc_t, factor_t, out_t>{factor});
}

void mean_kernel(TensorIterator& iter) {
  if (iter.dtype() == kHalf) {
    mean_kernel_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    mean_kernel_impl<at::Half, float, float>(iter);
  } else if (iter.dtype() == kBFloat16) {
    mean_kernel_impl<at::BFloat16, float>(iter);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    mean_kernel_impl<at::BFloat16, float, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(
        iter.dtype(), "mean_xpu", [&]() { mean_kernel_impl<scalar_t>(iter); });
  }
}

} // namespace xpu
} // namespace native
} // namespace at
