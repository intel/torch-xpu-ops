/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch_v2.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/SharedReduceOps.h>

#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <ATen/native/xpu/sycl/Reduce.h>

#include <ATen/native/xpu/sycl/ReduceMinValuesKernels.h>

namespace at {
namespace native {
namespace xpu {

template <typename acc_t>
struct MinNanFunctor {
  inline acc_t operator()(acc_t a, acc_t b) const {
    return (at::_isnan(a) || a < b) ? a : b;
  }
};

template <typename scalar_t, typename acc_t = scalar_t>
void min_values_kernel_xpu_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(
      iter,
      func_wrapper<acc_t>(MinNanFunctor<acc_t>()),
      at::numeric_limits<acc_t>::upper_bound());
}

void min_values_kernel(TensorIterator& iter) {
  AT_DISPATCH_V2(
      iter.dtype(),
      "min_values_xpu",
      AT_WRAP([&]() {
        min_values_kernel_xpu_impl<scalar_t>(iter);
      }),
      AT_EXPAND(AT_ALL_TYPES),
      kBFloat16,
      kHalf,
      kBool,
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

void min_kernel(TensorIterator& iter) {
  AT_DISPATCH_V2(
      iter.input_dtype(),
      "min_xpu",
      AT_WRAP([&]() {
        gpu_reduce_kernel<scalar_t, scalar_t, 4, 2>(
            iter,
            MinOps<scalar_t>{},
            at::xpu::pair<scalar_t, int64_t>(
                at::numeric_limits<scalar_t>::upper_bound(), 0));
      }),
      AT_EXPAND(AT_ALL_TYPES),
      kBFloat16,
      kHalf,
      kBool,
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

void min_all_kernel(TensorIterator& iter) {
  AT_DISPATCH_V2(
      iter.input_dtype(),
      "min_all_xpu",
      AT_WRAP([&] {
        min_values_kernel_xpu_impl<scalar_t>(iter);
      }),
      AT_EXPAND(AT_ALL_TYPES),
      kBFloat16,
      kHalf,
      kBool,
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

} // namespace xpu
} // namespace native
} // namespace at
