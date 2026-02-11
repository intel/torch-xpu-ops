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
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <ATen/native/xpu/sycl/Reduce.h>
#include <ATen/native/xpu/sycl/SharedReduceOps.h>

#include <ATen/native/xpu/sycl/ReduceOpsKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
void _min_max_values_kernel_xpu_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(
      iter,
      MinMaxOps<scalar_t, scalar_t, int32_t>{},
      at::xpu::pair<scalar_t, scalar_t>(
          at::numeric_limits<scalar_t>::upper_bound(),
          at::numeric_limits<scalar_t>::lower_bound()));
}

// Special handling for non-standard bool values
template <>
void _min_max_values_kernel_xpu_impl<bool>(TensorIterator& iter) {
  _min_max_values_kernel_xpu_impl<uint8_t>(iter);
}

template <typename scalar_t>
void aminmax_kernel_impl(TensorIterator& iter) {
  _min_max_values_kernel_xpu_impl<scalar_t>(iter);
}

void aminmax_allreduce_kernel(TensorIterator& iter) {
  AT_DISPATCH_V2(
      iter.input_dtype(),
      "aminmax_all_xpu",
      AT_WRAP([&]() {
        _min_max_values_kernel_xpu_impl<scalar_t>(iter);
      }),
      AT_EXPAND(AT_ALL_TYPES),
      kBFloat16,
      kHalf,
      kBool,
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

void aminmax_kernel(TensorIterator& iter) {
  AT_DISPATCH_V2(
      iter.input_dtype(),
      "aminmax_xpu",
      AT_WRAP([&]() {
        aminmax_kernel_impl<scalar_t>(iter);
      }),
      AT_EXPAND(AT_ALL_TYPES),
      kBFloat16,
      kHalf,
      kBool,
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

} // namespace at::native::xpu
