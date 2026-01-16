/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <ATen/native/xpu/sycl/Reduce.h>
#include <ATen/native/xpu/sycl/SharedReduceOps.h>

#include <ATen/native/xpu/sycl/ReduceOpsKernels.h>

namespace at::native::xpu {

template <typename scalar_t, typename acc_t = scalar_t>
void argmin_kernel_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, int64_t>(
      iter,
      ArgMinOps<acc_t>{},
      at::xpu::pair<acc_t, int64_t>(
          at::numeric_limits<acc_t>::upper_bound(), 0));
};

void argmin_kernel(TensorIterator& iter) {
  if (iter.dtype(1) == kHalf) {
    argmin_kernel_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kBFloat16) {
    argmin_kernel_impl<at::BFloat16, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(1), "argmin_xpu", [&]() {
      argmin_kernel_impl<scalar_t>(iter);
    });
  }
}

} // namespace at::native::xpu
