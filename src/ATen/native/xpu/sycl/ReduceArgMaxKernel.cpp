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

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t, typename acc_t = scalar_t>
void argmax_kernel_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, int64_t>(
      iter,
      ArgMaxOps<acc_t>{},
      at::xpu::pair<acc_t, int64_t>(
          at::numeric_limits<acc_t>::lower_bound(), 0));
};

void argmax_kernel(TensorIterator& iter) {
  if (iter.dtype(1) == kHalf) {
    argmax_kernel_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kBFloat16) {
    argmax_kernel_impl<at::BFloat16, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(1), "argmax_xpu", [&]() {
      argmax_kernel_impl<scalar_t>(iter);
    });
  }
}

} // namespace xpu
} // namespace native
} // namespace at
