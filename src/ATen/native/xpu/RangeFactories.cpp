/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions of this file are derived from PyTorch
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/RangeFactories.h>
#include <ATen/native/RangeUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/RangeFactoriesKernel.h>
#include <comm/xpu_aten.h>
#include <torch/library.h>

#include <ATen/ops/arange_native.h>
#include <ATen/ops/linspace_native.h>
#include <ATen/ops/logspace_native.h>
#include <ATen/ops/range_native.h>

namespace at {

namespace native {
Tensor& arange_xpu_out(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& out) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      out.scalar_type(),
      "arange_xpu_preprocess",
      [&]() {
        int64_t size = compute_arange_size<scalar_t>(start, end, step);
        int64_t numel = out.numel();

        if (numel != size) {
          if (numel > 0) {
            TORCH_WARN(
                "The number of elements in the out tensor of shape ",
                out.sizes(),
                " is ",
                numel,
                " which does not match the computed number of elements ",
                size,
                ". Note that this may occur as a result of rounding error. "
                "The out tensor will be resized to a tensor of shape (",
                size,
                ",).");
          }
          out.resize_({size});
        }
      });

  return xpu::arange_kernel(start, end, step, out);
}

Tensor& range_xpu_out(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& out) {
  arange_check_bounds(start, end, step);
  int64_t size = static_cast<int64_t>(
      ((end.to<double>() - start.to<double>()) / step.to<double>()) + 1);
  if (out.numel() != size) {
    out.resize_({size});
  }

  return at::native::xpu::range_kernel(start, end, step, out);
}

Tensor& linspace_xpu_out(
    const Scalar& start,
    const Scalar& end,
    int64_t steps,
    Tensor& out) {
  return at::native::xpu::linspace_kernel(start, end, steps, out);
}

Tensor& logspace_xpu_out(
    const Scalar& start,
    const Scalar& end,
    int64_t steps,
    double base,
    Tensor& result) {
  return at::native::xpu::logspace_kernel(start, end, steps, base, result);
}

} // namespace native
} // namespace at
