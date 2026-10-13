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

#include <cmath>
#include <limits>

namespace at {

namespace native {

namespace {

int64_t compute_complex_arange_size(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step) {
  const auto startc = start.to<c10::complex<double>>();
  const auto endc = end.to<c10::complex<double>>();
  const auto stepc = step.to<c10::complex<double>>();
  const auto distance = endc - startc;

  TORCH_CHECK(stepc.real() != 0 || stepc.imag() != 0, "step must be nonzero");
  TORCH_CHECK(
      std::isfinite(startc.real()) && std::isfinite(startc.imag()) &&
          std::isfinite(endc.real()) && std::isfinite(endc.imag()),
      "unsupported complex range");

  double size;
  if (stepc.real() == 0) {
    TORCH_CHECK(
        distance.real() == 0, "step real part is zero but range is nonzero");
    size = std::ceil(distance.imag() / stepc.imag());
  } else if (stepc.imag() == 0) {
    TORCH_CHECK(
        distance.imag() == 0,
        "step imaginary part is zero but range is nonzero");
    size = std::ceil(distance.real() / stepc.real());
  } else {
    const auto real_size = std::ceil(distance.real() / stepc.real());
    const auto imag_size = std::ceil(distance.imag() / stepc.imag());
    TORCH_CHECK(
        real_size == imag_size,
        "inconsistent number of elements required by the real and "
        "imaginary parts of the complex range");
    size = real_size;
  }

  TORCH_CHECK(
      size >= 0 &&
          size <= static_cast<double>(std::numeric_limits<int64_t>::max()),
      "invalid size, possible overflow?");
  return static_cast<int64_t>(size);
}

} // namespace

Tensor& arange_xpu_out(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& out) {
  if (isComplexType(out.scalar_type())) {
    const int64_t size = compute_complex_arange_size(start, end, step);
    if (out.numel() != size) {
      if (out.numel() > 0) {
        TORCH_WARN(
            "The number of elements in the out tensor of shape ",
            out.sizes(),
            " is ",
            out.numel(),
            " which does not match the computed number of elements ",
            size,
            ". The out tensor will be resized to a shape of (",
            size,
            ",).");
      }
      out.resize_({size});
    }
    const auto endc = start.to<c10::complex<double>>() +
        step.to<c10::complex<double>>() * static_cast<double>(size - 1);
    return xpu::linspace_kernel(start, Scalar(endc), size, out);
  }

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
