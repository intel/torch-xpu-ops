/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/ActivationLogSigmoidKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct LogSigmoidForwardFunctor {
  scalar_t operator()(scalar_t in_) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t in = in_;
    const auto min = std::min(opmath_t(0), in);
    const auto z = std::exp(-std::abs(in));
    return min - std::log1p(z);
  }
};

void log_sigmoid_forward_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "log_sigmoid_forward_xpu",
      [&]() { gpu_kernel(iter, LogSigmoidForwardFunctor<scalar_t>()); });
}

template <typename scalar_t>
struct LogSigmoidBackwardFunctor {
  scalar_t operator()(scalar_t in_, scalar_t grad_out_) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t in = in_;
    const opmath_t grad_out = grad_out_;

    auto in_negative = in < opmath_t(0);
    auto max_deriv = in_negative ? opmath_t(1) : opmath_t(0);
    auto sign = in_negative ? opmath_t(1) : -opmath_t(1);
    const auto z = std::exp(-std::abs(in));
    return grad_out * (max_deriv - sign * (z / (opmath_t(1) + z)));
  }
};

void log_sigmoid_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "log_sigmoid_backward_xpu",
      [&]() { gpu_kernel(iter, LogSigmoidBackwardFunctor<scalar_t>()); });
}

} // namespace at::native::xpu
