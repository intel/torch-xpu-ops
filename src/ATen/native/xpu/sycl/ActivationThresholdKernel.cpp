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
#include <ATen/native/Activation.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/ActivationThresholdKernel.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
struct ThresholdFunctor {
  scalar_t operator()(scalar_t x, scalar_t other) const {
    return x <= threshold_ ? value_ : other;
  }

  ThresholdFunctor(scalar_t threshold, scalar_t value)
      : threshold_(threshold), value_(value) {}

 private:
  scalar_t threshold_;
  scalar_t value_;
};

void threshold_kernel(
    TensorIteratorBase& iter,
    const Scalar& threshold,
    const Scalar& value) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf, kBFloat16, iter.dtype(), "threshold_xpu", [&]() {
        scalar_t threshold_ = threshold.to<scalar_t>();
        scalar_t value_ = value.to<scalar_t>();
        gpu_kernel_with_scalars(
            iter, ThresholdFunctor<scalar_t>(threshold_, value_));
      });
}

} // namespace xpu
} // namespace native
} // namespace at
