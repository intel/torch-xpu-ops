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
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>
#include <comm/XPUMathCompat.h>

#include <ATen/native/xpu/sycl/ActivationHardtanhKernels.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
struct HardtanhBackwardFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_t aop = static_cast<opmath_t>(a);
    opmath_t bop = static_cast<opmath_t>(b);
    return (bop <= min_val_) || (bop >= max_val_) ? opmath_t(0) : aop;
  }

  HardtanhBackwardFunctor(scalar_t min_val, scalar_t max_val)
      : min_val_(min_val), max_val_(max_val) {}

 private:
  scalar_t min_val_;
  scalar_t max_val_;
};

void hardtanh_backward_kernel(
    TensorIterator& iter,
    const Scalar& min,
    const Scalar& max) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardtanh_backward_xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto min_val = min.to<opmath_t>();
        auto max_val = max.to<opmath_t>();
        gpu_kernel(iter, HardtanhBackwardFunctor<scalar_t>(min_val, max_val));
      });
}

} // namespace xpu
} // namespace native
} // namespace at
