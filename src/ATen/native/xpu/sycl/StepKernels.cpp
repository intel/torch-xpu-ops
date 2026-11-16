/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <c10/util/BFloat16-math.h>

#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/StepKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct NextafterFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return std::nextafter(a, b);
  }
};

// The device builtin std::nextafter for float/double does not preserve the
// sign of a zero result: stepping from -min_subnormal toward a larger value
// yields +0.0 instead of the IEEE-correct -0.0. Reuse the builtin and fix up
// the zero result with copysign (sign follows `a`, or `b` when a is zero).
// TODO: tracked by https://github.com/intel/torch-xpu-ops/issues/5708;
// revisit this workaround once the underlying issue is resolved.
template <typename scalar_t>
static inline scalar_t nextafter_signed_zero(scalar_t a, scalar_t b) {
  scalar_t result = std::nextafter(a, b);
  if (result == scalar_t(0)) {
    return std::copysign(scalar_t(0), a == scalar_t(0) ? b : a);
  }
  return result;
}

template <>
struct NextafterFunctor<float> {
  float operator()(float a, float b) const {
    return nextafter_signed_zero(a, b);
  }
};

template <>
struct NextafterFunctor<double> {
  double operator()(double a, double b) const {
    return nextafter_signed_zero(a, b);
  }
};

template <typename scalar_t>
struct HeavisideFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a == 0 ? b : static_cast<scalar_t>(a > 0);
  }
};

void nextafter_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.common_dtype(), "nextafter_xpu", [&]() {
        gpu_kernel_with_scalars(iter, NextafterFunctor<scalar_t>());
      });
}

void heaviside_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBool, kBFloat16, iter.dtype(), "heaviside_xpu", [&]() {
        gpu_kernel_with_scalars(iter, HeavisideFunctor<scalar_t>());
      });
}

} // namespace at::native::xpu
