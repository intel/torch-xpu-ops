/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <cstdint>

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
// yields +0.0 instead of the IEEE-correct -0.0. Operate on the integer
// representation (musl algorithm, matching the bf16/fp16 path in
// c10/util/BFloat16-math.h) so the sign bit survives.
// TODO: track the underlying issue (JIRA to be filed) and revisit this
// workaround once it is resolved.
// Reference: https://git.musl-libc.org/cgit/musl/tree/src/math/nextafter.c
template <typename scalar_t, typename int_repr_t>
static inline scalar_t nextafter_bits(scalar_t from, scalar_t to) {
  constexpr int bits = sizeof(scalar_t) * 8;
  constexpr int_repr_t sign_mask = int_repr_t{1} << (bits - 1);
  union {
    scalar_t f;
    int_repr_t i;
  } ufrom = {from}, uto = {to};

  // NaN check: if either input is NaN, propagate NaN via from + to.
  if (from != from || to != to) {
    return from + to;
  }
  if (ufrom.i == uto.i) {
    return from;
  }

  int_repr_t abs_from = ufrom.i & ~sign_mask;
  int_repr_t abs_to = uto.i & ~sign_mask;
  if (abs_from == 0) {
    if (abs_to == 0) {
      return to;
    }
    ufrom.i = (uto.i & sign_mask) | int_repr_t{1};
    return ufrom.f;
  }

  if (abs_from > abs_to || ((ufrom.i ^ uto.i) & sign_mask)) {
    ufrom.i--;
  } else {
    ufrom.i++;
  }
  return ufrom.f;
}

template <>
struct NextafterFunctor<float> {
  float operator()(float a, float b) const {
    return nextafter_bits<float, uint32_t>(a, b);
  }
};

template <>
struct NextafterFunctor<double> {
  double operator()(double a, double b) const {
    return nextafter_bits<double, uint64_t>(a, b);
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
