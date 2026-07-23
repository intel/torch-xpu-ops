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

#pragma once

#include <ATen/AccumulateType.h>
#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <sycl/sycl.hpp>

#include <limits>
#include <numbers>
#include <type_traits>

namespace at::native::xpu {

/*
 * For licensing information, please refer to the cpu implementation located in
 * "ATen/native/Math.h".
 */
template <typename scalar_t, typename pi_t = double>
static inline C10_HOST_DEVICE scalar_t calc_digamma(scalar_t in) {
  // [C++ Standard Reference: Gamma Function]
  // https://en.cppreference.com/w/cpp/numeric/math/tgamma
  using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
  static const pi_t PI_f64 = 3.14159265358979323846;
  const accscalar_t PSI_10 = 2.25175258906672110764;
  const accscalar_t A[] = {
      8.33333333333333333333E-2,
      -2.10927960927960927961E-2,
      7.57575757575757575758E-3,
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2,
  };

  accscalar_t x = static_cast<accscalar_t>(in);
  if (x == accscalar_t(0)) {
    // As per C++ standard for gamma related functions and SciPy,
    // If the argument is ±0, ±∞ is returned
    return static_cast<scalar_t>(
        sycl::copysign(static_cast<accscalar_t>(INFINITY), -x));
  }

  bool x_is_integer = x == sycl::trunc(x);
  accscalar_t result = 0;
  if (x < accscalar_t(0)) {
    if (x_is_integer) {
      // As per C++ standard for gamma related functions and SciPy,
      // If the argument is a negative integer, NaN is returned
      return static_cast<scalar_t>(NAN);
    }
    // Extracts the fractional part of x as r, since tan(pi * r) is more
    // numerically accurate than tan(pi * x). While these operations are
    // mathematically equivalent since both x and r are in radians and tan() has
    // a periodicity of pi, in practice the computation of pi * x is a source of
    // error (when |x| > 1).
    pi_t q, r;
    r = std::modf(static_cast<pi_t>(x), &q);
    result = static_cast<accscalar_t>(-PI_f64 / std::tan(PI_f64 * r));
    x = 1 - x;
  }

  while (x < accscalar_t(10)) {
    result -= 1 / x;
    x += 1;
  }
  if (x == accscalar_t(10)) {
    return static_cast<scalar_t>(result + PSI_10);
  }

  accscalar_t y = 0;
  if (x < accscalar_t(1.0e17)) {
    accscalar_t z = accscalar_t(1) / (x * x);

    accscalar_t polevl_result = 0;
    for (int i = 0; i <= 6; i++) {
      polevl_result = polevl_result * z + A[i];
    }
    y = z * polevl_result;
  }

  return static_cast<scalar_t>(
      sycl::log(x) - (static_cast<accscalar_t>(0.5) / x) - y + result);
}

} // namespace at::native::xpu
