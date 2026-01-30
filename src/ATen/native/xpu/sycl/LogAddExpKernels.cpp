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
#include <ATen/OpMathType.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <c10/util/MathConstants.h>
#include <c10/util/complex.h>

#include <ATen/native/xpu/sycl/LogAddExpKernels.h>

#include <cmath>
#include <limits>

namespace at::native::xpu {

// min/max for complex numbers based on real part, with proper NaN propagation
template <typename scalar_t, bool min>
c10::complex<scalar_t> _logaddexp_minmax(
    const c10::complex<scalar_t>& x,
    const c10::complex<scalar_t>& y) {
  scalar_t xr = std::real(x);
  scalar_t yr = std::real(y);
  if (at::_isnan(yr) || (at::_isnan(std::imag(y)))) {
    return y;
  } else if (at::_isnan(xr) || (at::_isnan(std::imag(x)))) {
    return x;
  } else if (min) {
    return (xr < yr) ? x : y;
  } else {
    return (xr >= yr) ? x : y;
  }
}

// Fast complex exponential for finite values: exp(x+iy) = exp(x) * (cos(y) + i*sin(y))
template <typename scalar_t>
c10::complex<scalar_t> _fast_build_exp(const c10::complex<scalar_t>& x) {
  auto xreal = std::real(x);
  auto ximag = std::imag(x);
  auto exp_x_abs = std::exp(xreal);
  auto exp_x_real = exp_x_abs * std::cos(ximag);
  auto exp_x_imag = exp_x_abs * std::sin(ximag);
  return {exp_x_real, exp_x_imag};
}

// Fast complex exponential when real part is infinite
template <typename scalar_t>
c10::complex<scalar_t> _fast_build_exp_inf(const c10::complex<scalar_t>& x) {
  auto ximag = std::imag(x);
  constexpr auto exp_x_abs = std::numeric_limits<scalar_t>::infinity();
  if (!std::isfinite(ximag)) {  // Consistent with std::exp behavior
    return {exp_x_abs, std::numeric_limits<scalar_t>::quiet_NaN()};
  }
  auto sin = std::sin(ximag);
  auto cos = std::cos(ximag);
  // Handle exact multiples of pi/2 to avoid inf * 0 = NaN
  auto exp_x_real = (cos == 0) ? (scalar_t)0.0 : exp_x_abs * cos;
  auto exp_x_imag = (sin == 0) ? (scalar_t)0.0 : exp_x_abs * sin;
  return {exp_x_real, exp_x_imag};
}

// Complex logaddexp: log(exp(x) + exp(y)) with numerical stability
// Uses formula: log1p(exp(min - max)) + max to avoid overflow
template <typename scalar_t>
c10::complex<scalar_t> _log_add_exp_helper(
    const c10::complex<scalar_t>& x,
    const c10::complex<scalar_t>& y) {
  c10::complex<scalar_t> min =
      _logaddexp_minmax<scalar_t, /*min=*/true>(x, y);
  c10::complex<scalar_t> max =
      _logaddexp_minmax<scalar_t, /*min=*/false>(x, y);
  scalar_t min_real = std::real(min);
  scalar_t max_real = std::real(max);

  // Handle NaN propagation
  if (at::_isnan(min_real) || at::_isnan(std::imag(min))) {
    return {
        std::numeric_limits<scalar_t>::quiet_NaN(),
        std::numeric_limits<scalar_t>::quiet_NaN()};
  } else if ((!std::isfinite(min_real)) && (min_real == max_real)) {
    // Handle ±inf cases
    if (min_real < 0) {
      return min;
    } else {
      auto exp_min = _fast_build_exp_inf(min);
      auto exp_max = _fast_build_exp_inf(max);
      return std::log1p(exp_min + exp_max - c10::complex<scalar_t>(1, 0));
    }
  } else {
    // Normal case: use numerically stable formula
    auto minmax = min - max;
    c10::complex<scalar_t> exp_minmax;
    if (!std::isfinite(minmax.real())) {
      exp_minmax = minmax.real() < 0 ? c10::complex<scalar_t>{0.0, 0.0}
                                     : _fast_build_exp_inf(minmax);
    } else {
      exp_minmax = _fast_build_exp(minmax);
    }
    return std::log1p(exp_minmax) + max;
  }
}

// Functor for real floating-point types
// Computes: log(exp(a) + exp(b)) = max + log1p(exp(-|a - b|))
template <typename scalar_t>
struct LogAddExpFunctor {
  scalar_t operator()(scalar_t a_, scalar_t b_) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const auto a = static_cast<opmath_t>(a_);
    const auto b = static_cast<opmath_t>(b_);
    if (std::isinf(a) && a == b) {
      return a;
    } else {
      const auto m = std::max(a, b);
      return m + std::log1p(std::exp(-std::abs(a - b)));
    }
  }
};

// Functor specialization for complex types
template <typename T>
struct LogAddExpFunctor<c10::complex<T>> {
  c10::complex<T> operator()(c10::complex<T> a_, c10::complex<T> b_) const {
    using opmath_t = at::opmath_type<c10::complex<T>>;
    const opmath_t a{a_};
    const opmath_t b{b_};
    return _log_add_exp_helper(a, b);
  }
};

void logaddexp_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::BFloat16,
      ScalarType::Half,
      iter.dtype(),
      "logaddexp_xpu",
      [&]() { gpu_kernel(iter, LogAddExpFunctor<scalar_t>()); });
}

template <typename scalar_t>
struct LogAddExp2Functor {
  scalar_t operator()(scalar_t a_, scalar_t b_) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const auto inv_log_2 = static_cast<opmath_t>(1.0 / c10::ln_2<double>);
    const auto a = static_cast<opmath_t>(a_);
    const auto b = static_cast<opmath_t>(b_);
    if (std::isinf(a) && a == b) {
      return a;
    } else {
      const auto m = std::max(a, b);
      return m + std::log1p(std::exp2(-std::abs(a - b))) * inv_log_2;
    }
  }
};

void logaddexp2_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::BFloat16,
      ScalarType::Half,
      iter.dtype(),
      "logaddexp2_xpu",
      [&]() { gpu_kernel(iter, LogAddExp2Functor<scalar_t>()); });
}

} // namespace at::native::xpu
