/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once
#include <ATen/OpMathType.h>
#include <ATen/native/Pow.h>
#include <c10/core/Scalar.h>
#include <sycl/sycl.hpp>

namespace at::native::xpu {

namespace {

// As for pow, the following signatures are defined as the device
// function:
//   pow(float, int)
//   pow(double, int)
//   pow(float, float)
//   pow(double, double)
#ifdef _MSC_VER
// Functions for pow
// pow for at::Half
static inline at::Half pow_(at::Half base, at::Half exp) {
  return static_cast<at::Half>(
      sycl::pow(static_cast<float>(base), static_cast<float>(exp)));
}
// pow for at::BFloat16
static inline at::BFloat16 pow_(at::BFloat16 base, at::BFloat16 exp) {
  return static_cast<at::BFloat16>(
      sycl::pow(static_cast<float>(base), static_cast<float>(exp)));
}
// pow (floating, floating/int)
template <typename Base_type, typename Exp_type>
  requires(
      std::is_floating_point_v<Base_type> &&
      (std::is_same_v<Base_type, Exp_type> || std::is_same_v<Exp_type, int>))
static inline Base_type pow_(Base_type base, Exp_type exp) {
  if constexpr (std::is_same_v<Exp_type, int>) {
    return sycl::pown(base, exp);
  } else {
    return sycl::pow(base, exp);
  }
}
// pow (Otherwise)
template <typename Base_type, typename Exp_type>
  requires(
      !std::is_same_v<Base_type, Exp_type> && !std::is_same_v<Exp_type, int>)
static inline Base_type pow_(Base_type base, Exp_type exp) {
  return static_cast<Base_type>(
      sycl::pow(static_cast<double>(base), static_cast<double>(exp)));
}
#else
template <typename Base_type, typename Exp_type>
static inline Base_type pow_(Base_type base, Exp_type exp) {
  // Both base and exp have the same scalar type in all current call paths
  // They are promoted to opmath_t before entering at::native::xpu::pow_.
  // Therefore a single opmath_t derived from Base_type.
  // is sufficient for both operands.
  using opmath_t = at::opmath_type<Base_type>;
  if constexpr (
      std::is_integral<Exp_type>::value && sizeof(Exp_type) <= sizeof(int)) {
    return sycl::pown(static_cast<opmath_t>(base), static_cast<int>(exp));
  } else {
    return sycl::pow(static_cast<opmath_t>(base), static_cast<opmath_t>(exp));
  }
}
#endif

template <typename T>
  requires std::is_integral_v<T>
static inline T pow_(T base, T exp) {
  return at::native::powi(base, exp);
}

template <typename T>
static inline c10::complex<T> pow_(c10::complex<T> base, c10::complex<T> exp) {
  return c10_complex_math::pow(base, exp);
}

} // namespace
} // namespace at::native::xpu
