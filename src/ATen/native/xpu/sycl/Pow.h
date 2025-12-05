/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once
#include <ATen/native/Pow.h>
#include <c10/core/Scalar.h>

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
      std::pow(static_cast<float>(base), static_cast<float>(exp)));
}
// pow for at::BFloat16
static inline at::BFloat16 pow_(at::BFloat16 base, at::BFloat16 exp) {
  return static_cast<at::BFloat16>(
      std::pow(static_cast<float>(base), static_cast<float>(exp)));
}
// pow (floating, floating/int)
template <typename Base_type, typename Exp_type>
static inline typename std::enable_if<
    std::is_floating_point<Base_type>::value &&
        (std::is_same<Base_type, Exp_type>::value ||
         std::is_same<Exp_type, int>::value),
    Base_type>::type
pow_(Base_type base, Exp_type exp) {
  return std::pow(base, exp);
}
// pow (Otherwise)
template <typename Base_type, typename Exp_type>
static inline typename std::enable_if<
    !std::is_same<Base_type, Exp_type>::value &&
        !std::is_same<Exp_type, int>::value,
    Base_type>::type
pow_(Base_type base, Exp_type exp) {
  return static_cast<Base_type>(
      std::pow(static_cast<double>(base), static_cast<double>(exp)));
}
#else
template <typename Base_type, typename Exp_type>
static inline Base_type pow_(Base_type base, Exp_type exp) {
  return std::pow(base, exp);
}
#endif

template <typename T>
static inline std::enable_if_t<std::is_integral<T>::value, T> pow_(
    T base,
    T exp) {
  return at::native::powi(base, exp);
}

template <typename T>
static inline c10::complex<T> pow_(c10::complex<T> base, c10::complex<T> exp) {
  return c10_complex_math::pow(base, exp);
}

} // namespace
} // namespace at::native::xpu
