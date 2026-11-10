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

#include <c10/util/complex.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

// SYCL libdevice evaluates complex asin/acos/asinh/acosh as
// log(z + sqrt(z * z +- 1)). This cancels for Re(z) < 0 (fixed upstream in
// intel/llvm#23375) and overflows z * z for large |z|. The helpers below
// reflect into Re(z) >= 0 and use log(2z) for large |z|; non-finite inputs are
// left to the library, which handles them correctly.

namespace at::native::xpu {

template <typename T>
inline bool inv_trig_use_log_2z(const c10::complex<T>& z) {
  // Beyond 1/sqrt(eps), log(z + sqrt(z * z +- 1)) == log(2z) in working
  // precision.
  constexpr T threshold = std::is_same_v<T, double> ? T(1e8) : T(1e4);
  return std::max(std::abs(z.real()), std::abs(z.imag())) > threshold;
}

template <typename T>
inline c10::complex<T> inv_trig_log_2z(const c10::complex<T>& z) {
  constexpr T ln_2 = T(0.693147180559945309417232121458176568);
  return c10::complex<T>(
      std::log(std::hypot(z.real(), z.imag())) + ln_2,
      std::atan2(z.imag(), z.real()));
}

template <typename T>
inline bool inv_trig_is_finite(const c10::complex<T>& z) {
  return std::isfinite(z.real()) && std::isfinite(z.imag());
}

template <typename T>
inline c10::complex<T> asinh_complex(const c10::complex<T>& z) {
  if (!inv_trig_is_finite(z)) {
    return std::asinh(z);
  }
  const bool refl = std::signbit(z.real());
  const c10::complex<T> zr = refl ? -z : z;
  const c10::complex<T> w =
      inv_trig_use_log_2z(zr) ? inv_trig_log_2z(zr) : std::asinh(zr);
  return refl ? -w : w;
}

template <typename T>
inline c10::complex<T> asin_complex(const c10::complex<T>& z) {
  // asin(z) = -i * asinh(i * z)
  const c10::complex<T> w =
      asinh_complex(c10::complex<T>(-z.imag(), z.real()));
  return c10::complex<T>(w.imag(), -w.real());
}

// log(zr + sqrt(zr * zr - 1)) for Re(zr) >= 0; real part >= 0.
template <typename T>
inline c10::complex<T> acosh_core(const c10::complex<T>& zr) {
  return inv_trig_use_log_2z(zr) ? inv_trig_log_2z(zr) : std::acosh(zr);
}

template <typename T>
inline c10::complex<T> acosh_complex(const c10::complex<T>& z) {
  if (!inv_trig_is_finite(z)) {
    return std::acosh(z);
  }
  constexpr T pi = T(3.14159265358979323846264338327950288);
  const bool refl = std::signbit(z.real());
  const c10::complex<T> w = acosh_core(refl ? -z : z);
  // acosh(z) = acosh(-z) +- i*pi for Re(z) < 0
  T im = std::abs(w.imag());
  if (refl) {
    im = pi - im;
  }
  return c10::complex<T>(std::abs(w.real()), std::copysign(im, z.imag()));
}

template <typename T>
inline c10::complex<T> acos_complex(const c10::complex<T>& z) {
  if (!inv_trig_is_finite(z)) {
    return std::acos(z);
  }
  constexpr T pi = T(3.14159265358979323846264338327950288);
  const bool refl = std::signbit(z.real());
  const c10::complex<T> w = acosh_core(refl ? -z : z);
  // acos(z) = pi - acos(-z) for Re(z) < 0
  T re = std::abs(w.imag());
  if (refl) {
    re = pi - re;
  }
  const T im =
      std::signbit(z.imag()) ? std::abs(w.real()) : -std::abs(w.real());
  return c10::complex<T>(re, im);
}

} // namespace at::native::xpu
