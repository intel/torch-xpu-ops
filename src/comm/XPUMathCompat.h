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

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <sycl/sycl.hpp>

#define __MATH_FUNCTIONS_DECL__ static inline

namespace c10::xpu::compat {

__MATH_FUNCTIONS_DECL__ float abs(float x) {
  return sycl::fabs(x);
}
__MATH_FUNCTIONS_DECL__ double abs(double x) {
  return sycl::fabs(x);
}

__MATH_FUNCTIONS_DECL__ float exp(float x) {
  return sycl::exp(x);
}
__MATH_FUNCTIONS_DECL__ double exp(double x) {
  return sycl::exp(x);
}

__MATH_FUNCTIONS_DECL__ float ceil(float x) {
  return sycl::ceil(x);
}
__MATH_FUNCTIONS_DECL__ double ceil(double x) {
  return sycl::ceil(x);
}

__MATH_FUNCTIONS_DECL__ float copysign(float x, float y) {
  return sycl::copysign(x, y);
}
__MATH_FUNCTIONS_DECL__ double copysign(double x, double y) {
  return sycl::copysign(x, y);
}

__MATH_FUNCTIONS_DECL__ float floor(float x) {
  return sycl::floor(x);
}
__MATH_FUNCTIONS_DECL__ double floor(double x) {
  return sycl::floor(x);
}

__MATH_FUNCTIONS_DECL__ float log(float x) {
  return sycl::log(x);
}
__MATH_FUNCTIONS_DECL__ double log(double x) {
  return sycl::log(x);
}

__MATH_FUNCTIONS_DECL__ float log1p(float x) {
  return sycl::log1p(x);
}
__MATH_FUNCTIONS_DECL__ double log1p(double x) {
  return sycl::log1p(x);
}

__MATH_FUNCTIONS_DECL__ float max(float x, float y) {
  return sycl::fmax(x, y);
}
__MATH_FUNCTIONS_DECL__ double max(double x, double y) {
  return sycl::fmax(x, y);
}

__MATH_FUNCTIONS_DECL__ float min(float x, float y) {
  return sycl::fmin(x, y);
}
__MATH_FUNCTIONS_DECL__ double min(double x, double y) {
  return sycl::fmin(x, y);
}

__MATH_FUNCTIONS_DECL__ float pow(float x, float y) {
  return sycl::pow(x, y);
}
__MATH_FUNCTIONS_DECL__ double pow(double x, double y) {
  return sycl::pow(x, y);
}

__MATH_FUNCTIONS_DECL__ void sincos(float x, float* sptr, float* cptr) {
  *sptr = sycl::sin(x);
  *cptr = sycl::cos(x);
}
__MATH_FUNCTIONS_DECL__ void sincos(double x, double* sptr, double* cptr) {
  *sptr = sycl::sin(x);
  *cptr = sycl::cos(x);
}

__MATH_FUNCTIONS_DECL__ float sqrt(float x) {
  return sycl::sqrt(x);
}
__MATH_FUNCTIONS_DECL__ double sqrt(double x) {
  return sycl::sqrt(x);
}

__MATH_FUNCTIONS_DECL__ float rsqrt(float x) {
  return sycl::rsqrt(x);
}
__MATH_FUNCTIONS_DECL__ double rsqrt(double x) {
  return sycl::rsqrt(x);
}

__MATH_FUNCTIONS_DECL__ float tan(float x) {
  return sycl::tan(x);
}
__MATH_FUNCTIONS_DECL__ double tan(double x) {
  return sycl::tan(x);
}

__MATH_FUNCTIONS_DECL__ float tanh(float x) {
  return sycl::tanh(x);
}
__MATH_FUNCTIONS_DECL__ double tanh(double x) {
  return sycl::tanh(x);
}

__MATH_FUNCTIONS_DECL__ float normcdf(float x) {
  return 0.5f * sycl::erfc(-x * static_cast<float>(M_SQRT1_2));
}
__MATH_FUNCTIONS_DECL__ double normcdf(double x) {
  return 0.5 * sycl::erfc(-x * M_SQRT1_2);
}

// To walk around SYCL compiler optimization on data type promotion.
// c10::Half gets data type promotion in +-*/ operations. See
// c10/util/Half-inl.h. XPU implementation gets worse precision on half div,
// since SYCL compiler optimizes the pattern. To align CPU/CUDA precision,
// we define compat div to suppress the optimization. Same as c10::BFloat16.
template <typename T>
inline T div(const T& a, const T& b) __ubsan_ignore_float_divide_by_zero__ {
  return a / b;
}

template <>
inline c10::Half div<c10::Half>(const c10::Half& a, const c10::Half& b)
    __ubsan_ignore_float_divide_by_zero__ {
  volatile float res = static_cast<float>(a) / static_cast<float>(b);
  return res;
}

template <>
inline c10::BFloat16 div<c10::BFloat16>(
    const c10::BFloat16& a,
    const c10::BFloat16& b) __ubsan_ignore_float_divide_by_zero__ {
  volatile float res = static_cast<float>(a) / static_cast<float>(b);
  return res;
}

} // namespace c10::xpu::compat
