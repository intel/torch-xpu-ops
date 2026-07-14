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
#include <numbers>

namespace c10::xpu::compat {

static inline float abs(float x) {
  return sycl::fabs(x);
}
static inline double abs(double x) {
  return sycl::fabs(x);
}

static inline float exp(float x) {
  return sycl::exp(x);
}
static inline double exp(double x) {
  return sycl::exp(x);
}

static inline float ceil(float x) {
  return sycl::ceil(x);
}
static inline double ceil(double x) {
  return sycl::ceil(x);
}

static inline float copysign(float x, float y) {
  return sycl::copysign(x, y);
}
static inline double copysign(double x, double y) {
  return sycl::copysign(x, y);
}

static inline float floor(float x) {
  return sycl::floor(x);
}
static inline double floor(double x) {
  return sycl::floor(x);
}

static inline float log(float x) {
  return sycl::log(x);
}
static inline double log(double x) {
  return sycl::log(x);
}

static inline float log1p(float x) {
  return sycl::log1p(x);
}
static inline double log1p(double x) {
  return sycl::log1p(x);
}

static inline float max(float x, float y) {
  return sycl::fmax(x, y);
}
static inline double max(double x, double y) {
  return sycl::fmax(x, y);
}

static inline float min(float x, float y) {
  return sycl::fmin(x, y);
}
static inline double min(double x, double y) {
  return sycl::fmin(x, y);
}

static inline float pow(float x, float y) {
  return sycl::pow(x, y);
}
static inline double pow(double x, double y) {
  return sycl::pow(x, y);
}

static inline void sincos(float x, float* sptr, float* cptr) {
  *sptr = sycl::sin(x);
  *cptr = sycl::cos(x);
}
static inline void sincos(double x, double* sptr, double* cptr) {
  *sptr = sycl::sin(x);
  *cptr = sycl::cos(x);
}

static inline float sqrt(float x) {
  return sycl::sqrt(x);
}
static inline double sqrt(double x) {
  return sycl::sqrt(x);
}

static inline float rsqrt(float x) {
  return sycl::rsqrt(x);
}
static inline double rsqrt(double x) {
  return sycl::rsqrt(x);
}

static inline float tan(float x) {
  return sycl::tan(x);
}
static inline double tan(double x) {
  return sycl::tan(x);
}

static inline float tanh(float x) {
  return sycl::tanh(x);
}
static inline double tanh(double x) {
  return sycl::tanh(x);
}

static inline float normcdf(float x) {
  return 0.5f * sycl::erfc(-x * static_cast<float>(std::numbers::inv_sqrt2));
}
static inline double normcdf(double x) {
  return 0.5 * sycl::erfc(-x * std::numbers::inv_sqrt2);
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
