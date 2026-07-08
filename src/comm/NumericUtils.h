/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// XPU numeric utilities ported from PyTorch's ATen/NumericUtils.h.
// Replaces __CUDACC__/__HIPCC__ paths with SYCL equivalents.
// __SYCL_DEVICE_ONLY__ -> sycl::native::func (fast path)
// otherwise -> std::func

#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#include <cmath>
#include <type_traits>

#if defined(__SYCL_DEVICE_ONLY__)
#include <sycl/sycl.hpp>
#endif

namespace at::xpu {

// _isnan

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T /*val*/) {
  return false;
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::isnan(val);
#else
  return std::isnan(val);
#endif
}

template <typename T, std::enable_if_t<c10::is_complex<T>::value, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::isnan(val.real()) || sycl::isnan(val.imag());
#else
  return std::isnan(val.real()) || std::isnan(val.imag());
#endif
}

template <typename T, std::enable_if_t<std::is_same_v<T, at::Half>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return at::xpu::_isnan(static_cast<float>(val));
}

template <
    typename T,
    std::enable_if_t<std::is_same_v<T, at::BFloat16>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(at::BFloat16 val) {
  return at::xpu::_isnan(static_cast<float>(val));
}

template <
    typename T,
    std::enable_if_t<std::is_same_v<T, at::Float8_e5m2>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return val.isnan();
}

template <
    typename T,
    std::enable_if_t<std::is_same_v<T, at::Float8_e4m3fn>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return val.isnan();
}

template <
    typename T,
    std::enable_if_t<std::is_same_v<T, at::Float8_e5m2fnuz>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return val.isnan();
}

template <
    typename T,
    std::enable_if_t<std::is_same_v<T, at::Float8_e4m3fnuz>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return val.isnan();
}

// _isinf

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
inline C10_HOST_DEVICE bool _isinf(T /*val*/) {
  return false;
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
inline C10_HOST_DEVICE bool _isinf(T val) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::isinf(val);
#else
  return std::isinf(val);
#endif
}

inline C10_HOST_DEVICE bool _isinf(at::Half val) {
  return at::xpu::_isinf(static_cast<float>(val));
}

inline C10_HOST_DEVICE bool _isinf(at::BFloat16 val) {
  return at::xpu::_isinf(static_cast<float>(val));
}

inline C10_HOST_DEVICE bool _isinf(at::Float8_e5m2 val) {
  return val.isinf();
}

inline C10_HOST_DEVICE bool _isinf(at::Float8_e4m3fn val [[maybe_unused]]) {
  return false;
}

inline C10_HOST_DEVICE bool _isinf(at::Float8_e5m2fnuz val [[maybe_unused]]) {
  return false;
}

inline C10_HOST_DEVICE bool _isinf(at::Float8_e4m3fnuz val [[maybe_unused]]) {
  return false;
}

// exp / log / log1p / tan

template <typename T>
C10_HOST_DEVICE inline T exp(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::native::exp(static_cast<float>(x));
#else
  return std::exp(static_cast<float>(x));
#endif
}

template <>
C10_HOST_DEVICE inline double exp<double>(double x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::exp(x);
#else
  return std::exp(x);
#endif
}

template <typename T>
C10_HOST_DEVICE inline T log(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::native::log(static_cast<float>(x));
#else
  return std::log(static_cast<float>(x));
#endif
}

template <>
C10_HOST_DEVICE inline double log<double>(double x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::log(x);
#else
  return std::log(x);
#endif
}

template <typename T>
C10_HOST_DEVICE inline T log1p(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
#if defined(__SYCL_DEVICE_ONLY__)
  // NOTE: sycl::native has no log1p, mirror CUDA behavior.
  return sycl::native::log(1.0f + static_cast<float>(x));
#else
  return std::log1p(static_cast<float>(x));
#endif
}

template <>
C10_HOST_DEVICE inline double log1p<double>(double x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::log1p(x);
#else
  return std::log1p(x);
#endif
}

template <typename T>
C10_HOST_DEVICE inline T tan(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::native::tan(static_cast<float>(x));
#else
  return std::tan(static_cast<float>(x));
#endif
}

template <>
C10_HOST_DEVICE inline double tan<double>(double x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::tan(x);
#else
  return std::tan(x);
#endif
}

} // namespace at::xpu
