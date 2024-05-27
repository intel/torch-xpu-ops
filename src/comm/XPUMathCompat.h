#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <sycl/sycl.hpp>

#define __MATH_FUNCTIONS_DECL__ static inline

namespace c10::xpu::compat {

__MATH_FUNCTIONS_DECL__ float exp(float x) {
  return ::expf(x);
}
__MATH_FUNCTIONS_DECL__ double exp(double x) {
  return ::exp(x);
}

__MATH_FUNCTIONS_DECL__ float tanh(float x) {
  return ::tanhf(x);
}
__MATH_FUNCTIONS_DECL__ double tanh(double x) {
  return ::tanh(x);
}

__MATH_FUNCTIONS_DECL__ float rsqrt(float x) {
  return sycl::rsqrt(x);
}
__MATH_FUNCTIONS_DECL__ double rsqrt(double x) {
  return sycl::rsqrt(x);
}

template <typename T>
inline auto div(const T& a, const T& b) {
  return a / b;
}

template <>
inline auto div<c10::Half>(const c10::Half& a, const c10::Half& b)
    __ubsan_ignore_float_divide_by_zero__ {
  // Suppress compiler optimization to get data type promotion.
  volatile float res = static_cast<float>(a) / static_cast<float>(b);
  return res;
}

template <>
inline auto div<c10::BFloat16>(const c10::BFloat16& a, const c10::BFloat16& b)
    __ubsan_ignore_float_divide_by_zero__ {
  // Suppress compiler optimization to get data type promotion.
  volatile float res = static_cast<float>(a) / static_cast<float>(b);
  return res;
}

template <typename T>
inline T div_trunc(const T& a, const T& b) {
  return a == b ? (T)1 : (T)std::trunc(div<T>(a, b));
}

} // namespace c10::xpu::compat
