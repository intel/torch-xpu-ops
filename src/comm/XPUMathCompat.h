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

// To walk around SYCL compiler optimization on data type promotion.
// c10::Half gets data type promotion in +-*/ operations. See
// c10/util/Half-inl.h. XPU implementation gets worse precision on half div,
// since SYCL compiler optimizes the pattern. To align CPU/CUDA precision,
// we define compat div to suppress the optimization. Same as c10::BFloat16.
template <typename T>
inline T div(const T& a, const T& b) {
  return a / b;
  }
}

template <>
inline c10::Half div<c10::Half>(const c10::Half& a, const c10::Half& b)
    __ubsan_ignore_float_divide_by_zero__ {
  volatile auto res = a / b;
  return res;
}

template <>
inline c10::BFloat16 div<c10::BFloat16>(
    const c10::BFloat16& a,
    const c10::BFloat16& b) __ubsan_ignore_float_divide_by_zero__ {
  volatile auto res = a / b;
  return res;
}

template <typename T>
inline T div_trunc(const T& a, const T& b) {
  // In the SYCL compilation environment, dividing the same numbers does not
  // equal 1.f sometimes. Therefore, additional checks are required.
  return ((a == b) && std::isfinite(a)) ? (T)1 : (T)std::trunc(div<T>(a, b));
}

} // namespace c10::xpu::compat
