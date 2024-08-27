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

template <>
inline c10::complex<float> div<c10::complex<float>>(
    const c10::complex<float>& lhs,
    const c10::complex<float>& rhs) __ubsan_ignore_float_divide_by_zero__ {
  float a = lhs.real();
  float b = lhs.imag();
  float c = rhs.real();
  float d = rhs.imag();

  float real_;
  float imag_;

  auto abs_c = std::abs(c);
  auto abs_d = std::abs(d);

  if (abs_c >= abs_d) {
    if (abs_c == 0.f && abs_d == 0.f) {
      /* divide by zeros should yield a complex inf or nan */
      real_ = a / abs_c;
      imag_ = b / abs_d;
    } else {
      float rat = d / c;
      float scl = 1.0f / (c + d * rat);
      real_ = (a + b * rat) * scl;
      imag_ = (b - a * rat) * scl;
    }
  } else {
    float rat = c / d;
    float scl = 1.0f / (d + c * rat);
    real_ = (a * rat + b) * scl;
    imag_ = (b * rat - a) * scl;
  }

  return c10::complex<float>(real_, imag_);
}

} // namespace c10::xpu::compat
