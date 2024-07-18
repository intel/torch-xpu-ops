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

template <typename scalar_t>
inline C10_HOST_DEVICE scalar_t div_floor_floating(scalar_t a, scalar_t b)
    __ubsan_ignore_float_divide_by_zero__ {
  if (C10_UNLIKELY(b == 0)) {
    // Divide by zero: return standard IEEE result
    return a / b;
  }

  auto mod = std::fmod(a, b);

  // Compatible part. Suppress compiler optimization of data type conversion.
  volatile auto div = (a - mod) / b;
  if ((mod != 0) && (b < 0) != (mod < 0)) {
    div -= scalar_t(1);
  }

  scalar_t floordiv;
  if (div != 0) {
    floordiv = std::floor(div);
    if (div - floordiv > scalar_t(0.5)) {
      floordiv += scalar_t(1.0);
    }
  } else {
    floordiv = C10_COMPAT_COPYSIGN(scalar_t(0), a / b);
  }
  return floordiv;
}

} // namespace c10::xpu::compat
