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

} // namespace c10::xpu::compat
