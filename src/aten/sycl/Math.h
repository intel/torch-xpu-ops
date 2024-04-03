#pragma once

#include <ATen/AccumulateType.h>
#include <ATen/NumericUtils.h>
#include <c10/macros/Macros.h>
#include <c10/util/MathConstants.h>
#include <c10/util/complex.h>

namespace at {
namespace native {

template <typename scalar_t>
static inline scalar_t calc_gcd(scalar_t a_in, scalar_t b_in) {
  scalar_t a = ::abs(a_in);
  scalar_t b = ::abs(b_in);
  while (a != 0) {
    scalar_t c = a;
    a = b % a;
    b = c;
  }
  return b;
}
} // namespace native
} // namespace at