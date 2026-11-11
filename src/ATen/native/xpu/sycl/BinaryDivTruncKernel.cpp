/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/BinaryInternal.h>
#include <ATen/native/xpu/sycl/BinaryKernels.h>
#include <ATen/native/xpu/sycl/Loops.h>

#include <comm/XPUMathCompat.h>

namespace at::native::xpu {

template <typename scalar_t>
struct DivTruncFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    using opmath_t = at::opmath_type<scalar_t>;
    return sycl::trunc(static_cast<opmath_t>(c10::xpu::compat::div(a, b)));
  }
};

// Divides and truncates in opmath_t, mirroring CPU's
// trunc(static_cast<opmath_t>(a) / b). The parameter types also decide how
// opmath_gpu_kernel_with_scalars reads a CPU scalar: the numerator as
// scalar_t (rounded to the input dtype, as on CPU), the divisor as opmath_t
// (kept at full precision, as CPU's original_scalar_value).
template <typename scalar_t, typename opmath_t>
struct DivTruncOpmathFunctor {
  opmath_t operator()(scalar_t a, opmath_t b) const {
    // volatile suppresses the compiler's type-promotion optimization, as in
    // c10::xpu::compat::div.
    volatile opmath_t quotient = static_cast<opmath_t>(a) / b;
    return sycl::trunc(quotient);
  }
};

void div_trunc_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  if (isIntegralType(dtype, /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "div_trunc_xpu", [&]() {
      gpu_kernel_with_scalars(iter, DivFunctor<scalar_t>());
    });
  } else if (
      iter.is_scalar(2) && iter.data_ptr(2) != nullptr &&
      at::isReducedFloatingType(dtype)) {
    // Matches CPU: with a single-element divisor, a 16-bit quotient is
    // computed and truncated in opmath_t. Rounding the quotient (or a scalar
    // divisor) to scalar_t first can carry it across an integer, e.g. bf16
    // 5.90625 / -0.1376953125 = -42.89 -> -43, or 1.0 / bf16(0.1) = 9.99 -> 9.
    AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "div_trunc_xpu", [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      opmath_gpu_kernel_with_scalars<scalar_t>(
          iter, DivTruncOpmathFunctor<scalar_t, opmath_t>());
    });
  } else {
    // No a * reciprocal(b) for a CPU-scalar divisor (unlike div_true): the
    // extra rounding of 1/b can move an exact integer quotient just below it
    // (7.625 / 3.8125 -> 1.99999994), which trunc turns into an off-by-one.
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, dtype, "div_trunc_xpu", [&]() {
          gpu_kernel_with_scalars(iter, DivTruncFunctor<scalar_t>());
        });
  }
}

} // namespace at::native::xpu
