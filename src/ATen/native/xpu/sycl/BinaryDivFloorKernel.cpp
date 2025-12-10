/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>
#include <c10/util/generic_math.h>

#include <ATen/native/xpu/sycl/BinaryKernels.h>
#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct DivFloorFloatFunctor {
  scalar_t operator()(scalar_t a_, scalar_t b_) const {
    using acc_t = at::acc_type_device<scalar_t, c10::DeviceType::XPU>;
    acc_t a = static_cast<acc_t>(a_);
    acc_t b = static_cast<acc_t>(b_);

    // suppress compiler optimization on data type promotion
    volatile acc_t res = c10::div_floor_floating(a, b);
    return res;
  }
};

template <typename scalar_t>
struct DivFloorIntergerFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return c10::div_floor_integer(a, b);
  }
};

template <typename scalar_t, typename accscalar_t>
struct DivFloorWithScalarFunctor {
  DivFloorWithScalarFunctor(accscalar_t b, accscalar_t inv_b)
      : b_(b), inv_b_(inv_b) {}

  scalar_t operator()(scalar_t a) const {
    auto mod = std::fmod(a, b_);
    auto div = (a - mod) * inv_b_;
    if ((mod != 0) && (b_ < 0) != (mod < 0)) {
      div -= scalar_t(1);
    }

    scalar_t floordiv;
    if (div != 0) {
      floordiv = std::floor(div);
      if (div - floordiv > scalar_t(0.5)) {
        floordiv += scalar_t(1.0);
      }
    } else {
      floordiv = std::copysign(scalar_t(0), a * inv_b_);
    }
    return floordiv;
  }

 private:
  accscalar_t b_;
  accscalar_t inv_b_;
};

void div_floor_kernel(TensorIteratorBase& iter) {
  // See NOTE: [Floor Division in Python]
  const auto dtype = iter.common_dtype();
  if (dtype == kByte) {
    // In the special case of unsigned integer division, floor division is
    // equivalent to truncation division (since the signs of the divisor and
    // dividend are always the same)
    return div_trunc_kernel(iter);
  } else if (isIntegralType(dtype, /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "div_floor_xpu", [&]() {
      gpu_kernel_with_scalars(iter, DivFloorIntergerFunctor<scalar_t>());
    });
  } else if (iter.is_cpu_scalar(2)) {
    // optimization for floating-point types: if the second operand is a CPU
    // scalar, compute a * reciprocal(b). Note that this may lose one bit of
    // precision compared to computing the division.
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, dtype, "div_floor_xpu", [&]() {
          using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
          auto b = iter.scalar_value<accscalar_t>(2);
          if (C10_UNLIKELY(b == 0)) {
            return div_true_kernel(iter);
          }

          auto inv_b = accscalar_t(1.0) / b;
          iter.remove_operand(2);
          gpu_kernel(
              iter, DivFloorWithScalarFunctor<scalar_t, accscalar_t>(b, inv_b));
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, dtype, "div_floor_xpu", [&]() {
          gpu_kernel_with_scalars(iter, DivFloorFloatFunctor<scalar_t>());
        });
  }
}
} // namespace at::native::xpu
