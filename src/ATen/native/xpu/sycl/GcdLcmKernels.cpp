/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/GcdLcmKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct GcdFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return calc_gcd(a, b);
  }
};

void gcd_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "gcd_xpu", [&]() {
    gpu_kernel(iter, GcdFunctor<scalar_t>());
  });
}

template <typename scalar_t>
struct LcmFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    scalar_t g = calc_gcd(a, b);
    return (g == 0) ? 0 : std::abs(a / g * b);
  }
};

void lcm_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "lcm_xpu", [&]() {
    gpu_kernel(iter, LcmFunctor<scalar_t>());
  });
}

} // namespace at::native::xpu
