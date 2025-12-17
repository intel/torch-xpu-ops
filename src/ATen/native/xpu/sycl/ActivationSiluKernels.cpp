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
#include <ATen/NumericUtils.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>
#include <comm/XPUMathCompat.h>

#include <ATen/native/xpu/sycl/ActivationSiluKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct SiluFunctor {
  scalar_t operator()(scalar_t x) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t x_acc = static_cast<opmath_t>(x);
    return x_acc / (opmath_t(1) + std::exp(-x_acc));
  }
};

template <typename scalar_t>
struct SiluBackwardFunctor {
  scalar_t operator()(scalar_t dy, scalar_t x) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t dy_acc = static_cast<opmath_t>(dy);
    const opmath_t x_acc = static_cast<opmath_t>(x);
    const opmath_t s_acc = opmath_t(1) / (opmath_t(1) + std::exp(-x_acc));
    return dy_acc * s_acc * (opmath_t(1) + x_acc * (opmath_t(1) - s_acc));
  }
};

void silu_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "silu_xpu",
      [&]() { gpu_kernel(iter, SiluFunctor<scalar_t>()); });
}

void silu_backward_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "silu_backward_xpu",
      [&]() { gpu_kernel(iter, SiluBackwardFunctor<scalar_t>()); });
}

} // namespace at::native::xpu
