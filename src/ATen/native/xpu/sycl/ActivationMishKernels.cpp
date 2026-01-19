/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/Loops.h>
#include <comm/XPUMathCompat.h>

#include <ATen/native/xpu/sycl/ActivationMishKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct MishFunctor {
  scalar_t operator()(scalar_t x) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t x_acc = static_cast<opmath_t>(x);
    return x_acc * std::tanh(std::log1p(std::exp(x_acc)));
  }
};

void mish_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mish_xpu",
      [&]() { gpu_kernel(iter, MishFunctor<scalar_t>()); });
}

template <typename scalar_t>
struct MishBackwardFunctor {
  scalar_t operator()(scalar_t dy, scalar_t x) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t dy_acc = static_cast<opmath_t>(dy);
    const opmath_t x_acc = static_cast<opmath_t>(x);
    const opmath_t s_acc = opmath_t(1) / (opmath_t(1) + std::exp(-x_acc));
    const opmath_t t_acc = std::tanh(std::log1p(std::exp(x_acc)));
    return dy_acc * (t_acc + x_acc * s_acc * (opmath_t(1) - t_acc * t_acc));
  }
};

void mish_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mish_backward_xpu",
      [&]() { gpu_kernel(iter, MishBackwardFunctor<scalar_t>()); });
}

} // namespace at::native::xpu
