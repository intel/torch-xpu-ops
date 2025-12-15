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

#include <ATen/native/xpu/sycl/ActivationLeakyReluKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct LeakyReluFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t operator()(scalar_t a) const {
    opmath_t aop = static_cast<opmath_t>(a);
    return aop > opmath_t(0) ? aop : aop * negval_;
  }
  LeakyReluFunctor(opmath_t negval) : negval_(negval) {}

 private:
  scalar_t negval_;
};

template <typename scalar_t>
struct LeakyReluBackwardFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t operator()(scalar_t a, scalar_t b) const {
    opmath_t aop = static_cast<opmath_t>(a);
    opmath_t bop = static_cast<opmath_t>(b);
    return aop > opmath_t(0) ? bop : bop * negval_;
  }
  LeakyReluBackwardFunctor(opmath_t negval) : negval_(negval) {}

 private:
  scalar_t negval_;
};

void leaky_relu_kernel(TensorIteratorBase& iter, const Scalar& negval_) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "leaky_relu_xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto negval = negval_.to<opmath_t>();
        gpu_kernel(iter, LeakyReluFunctor<scalar_t>(negval));
      });
}

void leaky_relu_backward_kernel(
    TensorIteratorBase& iter,
    const Scalar& negval_) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "leaky_relu_backward_xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto negval = negval_.to<opmath_t>();
        gpu_kernel(iter, LeakyReluBackwardFunctor<scalar_t>(negval));
      });
}

} // namespace at::native::xpu
