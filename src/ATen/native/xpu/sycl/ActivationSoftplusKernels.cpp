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
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/ActivationSoftplusKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct SoftplusFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t operator()(scalar_t a) const {
    opmath_t aop = static_cast<opmath_t>(a);
    return (aop * beta_) > threshold_
        ? aop
        : (std::log1p(std::exp(aop * beta_))) / beta_;
  }

  SoftplusFunctor(opmath_t beta, opmath_t threshold)
      : beta_(beta), threshold_(threshold) {}

 private:
  opmath_t beta_;
  opmath_t threshold_;
};

void softplus_kernel(
    TensorIteratorBase& iter,
    const Scalar& beta_,
    const Scalar& threshold_) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "softplus_xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto beta = beta_.to<opmath_t>();
        auto threshold = threshold_.to<opmath_t>();
        SoftplusFunctor<scalar_t> f(beta, threshold);
        gpu_kernel(iter, f);
      });
}

template <typename scalar_t>
struct SoftplusBackwardFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t operator()(scalar_t a, scalar_t b) const {
    opmath_t aop = static_cast<opmath_t>(a);
    opmath_t bop = static_cast<opmath_t>(b);
    opmath_t z = std::exp(bop * beta_);
    return (bop * beta_) > threshold_ ? aop : aop * z / (z + opmath_t(1.));
  }

  SoftplusBackwardFunctor(opmath_t beta, opmath_t threshold)
      : beta_(beta), threshold_(threshold) {}

 private:
  opmath_t beta_;
  opmath_t threshold_;
};

void softplus_backward_kernel(
    TensorIteratorBase& iter,
    const Scalar& beta_,
    const Scalar& threshold_) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "softplus_backward_xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto beta = beta_.to<opmath_t>();
        auto threshold = threshold_.to<opmath_t>();
        SoftplusBackwardFunctor<scalar_t> f(beta, threshold);
        gpu_kernel(iter, f);
      });
}
} // namespace at::native::xpu
