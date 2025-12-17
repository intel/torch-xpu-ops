/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/OpMathType.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/ActivationEluKernels.h>

namespace at::native::xpu {

template <typename scalar_t, typename opmath_t>
struct EluOutFunctor {
  scalar_t operator()(scalar_t a) const {
    opmath_t aop = static_cast<opmath_t>(a);
    return aop > 0 ? aop * poscoef_ : std::expm1(aop * negiptcoef_) * negcoef_;
  }

  EluOutFunctor(opmath_t negcoef, opmath_t poscoef, opmath_t negiptcoef)
      : negcoef_(negcoef), poscoef_(poscoef), negiptcoef_(negiptcoef) {}

 private:
  opmath_t negcoef_;
  opmath_t poscoef_;
  opmath_t negiptcoef_;
};

void elu_kernel(
    TensorIteratorBase& iter,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "elu_xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto negcoef = alpha.to<opmath_t>() * scale.to<opmath_t>();
        auto poscoef = scale.to<opmath_t>();
        auto negiptcoef = input_scale.to<opmath_t>();
        EluOutFunctor<scalar_t, opmath_t> f(negcoef, poscoef, negiptcoef);
        gpu_kernel(iter, f);
      });
}

template <typename scalar_t, typename opmath_t>
struct EluBackwardOutFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    opmath_t aop = static_cast<opmath_t>(a);
    opmath_t bop = static_cast<opmath_t>(b);

    if (is_result_) {
      return bop <= 0 ? aop * negiptcoef_ * (bop + negcoef_) : aop * poscoef_;
    } else {
      return bop <= 0
          ? aop * negiptcoef_ * negcoef_ * std::exp(bop * negiptcoef_)
          : aop * poscoef_;
    }
  }

  EluBackwardOutFunctor(
      opmath_t negcoef,
      opmath_t poscoef,
      opmath_t negiptcoef,
      bool is_result)
      : negcoef_(negcoef),
        poscoef_(poscoef),
        negiptcoef_(negiptcoef),
        is_result_(is_result) {}

 private:
  opmath_t negcoef_;
  opmath_t poscoef_;
  opmath_t negiptcoef_;
  bool is_result_;
};

void elu_backward_kernel(
    TensorIteratorBase& iter,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    bool is_result) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "elu_backward_xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto negcoef = alpha.to<opmath_t>() * scale.to<opmath_t>();
        auto poscoef = scale.to<opmath_t>();
        auto negiptcoef = input_scale.to<opmath_t>();
        EluBackwardOutFunctor<scalar_t, opmath_t> f(
            negcoef, poscoef, negiptcoef, is_result);
        gpu_kernel(iter, f);
      });
}

} // namespace at::native::xpu
