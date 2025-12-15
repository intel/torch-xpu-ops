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
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/ActivationHardsigmoidKernels.h>

namespace at::native::xpu {

template <typename scalar_t, typename opmath_t>
struct HardsigmoidFunctor {
  scalar_t operator()(scalar_t self_val) const {
    opmath_t x = static_cast<opmath_t>(self_val);
    return std::min(std::max(x + three_, zero_), six_) * one_sixth_;
  }

  HardsigmoidFunctor(
      const opmath_t zero,
      const opmath_t one_sixth,
      const opmath_t three,
      const opmath_t six)
      : zero_(zero), one_sixth_(one_sixth), three_(three), six_(six) {}

 private:
  const opmath_t zero_;
  const opmath_t one_sixth_;
  const opmath_t three_;
  const opmath_t six_;
};

void hardsigmoid_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardsigmoid_xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        const opmath_t zero(0.0f);
        const opmath_t one_sixth(1.0f / 6.0f);
        const opmath_t three(3.0f);
        const opmath_t six(6.0f);
        HardsigmoidFunctor<scalar_t, opmath_t> f(zero, one_sixth, three, six);
        gpu_kernel(iter, f);
      });
}

template <typename scalar_t, typename opmath_t>
struct HardsigmoidBackwardFunctor {
  scalar_t operator()(scalar_t grad_val_, scalar_t self_val_) const {
    opmath_t grad_val = static_cast<opmath_t>(grad_val_);
    opmath_t self_val = static_cast<opmath_t>(self_val_);
    return (self_val > neg_three_ && self_val < three_) ? grad_val * one_sixth_
                                                        : zero_;
  }

  HardsigmoidBackwardFunctor(
      const opmath_t zero,
      const opmath_t three,
      const opmath_t neg_three,
      const opmath_t one_sixth)
      : zero_(zero),
        three_(three),
        neg_three_(neg_three),
        one_sixth_(one_sixth) {}

 private:
  const opmath_t zero_;
  const opmath_t three_;
  const opmath_t neg_three_;
  const opmath_t one_sixth_;
};

void hardsigmoid_backward_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardsigmoid_backward_xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        const opmath_t zero(0.0f);
        const opmath_t three(3.0f);
        const opmath_t neg_three(-3.0f);
        const opmath_t one_sixth(1.0f / 6.0f);
        HardsigmoidBackwardFunctor<scalar_t, opmath_t> f(
            zero, three, neg_three, one_sixth);
        gpu_kernel(iter, f);
      });
}

} // namespace at::native::xpu
