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
#include <ATen/NumericUtils.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/BinaryMiscBackwardOpsKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct SigmoidBackwardComplexFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const auto one = opmath_t{1.};
    const auto comp_b = static_cast<opmath_t>(b);
    const auto comp_a = static_cast<opmath_t>(a);
    return static_cast<scalar_t>(comp_a * std::conj((one - comp_b) * comp_b));
  }
};

template <typename scalar_t>
struct SigmoidBackwardFloatFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a * (scalar_t(1.) - b) * b;
  }
};

void sigmoid_backward_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, dtype, "sigmoid_backward_xpu", [&]() {
          gpu_kernel(iter, SigmoidBackwardComplexFunctor<scalar_t>());
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        dtype,
        "sigmoid_backward_xpu",
        [&]() { gpu_kernel(iter, SigmoidBackwardFloatFunctor<scalar_t>()); });
  }
}

template <typename scalar_t>
struct LogitBackward0Functor {
  using T_ACC = acc_type_device<scalar_t, c10::DeviceType::XPU>;
  scalar_t operator()(scalar_t dy, scalar_t x) const {
    const T_ACC dy_acc = static_cast<T_ACC>(dy);
    const T_ACC x_acc = static_cast<T_ACC>(x);
    // suppress compiler optimization on data type promotion.
    volatile T_ACC res = (x_acc < T_ACC(0) || x_acc > T_ACC(1))
        ? std::numeric_limits<T_ACC>::quiet_NaN()
        : dy_acc / (x_acc * (T_ACC(1) - x_acc));
    return res;
  }
};

template <typename scalar_t>
struct LogitBackward1Functor {
  using T_ACC = acc_type_device<scalar_t, c10::DeviceType::XPU>;
  scalar_t operator()(scalar_t dy, scalar_t x) const {
    const T_ACC dy_acc = static_cast<T_ACC>(dy);
    const T_ACC x_acc = static_cast<T_ACC>(x);
    // suppress compiler optimization on data type promotion.
    volatile T_ACC res = (x_acc < lo_ || x_acc > hi_)
        ? T_ACC(0)
        : dy_acc / (x_acc * (T_ACC(1) - x_acc));
    return res;
  }
  LogitBackward1Functor(const T_ACC lo, const T_ACC hi) : lo_(lo), hi_(hi) {}

 private:
  T_ACC lo_;
  T_ACC hi_;
};

void logit_backward_kernel(TensorIteratorBase& iter, const Scalar& eps_scalar) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "logit_xpu",
      [&]() {
        using T_ACC = acc_type_device<scalar_t, kXPU>;
        const T_ACC eps = eps_scalar.to<T_ACC>();
        if (eps < T_ACC(0)) {
          gpu_kernel(iter, LogitBackward0Functor<scalar_t>());
        } else {
          const T_ACC lo = eps;
          const T_ACC hi = T_ACC(1) - eps;
          gpu_kernel(iter, LogitBackward1Functor<scalar_t>(lo, hi));
        }
      });
}

template <typename scalar_t>
struct TanhBackwardComplexFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    using comp_t = at::opmath_type<scalar_t>;
    const auto one = comp_t{1.};
    const auto comp_b = static_cast<comp_t>(b);
    const auto comp_a = static_cast<comp_t>(a);
    return static_cast<scalar_t>(comp_a * std::conj(one - comp_b * comp_b));
  }
};

template <typename scalar_t>
struct TanhBackwardFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a * (scalar_t{1.} - b * b);
  }
};

void tanh_backward_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, dtype, "tanh_backward_complex_xpu", [&]() {
          gpu_kernel(iter, TanhBackwardComplexFunctor<scalar_t>());
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        dtype,
        "tanh_backward_xpu",
        [&]() { gpu_kernel(iter, TanhBackwardFunctor<scalar_t>()); });
  }
}

} // namespace at::native::xpu
