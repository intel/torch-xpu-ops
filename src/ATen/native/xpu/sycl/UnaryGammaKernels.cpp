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
#include <ATen/core/Tensor.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/CopyKernel.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/MathExtensions.h>
#include <c10/core/ScalarType.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/UnaryGammaKernels.h>

namespace at::native::xpu {

template <typename scalar_t, bool USE_FP64_PI>
struct DigammaFunctor {
  scalar_t operator()(scalar_t a) const {
    if constexpr (USE_FP64_PI) {
      return calc_digamma<scalar_t, double>(a);
    } else {
      using pi_t = at::acc_type_device<scalar_t, kXPU>;
      return calc_digamma<scalar_t, pi_t>(a);
    }
  }
};

void digamma_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "digamma_xpu",
      [&]() {
        if (syclHasFloat64()) {
          gpu_kernel(iter, DigammaFunctor<scalar_t, true>());
        } else {
          gpu_kernel(iter, DigammaFunctor<scalar_t, false>());
        }
      });
}

template <typename scalar_t>
struct TrigammaFunctor {
  scalar_t operator()(scalar_t a) const {
    return calc_trigamma(a);
  }
};

void trigamma_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "trigamma_xpu",
      [&]() { gpu_kernel(iter, TrigammaFunctor<scalar_t>()); });
}

template <typename scalar_t>
struct PolygammaFunctor {
  scalar_t operator()(scalar_t a) const {
    return calc_polygamma<scalar_t, true>(a, static_cast<int>(n_));
  }
  PolygammaFunctor(const int64_t n) : n_(n) {}

 private:
  int64_t n_;
};

void polygamma_kernel(TensorIteratorBase& iter, int64_t n) {
  if (n == 0) {
    digamma_kernel(iter);
  } else if (n == 1) {
    trigamma_kernel(iter);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.common_dtype(),
        "polygamma_xpu",
        [&]() { gpu_kernel(iter, PolygammaFunctor<scalar_t>(n)); });
  }
}

template <typename scalar_t>
struct LgammaFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::lgamma(a);
  }
};

void lgamma_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "lgamma_xpu",
      [&]() { gpu_kernel(iter, LgammaFunctor<scalar_t>()); });
}

} // namespace at::native::xpu
