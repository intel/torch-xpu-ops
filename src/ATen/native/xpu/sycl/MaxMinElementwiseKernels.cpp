/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/MaxMinElementwiseKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct MaximumIntFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return std::max(a, b);
  }
};

template <>
struct MaximumIntFunctor<bool> {
  bool operator()(bool a, bool b) const {
    return a || b;
    ;
  }
};

template <typename scalar_t>
struct MaximumFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    if (a != a) {
      return a;
    } else if (b != b) {
      return b;
    } else {
      return std::max(a, b);
    }
  }
};

void maximum_kernel(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    opmath_symmetric_gpu_kernel_with_scalars<bool>(
        iter, MaximumIntFunctor<bool>());
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/false)) {
    AT_DISPATCH_V2(
        iter.dtype(),
        "max_elementwise_xpu",
        AT_WRAP([&]() {
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, MaximumIntFunctor<scalar_t>());
        }),
        AT_EXPAND(AT_INTEGRAL_TYPES_V2));
  } else {
    AT_DISPATCH_V2(
        iter.dtype(),
        "max_elementwise_xpu",
        AT_WRAP([&]() {
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, MaximumFunctor<scalar_t>());
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        kHalf,
        kBFloat16);
  }
}

template <typename scalar_t>
struct MinimumIntFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return std::min(a, b);
  }
};

template <>
struct MinimumIntFunctor<bool> {
  bool operator()(bool a, bool b) const {
    return a && b;
    ;
  }
};

template <typename scalar_t>
struct MinimumFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    if (a != a) {
      return a;
    } else if (b != b) {
      return b;
    } else {
      return std::min(a, b);
    }
  }
};

void minimum_kernel(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    opmath_symmetric_gpu_kernel_with_scalars<bool>(
        iter, MinimumIntFunctor<bool>());
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/false)) {
    AT_DISPATCH_V2(
        iter.dtype(),
        "min_elementwise_xpu",
        AT_WRAP([&]() {
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, MinimumIntFunctor<scalar_t>());
        }),
        AT_EXPAND(AT_INTEGRAL_TYPES_V2));
  } else {
    AT_DISPATCH_V2(
        iter.dtype(),
        "min_elementwise_xpu",
        AT_WRAP([&]() {
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, MinimumFunctor<scalar_t>());
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        kHalf,
        kBFloat16);
  }
}

template <typename scalar_t>
struct FmaxFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return std::fmax(a, b);
  }
};

void fmax_kernel(TensorIteratorBase& iter) {
  if (isFloatingType(iter.common_dtype())) {
    AT_DISPATCH_V2(
        iter.common_dtype(),
        "fmax_xpu",
        AT_WRAP([&]() {
          FmaxFunctor<scalar_t> f;
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        kHalf,
        kBFloat16);
  } else {
    maximum_kernel(iter);
  }
}

template <typename scalar_t>
struct FminFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return std::fmin(a, b);
  }
};

void fmin_kernel(TensorIteratorBase& iter) {
  if (isFloatingType(iter.common_dtype())) {
    AT_DISPATCH_V2(
        iter.common_dtype(),
        "fmin_xpu",
        AT_WRAP([&]() {
          FminFunctor<scalar_t> f;
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        kHalf,
        kBFloat16);
  } else {
    minimum_kernel(iter);
  }
}

} // namespace at::native::xpu
