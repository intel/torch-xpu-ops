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
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "max_elementwise_xpu", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
          iter, MaximumIntFunctor<scalar_t>());
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "max_elementwise_xpu",
        [&]() {
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, MaximumFunctor<scalar_t>());
        });
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
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "min_elementwise_xpu", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
          iter, MinimumIntFunctor<scalar_t>());
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "min_elementwise_xpu",
        [&]() {
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, MinimumFunctor<scalar_t>());
        });
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
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.common_dtype(),
        "fmax_xpu",
        [&]() {
          FmaxFunctor<scalar_t> f;
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
        });
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
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.common_dtype(),
        "fmin_xpu",
        [&]() {
          FminFunctor<scalar_t> f;
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
        });
  } else {
    minimum_kernel(iter);
  }
}

} // namespace at::native::xpu
