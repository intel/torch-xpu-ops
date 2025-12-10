/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <comm/xpu_aten.h>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/xpu/sycl/CopyKernel.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/UnaryLogKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct LogFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::log(a);
  }
};

void log_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, iter.common_dtype(), "log_xpu", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          gpu_kernel(iter, LogFunctor<opmath_t>());
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        iter.common_dtype(),
        "log_xpu",
        [&]() { gpu_kernel(iter, LogFunctor<scalar_t>()); });
  }
}

template <typename scalar_t>
struct Log10Functor {
  scalar_t operator()(scalar_t x) const {
    return std::log10(x);
  }
};

void log10_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      iter.common_dtype(),
      "log10_xpu",
      [&]() { gpu_kernel(iter, Log10Functor<scalar_t>()); });
}

template <typename scalar_t>
struct Log1pFunctor {
  scalar_t operator()(scalar_t x) const {
    return std::log1p(x);
  }
};

void log1p_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      iter.common_dtype(),
      "log1p_xpu",
      [&]() { gpu_kernel(iter, Log1pFunctor<scalar_t>()); });
}

template <typename scalar_t>
struct Log2Functor {
  scalar_t operator()(scalar_t x) const {
    return std::log2(x);
  }
};

void log2_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      iter.common_dtype(),
      "log2_xpu",
      [&]() { gpu_kernel(iter, Log2Functor<scalar_t>()); });
}

} // namespace at::native::xpu
