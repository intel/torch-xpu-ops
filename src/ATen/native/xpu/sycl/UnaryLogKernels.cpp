/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <comm/xpu_aten.h>

#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <c10/util/complex.h>

#include <ATen/native/xpu/sycl/CopyKernel.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/UnaryLogKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct LogFunctor {
  scalar_t operator()(scalar_t a) const {
    if constexpr (c10::is_complex<scalar_t>::value) {
      return std::log(a);
    } else {
      using opmath_t = at::opmath_type<scalar_t>;
      return sycl::log(static_cast<opmath_t>(a));
    }
  }
};

void log_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    AT_DISPATCH_V2(
        iter.common_dtype(),
        "log_xpu",
        AT_WRAP([&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          gpu_kernel(iter, LogFunctor<opmath_t>());
        }),
        AT_EXPAND(AT_COMPLEX_TYPES),
        kComplexHalf,
        kBComplex32);
  } else {
    AT_DISPATCH_V2(
        iter.common_dtype(),
        "log_xpu",
        AT_WRAP([&]() { gpu_kernel(iter, LogFunctor<scalar_t>()); }),
        AT_EXPAND(AT_FLOATING_TYPES),
        ScalarType::Half,
        ScalarType::BFloat16);
  }
}

template <typename scalar_t>
struct Log10Functor {
  scalar_t operator()(scalar_t x) const {
    if constexpr (c10::is_complex<scalar_t>::value) {
      return std::log10(x);
    } else {
      using opmath_t = at::opmath_type<scalar_t>;
      return sycl::log10(static_cast<opmath_t>(x));
    }
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
    if constexpr (c10::is_complex<scalar_t>::value) {
      return std::log1p(x);
    } else {
      using opmath_t = at::opmath_type<scalar_t>;
      return sycl::log1p(static_cast<opmath_t>(x));
    }
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
    if constexpr (c10::is_complex<scalar_t>::value) {
      return std::log2(x);
    } else {
      using opmath_t = at::opmath_type<scalar_t>;
      return sycl::log2(static_cast<opmath_t>(x));
    }
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
