/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/AbsKernel.h>

namespace at::native::xpu {

template <typename scalar_t>
struct AbsFunctor {
  scalar_t operator()(const scalar_t a) const {
    return std::abs(a);
  }
};

void abs_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_V2(
        dtype,
        "abs_xpu",
        AT_WRAP([&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          gpu_kernel(iter, AbsFunctor<opmath_t>());
        }),
        AT_EXPAND(AT_COMPLEX_TYPES),
        kComplexHalf,
        kBComplex32);
  } else {
    AT_DISPATCH_V2(
        iter.dtype(),
        "abs_xpu",
        AT_WRAP([&]() { gpu_kernel(iter, AbsFunctor<scalar_t>()); }),
        AT_EXPAND(AT_ALL_TYPES),
        ScalarType::Half,
        ScalarType::BFloat16,
        ScalarType::Bool);
  }
}

} // namespace at::native::xpu
