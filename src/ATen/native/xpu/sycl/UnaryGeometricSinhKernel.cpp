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

#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/UnaryGeometricSinhKernel.h>

namespace at::native::xpu {

template <typename scalar_t>
struct SinhComplexFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t operator()(scalar_t a) const {
    return std::sinh(static_cast<opmath_t>(a));
  }
};

template <typename scalar_t>
struct SinhFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::sinh(a);
  }
};

void sinh_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "sinh_xpu", [&]() {
          gpu_kernel(iter, SinhComplexFunctor<scalar_t>());
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        common_dtype,
        "sinh_xpu",
        [&]() { gpu_kernel(iter, SinhFunctor<scalar_t>()); });
  }
}

} // namespace at::native::xpu
