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
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/UnaryGeometricAcoshKernel.h>

namespace at::native::xpu {

template <typename scalar_t, typename acc_t = scalar_t>
struct AcoshFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::acosh(static_cast<acc_t>(a));
  }
};

void acosh_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "acosh_xpu", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          auto caller = AcoshFunctor<scalar_t, opmath_t>();
          gpu_kernel(iter, caller);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        common_dtype,
        "acosh_xpu",
        [&]() {
          auto caller = AcoshFunctor<scalar_t>();
          gpu_kernel(iter, caller);
        });
  }
}

} // namespace at::native::xpu
