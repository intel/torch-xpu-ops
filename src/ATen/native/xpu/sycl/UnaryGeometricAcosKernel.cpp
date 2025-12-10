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

#include <ATen/native/xpu/sycl/UnaryGeometricAcosKernel.h>

namespace at::native::xpu {
template <typename scalar_t, typename acc_t = scalar_t>
struct AcosFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::acos(static_cast<acc_t>(a));
  }
};

void acos_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "acos_xpu", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          auto caller = AcosFunctor<scalar_t, opmath_t>();
          gpu_kernel(iter, caller);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        common_dtype,
        "acos_xpu",
        [&]() {
          auto caller = AcosFunctor<scalar_t>();
          gpu_kernel(iter, caller);
        });
  }
}

} // namespace at::native::xpu
