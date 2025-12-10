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
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/BinaryGeometricKernels.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
struct Atan2Functor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return std::atan2(a, b);
  }
};

void atan2_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.common_dtype(),
      "atan2_xpu",
      [&]() { gpu_kernel(iter, Atan2Functor<scalar_t>()); });
}

template <typename scalar_t>
struct HypotFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return std::hypot(a, b);
  }
};

void hypot_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "hypot_xpu",
      [&]() {
        opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
            iter, HypotFunctor<scalar_t>());
      });
}

} // namespace xpu
} // namespace native
} // namespace at
