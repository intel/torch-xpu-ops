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
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <c10/core/Scalar.h>

#include <ATen/native/xpu/sycl/ScaledModifiedBesselK0Kernel.h>

namespace at::native::xpu {

template <typename scalar_t>
struct ScaledModifiedBesselK0Functor {
  scalar_t operator()(scalar_t a) const {
    return scaled_modified_bessel_k0_forward(a);
  }
};

void scaled_modified_bessel_k0_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(
      iter.common_dtype(), "scaled_modified_bessel_k0_xpu", [&]() {
        gpu_kernel(iter, ScaledModifiedBesselK0Functor<scalar_t>());
      });
}

} // namespace at::native::xpu
