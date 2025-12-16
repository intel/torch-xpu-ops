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

#include <ATen/native/xpu/sycl/BesselJ0Kernel.h>

namespace at::native::xpu {

template <typename scalar_t>
struct BesselJ0Functor {
  scalar_t operator()(scalar_t a) const {
    return bessel_j0_forward(a);
  }
};

void bessel_j0_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "bessel_j0_xpu", [&]() {
    gpu_kernel(iter, BesselJ0Functor<scalar_t>());
  });
}

} // namespace at::native::xpu
