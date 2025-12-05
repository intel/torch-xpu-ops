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
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Math.h>
#include <ATen/native/xpu/sycl/HermitePolynomialHKernel.h>
#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct HermitePolynomialHFunctor {
  scalar_t operator()(scalar_t x, scalar_t n_) const {
    auto n = static_cast<int64_t>(n_);
    if (n < 0) {
      return scalar_t(0.0);
    } else if (n == 0) {
      return scalar_t(1.0);
    } else if (n == 1) {
      return x + x;
    } else if (n > getHermitianLimit<scalar_t>()) {
      return std::numeric_limits<scalar_t>::quiet_NaN();
    }

    scalar_t p = scalar_t(1.0);
    scalar_t q = x + x;
    scalar_t r = scalar_t(0.0);

    for (int64_t k = 2; k < n + n; k += 2) {
      r = (x + x) * q - k * p;
      p = q;
      q = r;
    }

    return r;
  }
};

void hermite_polynomial_h_kernel(TensorIteratorBase& iterator) {
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "hermite_polynomial_h_xpu", [&]() {
        gpu_kernel_with_scalars(
            iterator, HermitePolynomialHFunctor<scalar_t>());
      });
}

} // namespace at::native::xpu
