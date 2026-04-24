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
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/IGammaKernel.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/MathExtensions.h>

namespace at::native::xpu {

template <typename scalar_t>
struct IgammaFunctor {
  IgammaFunctor(bool calc_igammac) : calc_igammac_(calc_igammac) {}
  bool calc_igammac_;
#if defined(__clang__)
  [[clang::optnone]]
#elif defined(__GNUC__)
  __attribute__((optimize("O0")))
#endif
  scalar_t operator()(scalar_t a, scalar_t b) const {
    if (calc_igammac_) {
      return calc_igammac<scalar_t>(a, b);
    } else {
      return calc_igamma<scalar_t>(a, b);
    }
  }
};

void igamma_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "igamma_xpu", [&]() {
    gpu_kernel(iter, IgammaFunctor<scalar_t>(false));
  });
}

void igammac_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "igammac_xpu", [&]() {
    gpu_kernel(iter, IgammaFunctor<scalar_t>(true));
  });
}

} // namespace at::native::xpu
