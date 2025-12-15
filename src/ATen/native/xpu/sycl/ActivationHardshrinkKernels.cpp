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
#include <ATen/native/xpu/sycl/ActivationHardshrinkKernels.h>
#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct HardshrinkFunctor {
  scalar_t operator()(scalar_t a) const {
    return (a >= -lambd_ && a <= lambd_) ? scalar_t(0) : a;
  }

  HardshrinkFunctor(const scalar_t lambd) : lambd_(lambd) {}

 private:
  const scalar_t lambd_;
};

void hardshrink_kernel(TensorIteratorBase& iter, const Scalar& value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardshrink_xpu",
      [&]() {
        auto lambd = value.to<scalar_t>();
        auto caller = HardshrinkFunctor<scalar_t>(lambd);
        gpu_kernel(iter, caller);
      });
}

} // namespace at::native::xpu
