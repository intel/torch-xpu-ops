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

#include <ATen/native/xpu/sycl/CopysignKernel.h>

namespace at::native::xpu {

template <typename scalar_t>
struct CopysignFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return std::copysign(a, b);
  }
};

void copysign_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "copysign_xpu",
      [&]() { gpu_kernel_with_scalars(iter, CopysignFunctor<scalar_t>()); });
}

} // namespace at::native::xpu
