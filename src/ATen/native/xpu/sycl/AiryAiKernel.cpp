/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/MathExtensions.h>

#include <ATen/native/xpu/sycl/AiryAiKernel.h>

namespace at::native::xpu {
template <typename scalar_t>
struct AiryAiFunctor {
  scalar_t operator()(scalar_t a) const {
    return airy_ai_forward(a);
  }
};

void airy_ai_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "airy_ai_xpu", [&]() {
    gpu_kernel(iter, AiryAiFunctor<scalar_t>());
  });
}

} // namespace at::native::xpu