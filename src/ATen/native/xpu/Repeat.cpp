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
#include <ATen/native/xpu/sycl/RepeatKernel.h>

namespace at {
namespace native {
Tensor repeat_interleave_xpu(
    const Tensor& repeats,
    std::optional<int64_t> output_size) {
  return at::native::xpu::repeat_interleave_kernel(repeats, output_size);
}

} // namespace native
} // namespace at
