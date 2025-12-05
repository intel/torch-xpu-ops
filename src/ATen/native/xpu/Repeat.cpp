/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
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
