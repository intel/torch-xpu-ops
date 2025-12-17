/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/xpu/sycl/WeightNormKernels.h>

namespace at {
namespace native {
std::tuple<Tensor, Tensor> weight_norm_xpu(
    const Tensor& v,
    const Tensor& g,
    int64_t dim) {
  return native::xpu::weight_norm_kernel(v, g, dim);
}

std::tuple<Tensor, Tensor> weight_norm_backward_xpu(
    const Tensor& grad_w,
    const Tensor& saved_v,
    const Tensor& saved_g,
    const Tensor& saved_norms,
    int64_t dim) {
  TORCH_CHECK(saved_v.is_contiguous(), "saved_v must be contiguous");
  TORCH_CHECK(saved_g.is_contiguous(), "saved_g must be contiguous");
  TORCH_CHECK(saved_norms.is_contiguous(), "saved_norms must be contiguous");
  TORCH_CHECK(
      dim == 0 || dim == saved_v.dim() - 1,
      "fused kernels can only be applied for first or last dim")

  return native::xpu::weight_norm_backward_kernel(
      grad_w, saved_v, saved_g, saved_norms, dim);
}

} // namespace native
} // namespace at