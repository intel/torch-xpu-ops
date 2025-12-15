/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/core/op_registration/adaption.h>

#include <ATen/ops/embedding_dense_backward_native.h>

#include <ATen/native/xpu/sycl/EmbeddingKernels.h>
#include <comm/xpu_aten.h>

namespace at {
namespace native {
Tensor embedding_dense_backward_xpu(
    const Tensor& grad_output,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "xpu::embedding_dense_backward",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, indices, "xpu::embedding_dense_backward", "indices");
  return xpu::embedding_dense_backward_kernel(
      grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
  ;
}

Tensor& embedding_renorm_xpu_(
    Tensor& self,
    const Tensor& indices,
    double max_norm,
    double norm_type) {
  return native::xpu::embedding_renorm_kernel(
      self, indices, max_norm, norm_type);
}

} // namespace native
} // namespace at
