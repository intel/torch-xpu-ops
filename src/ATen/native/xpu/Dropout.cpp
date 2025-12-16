/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/DropoutKernels.h>

#include <ATen/ops/native_dropout_backward_native.h>
#include <ATen/ops/native_dropout_native.h>

#include <comm/xpu_aten.h>

namespace at {
namespace native {

::std::tuple<Tensor, Tensor> native_dropout_xpu(
    const Tensor& input,
    double p,
    ::std::optional<bool> train) {
  return xpu::dropout_kernel(input, p, train);
}

Tensor native_dropout_backward_xpu(
    const Tensor& grad_output,
    const Tensor& mask,
    double scale) {
  return xpu::dropout_backward_kernel(grad_output, mask, scale);
}

std::tuple<Tensor, Tensor> fused_dropout_xpu(
    const Tensor& self,
    double p,
    std::optional<Generator> gen_) {
  return xpu::fused_dropout_kernel(self, p, gen_);
}

Tensor masked_scale_xpu(const Tensor& self, const Tensor& mask, double scale) {
  return xpu::masked_scale_kernel(self, mask, scale);
}

} // namespace native
} // namespace at
