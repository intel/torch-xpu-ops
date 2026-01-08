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
#include <ATen/native/cpu/mixed_data_type.h>
#include <ATen/native/xpu/sycl/FractionalMaxPool2dKernels.h>

#include <ATen/ops/fractional_max_pool2d_backward_native.h>
#include <ATen/ops/fractional_max_pool2d_native.h>

namespace at::native {

TORCH_IMPL_FUNC(fractional_max_pool2d_out_xpu)
(const Tensor& input,
 IntArrayRef pool_size,
 IntArrayRef output_size,
 const Tensor& randomSamples,
 const Tensor& output,
 const Tensor& indices) {
  xpu::fractional_max_pool2d_kernel(
      input, pool_size, output_size, randomSamples, output, indices);
}

TORCH_IMPL_FUNC(fractional_max_pool2d_backward_xpu)
(const Tensor& gradOutput,
 const Tensor& input,
 IntArrayRef pool_size /* unused */,
 IntArrayRef output_size,
 const Tensor& indices,
 const Tensor& gradInput) {
  xpu::fractional_max_pool2d_backward_kernel(
      gradOutput, input, pool_size, output_size, indices, gradInput);
}

} // namespace at::native
