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
#include <ATen/native/Pool.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/AveragePool2dKernels.h>
#include <comm/RegisterUtils.h>

#include <ATen/ops/avg_pool2d_backward_native.h>
#include <ATen/ops/avg_pool2d_native.h>

namespace at {
namespace native {

TORCH_IMPL_FUNC(avg_pool2d_out_xpu)
(const Tensor& input_,
 int64_t kH_,
 int64_t kW_,
 int64_t dH_,
 int64_t dW_,
 int64_t padH_,
 int64_t padW_,
 bool ceil_mode,
 bool count_include_pad,
 std::optional<int64_t> divisor_override,
 const Tensor& output) {
  xpu::avg_pool2d_kernel(
      input_,
      kH_,
      kW_,
      dH_,
      dW_,
      padH_,
      padW_,
      ceil_mode,
      count_include_pad,
      divisor_override,
      output);
}

TORCH_IMPL_FUNC(avg_pool2d_backward_out_xpu)
(const Tensor& gradOutput_,
 const Tensor& input_,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 bool ceil_mode,
 bool count_include_pad,
 std::optional<int64_t> divisor_override,
 const Tensor& gradInput) {
  xpu::avg_pool2d_backward_kernel(
      gradOutput_,
      input_,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override,
      gradInput);
}

} // namespace native
} // namespace at
