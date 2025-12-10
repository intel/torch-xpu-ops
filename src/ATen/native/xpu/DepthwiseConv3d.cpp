/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/xpu/sycl/DepthwiseConv3dKernels.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <comm/SYCLContext.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/conv_depthwise3d_native.h>
#include <ATen/ops/empty.h>
#endif

#include <algorithm>
#include <limits>
#include <tuple>

namespace at::native {
Tensor conv_depthwise3d_xpu(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  return xpu::conv_depthwise3d_kernel(
      input, weight, kernel_size, bias_opt, stride, padding, dilation);
}

std::tuple<Tensor&, Tensor&, Tensor&> conv_depthwise3d_backward_xpu_out(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias) {
  if (grad_weight.defined()) {
    grad_weight.resize_(weight.sizes());
    grad_weight.zero_();
  }
  return xpu::_depthwise_3d_backward_kernel(
      grad_input,
      grad_weight,
      grad_bias,
      grad_output,
      input,
      weight,
      kernel_size,
      stride,
      padding,
      dilation,
      {true, true, true});
}

std::tuple<Tensor, Tensor, Tensor> conv_depthwise3d_backward_xpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    const std::array<bool, 3> output_mask) {
  auto options = grad_output.options();
  Tensor grad_input =
      (output_mask[0] ? at::empty(input.sizes(), options) : Tensor());
  Tensor grad_weight =
      (output_mask[1] ? at::empty(weight.sizes(), options) : Tensor());
  Tensor grad_bias; /* undefined temporarily */

  return xpu::_depthwise_3d_backward_kernel(
      grad_input,
      grad_weight,
      grad_bias,
      grad_output,
      input,
      weight,
      kernel_size,
      stride,
      padding,
      dilation,
      output_mask);
}

} // namespace at::native
