/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions of this file are derived from PyTorch
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ATen/NamedTensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/DilatedMaxPool3d.h>

#include <ATen/ops/empty.h>
#include <ATen/ops/max_pool3d_with_indices_backward_native.h>
#include <ATen/ops/max_pool3d_with_indices_native.h>
namespace at {
namespace native {

std::tuple<Tensor, Tensor> max_pool3d_with_indices_xpu(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  Tensor output = at::empty({0}, input.options());
  Tensor indices = at::empty({0}, input.options().dtype(kLong));

  NoNamesGuard guard;
  at::native::xpu::max_pool3d_with_indices_kernel(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      output,
      indices);
  guard.reset();
  namedinference::propagate_names(output, input);
  namedinference::propagate_names(indices, input);

  return std::tuple<Tensor, Tensor>(output, indices);
}

std::tuple<Tensor&, Tensor&> max_pool3d_with_indices_out_xpu(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    Tensor& output,
    Tensor& indices) {
  at::native::xpu::max_pool3d_with_indices_kernel(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      output,
      indices);

  return std::tuple<Tensor&, Tensor&>(output, indices);
}

Tensor& max_pool3d_with_indices_backward_out_xpu(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices,
    Tensor& gradInput) {
  globalContext().alertNotDeterministic(
      "max_pool3d_with_indices_backward_out_xpu");
  at::native::xpu::max_pool3d_with_indices_backward_kernel(
      gradInput,
      gradOutput,
      input,
      indices,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);

  return gradInput;
}

Tensor max_pool3d_with_indices_backward_xpu(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices) {
  globalContext().alertNotDeterministic("max_pool3d_with_indices_backward_xpu");
  auto gradInput = at::empty(input.sizes(), input.options());
  at::native::xpu::max_pool3d_with_indices_backward_kernel(
      gradInput,
      gradOutput,
      input,
      indices,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);

  return gradInput;
}
} // namespace native
} // namespace at
