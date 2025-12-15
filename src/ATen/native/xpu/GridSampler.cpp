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

#include <ATen/core/op_registration/adaption.h>

#include <ATen/native/xpu/sycl/GridSamplerKernels.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros_like.h>
#include <comm/xpu_aten.h>

namespace at {
namespace native {

Tensor grid_sampler_2d_xpu(
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  return xpu::grid_sampler_2d_kernel(
      input, grid, interpolation_mode, padding_mode, align_corners);
}

std::tuple<Tensor, Tensor> grid_sampler_2d_backward_xpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask) {
  auto input_requires_grad = output_mask[0];
  Tensor grad_input = ([&]() {
    if (input_requires_grad) {
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    } else {
      return Tensor();
    }
  })();
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  xpu::grid_sampler_2d_backward_kernel(
      grad_input,
      grad_grid,
      grad_output,
      input,
      grid,
      interpolation_mode,
      padding_mode,
      align_corners,
      output_mask);
  return std::make_tuple(grad_input, grad_grid);
}

Tensor grid_sampler_3d_xpu(
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  return xpu::grid_sampler_3d_kernel(
      input, grid, interpolation_mode, padding_mode, align_corners);
}

std::tuple<Tensor, Tensor> grid_sampler_3d_backward_xpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask) {
  auto input_requires_grad = output_mask[0];
  Tensor grad_input = ([&]() {
    if (input_requires_grad) {
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    } else {
      return Tensor();
    }
  })();
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  xpu::grid_sampler_3d_backward_kernel(
      grad_input,
      grad_grid,
      grad_output,
      input,
      grid,
      interpolation_mode,
      padding_mode,
      align_corners,
      output_mask);
  return std::make_tuple(grad_input, grad_grid);
}

} // namespace native
} // namespace at
