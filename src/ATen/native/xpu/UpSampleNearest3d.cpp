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
#include <ATen/native/xpu/sycl/UpSampleNearest3dKernels.h>

#include <ATen/ops/_upsample_nearest_exact3d.h>
#include <ATen/ops/_upsample_nearest_exact3d_backward.h>
#include <ATen/ops/_upsample_nearest_exact3d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact3d_native.h>
#include <ATen/ops/upsample_nearest3d.h>
#include <ATen/ops/upsample_nearest3d_backward.h>
#include <ATen/ops/upsample_nearest3d_backward_native.h>
#include <ATen/ops/upsample_nearest3d_native.h>

namespace at::native {

TORCH_IMPL_FUNC(_upsample_nearest_exact3d_out_xpu)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales_d,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  at::native::xpu::upsample_nearest3d_kernel(
      output, input, output_size, scales_d, scales_h, scales_w, true);
}

TORCH_IMPL_FUNC(upsample_nearest3d_out_xpu)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales_d,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  at::native::xpu::upsample_nearest3d_kernel(
      output, input, output_size, scales_d, scales_h, scales_w, false);
}

TORCH_IMPL_FUNC(_upsample_nearest_exact3d_backward_out_xpu)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales_d,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  at::native::xpu::upsample_nearest3d_backward_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      scales_d,
      scales_h,
      scales_w,
      true);
}

TORCH_IMPL_FUNC(upsample_nearest3d_backward_out_xpu)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales_d,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  at::native::xpu::upsample_nearest3d_backward_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      scales_d,
      scales_h,
      scales_w,
      false);
}

} // namespace at::native
