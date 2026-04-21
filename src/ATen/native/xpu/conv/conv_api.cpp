/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/xpu/conv/conv_api.h>
#include <torch/library.h>

#ifdef USE_SYCLTLA
#define SYCL_DISABLE_FSYCL_SYCLHPP_WARNING
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <ATen/xpu/XPUContext.h>
#pragma GCC diagnostic pop

namespace sycltla {

// --- conv2d impl forward declarations ---
at::Tensor conv2d_fprop_impl(
    sycl::queue& queue,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    bool has_bias,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilH,
    int dilW,
    int64_t groups);

at::Tensor conv2d_dgrad_impl(
    sycl::queue& queue,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    int H,
    int W,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilH,
    int dilW,
    int64_t groups);

at::Tensor conv2d_wgrad_impl(
    sycl::queue& queue,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int kH,
    int kW,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilH,
    int dilW,
    int64_t groups);

// --- conv3d impl forward declarations ---
at::Tensor conv3d_fprop_impl(
    sycl::queue& queue,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    bool has_bias,
    int strideD,
    int strideH,
    int strideW,
    int padD,
    int padH,
    int padW,
    int dilD,
    int dilH,
    int dilW,
    int64_t groups);

at::Tensor conv3d_dgrad_impl(
    sycl::queue& queue,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    int D,
    int H,
    int W,
    int strideD,
    int strideH,
    int strideW,
    int padD,
    int padH,
    int padW,
    int dilD,
    int dilH,
    int dilW,
    int64_t groups);

at::Tensor conv3d_wgrad_impl(
    sycl::queue& queue,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int kD,
    int kH,
    int kW,
    int strideD,
    int strideH,
    int strideW,
    int padD,
    int padH,
    int padW,
    int dilD,
    int dilH,
    int dilW,
    int64_t groups);

} // namespace sycltla
#endif // USE_SYCLTLA

// Helper macros for common checks
#define CONV_CHECK_DTYPE(name, t)                               \
  TORCH_CHECK(                                                  \
      (t).dtype() == at::kBFloat16 || (t).dtype() == at::kHalf, \
      "sycltla::" name ": only bf16/fp16 supported, got ",      \
      (t).dtype())

#define CONV_CHECK_XPU(name, t) \
  TORCH_CHECK((t).is_xpu(), "sycltla::" name ": tensor must be on XPU")

namespace sycltla {

bool is_conv_available() {
#ifndef USE_SYCLTLA
  return false;
#else
  return true;
#endif
}

// =========================================================================
// Conv2d fprop
// =========================================================================
at::Tensor conv2d_fprop(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
#ifndef USE_SYCLTLA
  TORCH_CHECK(false, "sycltla::conv2d_fprop: not compiled with SYCLTLA.");
  return at::Tensor();
#else
  CONV_CHECK_XPU("conv2d_fprop", input);
  CONV_CHECK_XPU("conv2d_fprop", weight);
  CONV_CHECK_DTYPE("conv2d_fprop", input);
  CONV_CHECK_DTYPE("conv2d_fprop", weight);
  TORCH_CHECK(
      input.dtype() == weight.dtype(), "sycltla::conv2d_fprop: dtype mismatch");
  TORCH_CHECK(input.dim() == 4 && weight.dim() == 4);
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous());
  TORCH_CHECK(
      stride.size() == 2 && padding.size() == 2 && dilation.size() == 2);
  TORCH_CHECK(input.size(1) == weight.size(1) * groups);

  auto& queue = at::xpu::getCurrentXPUStream().queue();
  bool has_bias = bias.has_value() && bias->defined();
  at::Tensor bias_tensor = has_bias ? *bias : at::Tensor();
  return sycltla::conv2d_fprop_impl(
      queue,
      input,
      weight,
      bias_tensor,
      has_bias,
      (int)stride[0],
      (int)stride[1],
      (int)padding[0],
      (int)padding[1],
      (int)dilation[0],
      (int)dilation[1],
      groups);
#endif
}

// =========================================================================
// Conv2d dgrad
// =========================================================================
at::Tensor conv2d_dgrad(
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    at::IntArrayRef input_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
#ifndef USE_SYCLTLA
  TORCH_CHECK(false, "not compiled with SYCLTLA");
  return at::Tensor();
#else
  CONV_CHECK_XPU("conv2d_dgrad", grad_output);
  CONV_CHECK_XPU("conv2d_dgrad", weight);
  CONV_CHECK_DTYPE("conv2d_dgrad", grad_output);
  TORCH_CHECK(grad_output.dtype() == weight.dtype());
  TORCH_CHECK(input_size.size() == 4); // [N, C, H, W]

  auto& queue = at::xpu::getCurrentXPUStream().queue();
  return sycltla::conv2d_dgrad_impl(
      queue,
      grad_output.contiguous(),
      weight.contiguous(),
      (int)input_size[2],
      (int)input_size[3],
      (int)stride[0],
      (int)stride[1],
      (int)padding[0],
      (int)padding[1],
      (int)dilation[0],
      (int)dilation[1],
      groups);
#endif
}

// =========================================================================
// Conv2d wgrad
// =========================================================================
at::Tensor conv2d_wgrad(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
#ifndef USE_SYCLTLA
  TORCH_CHECK(false, "not compiled with SYCLTLA");
  return at::Tensor();
#else
  CONV_CHECK_XPU("conv2d_wgrad", grad_output);
  CONV_CHECK_XPU("conv2d_wgrad", input);
  CONV_CHECK_DTYPE("conv2d_wgrad", grad_output);
  TORCH_CHECK(grad_output.dtype() == input.dtype());
  TORCH_CHECK(kernel_size.size() == 2); // [kH, kW]

  auto& queue = at::xpu::getCurrentXPUStream().queue();
  return sycltla::conv2d_wgrad_impl(
      queue,
      grad_output.contiguous(),
      input.contiguous(),
      (int)kernel_size[0],
      (int)kernel_size[1],
      (int)stride[0],
      (int)stride[1],
      (int)padding[0],
      (int)padding[1],
      (int)dilation[0],
      (int)dilation[1],
      groups);
#endif
}

// =========================================================================
// Conv3d fprop
// =========================================================================
at::Tensor conv3d_fprop(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
#ifndef USE_SYCLTLA
  TORCH_CHECK(false, "not compiled with SYCLTLA");
  return at::Tensor();
#else
  CONV_CHECK_XPU("conv3d_fprop", input);
  CONV_CHECK_XPU("conv3d_fprop", weight);
  CONV_CHECK_DTYPE("conv3d_fprop", input);
  CONV_CHECK_DTYPE("conv3d_fprop", weight);
  TORCH_CHECK(input.dtype() == weight.dtype());
  TORCH_CHECK(input.dim() == 5 && weight.dim() == 5);
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous());
  TORCH_CHECK(
      stride.size() == 3 && padding.size() == 3 && dilation.size() == 3);
  TORCH_CHECK(input.size(1) == weight.size(1) * groups);

  auto& queue = at::xpu::getCurrentXPUStream().queue();
  bool has_bias = bias.has_value() && bias->defined();
  at::Tensor bias_tensor = has_bias ? *bias : at::Tensor();
  return sycltla::conv3d_fprop_impl(
      queue,
      input,
      weight,
      bias_tensor,
      has_bias,
      (int)stride[0],
      (int)stride[1],
      (int)stride[2],
      (int)padding[0],
      (int)padding[1],
      (int)padding[2],
      (int)dilation[0],
      (int)dilation[1],
      (int)dilation[2],
      groups);
#endif
}

// =========================================================================
// Conv3d dgrad
// =========================================================================
at::Tensor conv3d_dgrad(
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    at::IntArrayRef input_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
#ifndef USE_SYCLTLA
  TORCH_CHECK(false, "not compiled with SYCLTLA");
  return at::Tensor();
#else
  CONV_CHECK_XPU("conv3d_dgrad", grad_output);
  CONV_CHECK_XPU("conv3d_dgrad", weight);
  CONV_CHECK_DTYPE("conv3d_dgrad", grad_output);
  TORCH_CHECK(grad_output.dtype() == weight.dtype());
  TORCH_CHECK(input_size.size() == 5); // [N, C, D, H, W]

  auto& queue = at::xpu::getCurrentXPUStream().queue();
  return sycltla::conv3d_dgrad_impl(
      queue,
      grad_output.contiguous(),
      weight.contiguous(),
      (int)input_size[2],
      (int)input_size[3],
      (int)input_size[4],
      (int)stride[0],
      (int)stride[1],
      (int)stride[2],
      (int)padding[0],
      (int)padding[1],
      (int)padding[2],
      (int)dilation[0],
      (int)dilation[1],
      (int)dilation[2],
      groups);
#endif
}

// =========================================================================
// Conv3d wgrad
// =========================================================================
at::Tensor conv3d_wgrad(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
#ifndef USE_SYCLTLA
  TORCH_CHECK(false, "not compiled with SYCLTLA");
  return at::Tensor();
#else
  CONV_CHECK_XPU("conv3d_wgrad", grad_output);
  CONV_CHECK_XPU("conv3d_wgrad", input);
  CONV_CHECK_DTYPE("conv3d_wgrad", grad_output);
  TORCH_CHECK(grad_output.dtype() == input.dtype());
  TORCH_CHECK(kernel_size.size() == 3); // [kD, kH, kW]

  auto& queue = at::xpu::getCurrentXPUStream().queue();
  return sycltla::conv3d_wgrad_impl(
      queue,
      grad_output.contiguous(),
      input.contiguous(),
      (int)kernel_size[0],
      (int)kernel_size[1],
      (int)kernel_size[2],
      (int)stride[0],
      (int)stride[1],
      (int)stride[2],
      (int)padding[0],
      (int)padding[1],
      (int)padding[2],
      (int)dilation[0],
      (int)dilation[1],
      (int)dilation[2],
      groups);
#endif
}

} // namespace sycltla

TORCH_LIBRARY(sycltla_conv, m) {
  m.def(
      "conv2d_fprop(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor");
  m.def(
      "conv2d_dgrad(Tensor grad_output, Tensor weight, int[] input_size, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor");
  m.def(
      "conv2d_wgrad(Tensor grad_output, Tensor input, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor");
  m.def(
      "conv3d_fprop(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor");
  m.def(
      "conv3d_dgrad(Tensor grad_output, Tensor weight, int[] input_size, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor");
  m.def(
      "conv3d_wgrad(Tensor grad_output, Tensor input, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor");
}

TORCH_LIBRARY_IMPL(sycltla_conv, XPU, m) {
  m.impl("conv2d_fprop", &sycltla::conv2d_fprop);
  m.impl("conv2d_dgrad", &sycltla::conv2d_dgrad);
  m.impl("conv2d_wgrad", &sycltla::conv2d_wgrad);
  m.impl("conv3d_fprop", &sycltla::conv3d_fprop);
  m.impl("conv3d_dgrad", &sycltla::conv3d_dgrad);
  m.impl("conv3d_wgrad", &sycltla::conv3d_wgrad);
}
