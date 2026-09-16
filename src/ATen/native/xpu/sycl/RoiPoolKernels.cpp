/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions of this file are derived from Torchvision
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <comm/Macros.h>
// clang-format off
DISABLE_RETURN_TYPE_WARNING_BEGIN
// clang-format on
#include <ATen/OpMathType.h>
#include <ATen/ceil_div.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <comm/SYCLContext.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/RoiPoolKernels.h>

namespace at::native::xpu {

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void roi_pool_forward_kernel_fn(
    int nthreads,
    const T* input,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    const T* rois,
    T* output,
    int* argmax) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  XPU_KERNEL_LOOP(item, index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    int roi_start_w = std::round(offset_rois[1] * spatial_scale);
    int roi_start_h = std::round(offset_rois[2] * spatial_scale);
    int roi_end_w = std::round(offset_rois[3] * spatial_scale);
    int roi_end_h = std::round(offset_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    using opmath_t = at::opmath_type<T>;
    int hstart = static_cast<int>(
        sycl::floor(static_cast<opmath_t>(static_cast<T>(ph) * bin_size_h)));
    int wstart = static_cast<int>(
        sycl::floor(static_cast<opmath_t>(static_cast<T>(pw) * bin_size_w)));
    int hend = static_cast<int>(sycl::ceil(
        static_cast<opmath_t>(ph + 1) * static_cast<opmath_t>(bin_size_h)));
    int wend = static_cast<int>(sycl::ceil(
        static_cast<opmath_t>(pw + 1) * static_cast<opmath_t>(bin_size_w)));

    // Add roi offsets and clip to input boundaries
    hstart = sycl::clamp(hstart + roi_start_h, 0, height);
    hend = sycl::clamp(hend + roi_start_h, 0, height);
    wstart = sycl::clamp(wstart + roi_start_w, 0, width);
    wend = sycl::clamp(wend + roi_start_w, 0, width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    T maxval = is_empty ? 0.0 : std::numeric_limits<float>::lowest();
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    const T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int input_index = h * width + w;
        if (offset_input[input_index] > maxval) {
          maxval = offset_input[input_index];
          maxidx = input_index;
        }
      }
    }
    output[index] = maxval;
    argmax[index] = maxidx;
  }
}

std::tuple<Tensor, Tensor> roi_pool_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  at::Tensor output = at::zeros(
      {num_rois, channels, pooled_height, pooled_width}, input.options());
  at::Tensor argmax = at::zeros(
      {num_rois, channels, pooled_height, pooled_width},
      input.options().dtype(at::kInt));

  auto output_size = num_rois * pooled_height * pooled_width * channels;
  int64_t global_range =
      xpuKernelLoopGroupRange(static_cast<int64_t>(output_size), 512);
  int64_t local_range = 512;

  if (output.numel() == 0) {
    return std::make_tuple(output, argmax);
  }

  auto input_ = input.contiguous();
  auto rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "roi_pool_forward_kernel_xpu", [&] {
        constexpr auto kptr = roi_pool_forward_kernel_fn<scalar_t>;
        sycl_kernel_submit<kptr>(
            global_range * local_range,
            local_range,
            at::xpu::getCurrentSYCLQueue(),
            0,
            output_size,
            input_.const_data_ptr<scalar_t>(),
            static_cast<scalar_t>(spatial_scale),
            static_cast<int>(channels),
            static_cast<int>(height),
            static_cast<int>(width),
            static_cast<int>(pooled_height),
            static_cast<int>(pooled_width),
            rois_.const_data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            argmax.data_ptr<int>());
      });
  return std::make_tuple(output, argmax);
}

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void roi_pool_backward_kernel_fn(
    int nthreads,
    const T* grad_output,
    const int* argmax_data,
    int num_rois,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    T* grad_input,
    const T* rois,
    int n_stride,
    int c_stride,
    int h_stride,
    int w_stride) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  XPU_KERNEL_LOOP(item, index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    const int output_offset = n * n_stride + c * c_stride;
    const int* argmax_data_offset =
        argmax_data + (n * channels + c) * pooled_height * pooled_width;
    const int argmax = argmax_data_offset[ph * pooled_width + pw];
    const int offset = (roi_batch_ind * channels + c) * height * width;

    if (argmax != -1) {
      atomicAdd(
          (sycl_global_ptr<T>)(grad_input + offset + argmax),
          static_cast<T>(
              grad_output[output_offset + ph * h_stride + pw * w_stride]));
    }
  }
}

Tensor roi_pool_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width) {
  at::Tensor grad_input =
      at::zeros({batch_size, channels, height, width}, grad.options());
  int64_t global_range =
      xpuKernelLoopGroupRange(static_cast<int64_t>(grad.numel()), 512);
  int64_t local_range = 512;

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    return grad_input;
  }

  int n_stride = grad.stride(0);
  int c_stride = grad.stride(1);
  int h_stride = grad.stride(2);
  int w_stride = grad.stride(3);

  auto num_rois = rois.size(0);
  auto argmax_ = argmax.contiguous();
  auto rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "roi_pool_backward_kernel_xpu", [&] {
        constexpr auto kptr = roi_pool_backward_kernel_fn<scalar_t>;
        sycl_kernel_submit<kptr>(
            global_range * local_range,
            local_range,
            at::xpu::getCurrentSYCLQueue(),
            0,
            static_cast<int>(grad.numel()),
            grad.const_data_ptr<scalar_t>(),
            argmax_.const_data_ptr<int>(),
            static_cast<int>(num_rois),
            static_cast<scalar_t>(spatial_scale),
            static_cast<int>(channels),
            static_cast<int>(height),
            static_cast<int>(width),
            static_cast<int>(pooled_height),
            static_cast<int>(pooled_width),
            grad_input.data_ptr<scalar_t>(),
            rois_.const_data_ptr<scalar_t>(),
            n_stride,
            c_stride,
            h_stride,
            w_stride);
      });
  return grad_input;
}

} // namespace at::native::xpu

// clang-format off
DISABLE_RETURN_TYPE_WARNING_END
// clang-format on
