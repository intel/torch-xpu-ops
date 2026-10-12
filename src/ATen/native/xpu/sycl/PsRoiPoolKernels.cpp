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

#include <ATen/native/xpu/sycl/PsRoiPoolKernels.h>

namespace at::native::xpu {

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void ps_roi_pool_forward_kernel_impl(
    int nthreads,
    const T* input,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    const T* rois,
    int channels_out,
    T* output,
    int* channel_mapping) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  XPU_KERNEL_LOOP(item, index, nthreads) {
    // (n, c_out, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c_out = (index / pooled_width / pooled_height) % channels_out;
    int n = index / pooled_width / pooled_height / channels_out;

    // (n, c_in, ph, pw) is the associated element in the input
    int c_in = (c_out * pooled_height + ph) * pooled_width + pw;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    int roi_start_w = std::round(offset_rois[1] * spatial_scale);
    int roi_start_h = std::round(offset_rois[2] * spatial_scale);
    int roi_end_w = std::round(offset_rois[3] * spatial_scale);
    int roi_end_h = std::round(offset_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = std::max(roi_end_w - roi_start_w, 1);
    int roi_height = std::max(roi_end_h - roi_start_h, 1);
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
    hstart = sycl::clamp(hstart + roi_start_h, 0, height - 1);
    hend = sycl::clamp(hend + roi_start_h, 0, height - 1);
    wstart = sycl::clamp(wstart + roi_start_w, 0, width - 1);
    wend = sycl::clamp(wend + roi_start_w, 0, width - 1);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    const T* offset_input =
        input + (roi_batch_ind * channels + c_in) * height * width;
    T out_sum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int input_index = h * width + w;
        out_sum += offset_input[input_index];
      }
    }

    T bin_area = (hend - hstart) * (wend - wstart);
    output[index] = is_empty ? static_cast<T>(0) : out_sum / bin_area;
    channel_mapping[index] = c_in;
  }
}

std::tuple<Tensor, Tensor> ps_roi_pool_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  TORCH_CHECK(
      channels % (pooled_height * pooled_width) == 0,
      "input channels must be a multiple of pooling height * pooling width");
  int channels_out = channels / (pooled_height * pooled_width);

  at::Tensor output = at::zeros(
      {num_rois, channels_out, pooled_height, pooled_width}, input.options());
  at::Tensor channel_mapping =
      at::zeros(output.sizes(), input.options().dtype(at::kInt));

  auto output_size = output.numel();
  int64_t global_range =
      xpuKernelLoopGroupRange(static_cast<int64_t>(output_size), 512);
  int64_t local_range = 512;

  if (output_size == 0) {
    return std::make_tuple(output, channel_mapping);
  }

  auto input_ = input.contiguous();
  auto rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "ps_roi_pool_forward_kernel_xpu", [&] {
        sycl_kernel_submit<ps_roi_pool_forward_kernel_impl<scalar_t>>(
            global_range * local_range,
            local_range,
            at::xpu::getCurrentSYCLQueue(),
            0,
            output_size,
            input_.const_data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            rois_.const_data_ptr<scalar_t>(),
            channels_out,
            output.data_ptr<scalar_t>(),
            channel_mapping.data_ptr<int>());
      });
  return std::make_tuple(output, channel_mapping);
}

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void ps_roi_pool_backward_kernel_impl(
    int nthreads,
    const T* grad_output,
    const int* channel_mapping,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int channels_out,
    T* grad_input,
    const T* rois) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  XPU_KERNEL_LOOP(item, index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int n = index / pooled_width / pooled_height / channels_out;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    int roi_start_w = std::roundf(offset_rois[1] * spatial_scale);
    int roi_start_h = std::roundf(offset_rois[2] * spatial_scale);
    int roi_end_w = std::roundf(offset_rois[3] * spatial_scale);
    int roi_end_h = std::roundf(offset_rois[4] * spatial_scale);

    // Force too small ROIs to be 1x1
    int roi_width = std::max(roi_end_w - roi_start_w, 1);
    int roi_height = std::max(roi_end_h - roi_start_h, 1);
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

    int c_in = channel_mapping[index];
    T bin_area = (hend - hstart) * (wend - wstart);
    T diff_val = is_empty ? static_cast<T>(0) : grad_output[index] / bin_area;

    const int offset = (roi_batch_ind * channels + c_in) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int grad_input_index = h * width + w;
        atomicAdd(
            (sycl_global_ptr<T>)(grad_input + offset + grad_input_index),
            static_cast<T>(diff_val));
      }
    }
  }
}

Tensor ps_roi_pool_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& channel_mapping,
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

  int channels_out = channels / (pooled_height * pooled_width);

  auto grad_ = grad.contiguous();
  auto rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "ps_roi_pool_backward_kernel_xpu", [&] {
        sycl_kernel_submit<ps_roi_pool_backward_kernel_impl<scalar_t>>(
            global_range * local_range,
            local_range,
            at::xpu::getCurrentSYCLQueue(),
            0,
            grad.numel(),
            grad_.const_data_ptr<scalar_t>(),
            channel_mapping.const_data_ptr<int>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            channels_out,
            grad_input.data_ptr<scalar_t>(),
            rois_.const_data_ptr<scalar_t>());
      });
  return grad_input;
}

} // namespace at::native::xpu

// clang-format off
DISABLE_RETURN_TYPE_WARNING_END
// clang-format on
