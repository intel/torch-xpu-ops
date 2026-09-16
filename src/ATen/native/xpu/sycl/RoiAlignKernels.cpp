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

#include <ATen/xpu/XPUContext.h>
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

#include <ATen/native/xpu/sycl/RoiAlignKernels.h>

namespace at::native::xpu {

template <typename T>
T bilinear_interpolate(
    const T* input,
    int height,
    int width,
    T y,
    T x,
    int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  y = std::max(T(0), y);
  x = std::max(T(0), x);

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  y_low = std::min(height - 1, y_low);
  x_low = std::min(width - 1, x_low);
  y_high = std::min(y_low + 1, height - 1);
  x_high = std::min(x_low + 1, width - 1);

  if (y_low == height - 1) {
    y = (T)y_low;
  }

  if (x_low == width - 1) {
    x = (T)x_low;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // do bilinear interpolation
  T v1 = input[y_low * width + x_low];
  T v2 = input[y_low * width + x_high];
  T v3 = input[y_high * width + x_low];
  T v4 = input[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}
template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void roi_align_forward_kernel_fn(
    const T* input,
    const T spatial_scale,
    int items_per_roi,
    int wgs_per_roi,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int sampling_ratio,
    bool aligned,
    const T* rois,
    T* output) {
  auto item = syclext::this_work_item::get_nd_item<1>();

  // each roi will have 5 values, batch_idx,x1,y1,x2,y2
  constexpr int roi_size = 5;
  syclexp::work_group_static<T[5]> cached_roi_local;
  T* cached_roi = &cached_roi_local[0];

  auto wg = item.get_group(0);
  int n = wg / wgs_per_roi;
  int output_index_on_batch_n =
      (wg - n * wgs_per_roi) * item.get_local_range(0) + item.get_local_id(0);
  const T* current_roi = rois + n * roi_size;
  if (item.get_local_id(0) == 0) {
    cached_roi[0] = current_roi[0];

    // Do not using rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;
    cached_roi[1] = current_roi[1] * spatial_scale - offset;
    cached_roi[2] = current_roi[2] * spatial_scale - offset;
    cached_roi[3] = current_roi[3] * spatial_scale - offset;
    cached_roi[4] = current_roi[4] * spatial_scale - offset;
  }
  sycl::group_barrier(item.get_group());

  if (output_index_on_batch_n < items_per_roi) {
    int pw = output_index_on_batch_n % pooled_width;
    int ph = (output_index_on_batch_n / pooled_width) % pooled_height;
    int c = (output_index_on_batch_n / pooled_width / pooled_height) % channels;

    int roi_batch_ind = cached_roi[0];
    T roi_start_w = cached_roi[1];
    T roi_start_h = cached_roi[2];
    T roi_end_w = cached_roi[3];
    T roi_end_h = cached_roi[4];

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    if (!aligned) {
      // Force malformed ROIs to be 1x1
      roi_width = std::max(roi_width, (T)1.);
      roi_height = std::max(roi_height, (T)1.);
    }

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    using opmath_t = at::opmath_type<T>;
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : static_cast<int>(sycl::ceil(
              static_cast<opmath_t>(roi_height) /
              static_cast<opmath_t>(pooled_height))); // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0)
        ? sampling_ratio
        : static_cast<int>(sycl::ceil(
              static_cast<opmath_t>(roi_width) /
              static_cast<opmath_t>(pooled_width)));

    // We do average (integral) pooling inside a bin
    // When the grid is empty, output zeros.
    const T count =
        std::max((int)(roi_bin_grid_h * roi_bin_grid_w), (int)(1)); // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate(
            offset_input,
            height,
            width,
            y,
            x,
            output_index_on_batch_n + n * items_per_roi);
        output_val += val;
      }
    }
    output_val /= count;

    output[output_index_on_batch_n + n * items_per_roi] = output_val;
  }
}

template <typename T>
void bilinear_interpolate_gradient(
    int height,
    int width,
    T y,
    T x,
    T& w1,
    T& w2,
    T& w3,
    T& w4,
    int& x_low,
    int& x_high,
    int& y_low,
    int& y_high,
    int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = input[y_low * width + x_low];
  // T v2 = input[y_low * width + x_high];
  // T v3 = input[y_high * width + x_low];
  // T v4 = input[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
}

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void roi_align_backward_kernel_fn(
    int nthreads,
    const T* grad_output,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int sampling_ratio,
    bool aligned,
    T* grad_input,
    const T* rois,
    int n_stride,
    int c_stride,
    int h_stride,
    int w_stride,
    const int memory_span) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  XPU_KERNEL_LOOP(item, index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    // Do not using rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;
    T roi_start_w = offset_rois[1] * spatial_scale - offset;
    T roi_start_h = offset_rois[2] * spatial_scale - offset;
    T roi_end_w = offset_rois[3] * spatial_scale - offset;
    T roi_end_h = offset_rois[4] * spatial_scale - offset;

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    if (!aligned) {
      // Force malformed ROIs to be 1x1
      roi_width = std::max(roi_width, (T)1.);
      roi_height = std::max(roi_height, (T)1.);
    }

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    // We need to index the gradient using the tensor strides to access the
    // correct values.
    const int output_offset = n * n_stride + c * c_stride;
    const T* offset_grad_output = grad_output + output_offset;
    const T grad_output_this_bin =
        offset_grad_output[ph * h_stride + pw * w_stride];

    // We use roi_bin_grid to sample the grid and mimic integral
    using opmath_t = at::opmath_type<T>;
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : static_cast<int>(sycl::ceil(
              static_cast<opmath_t>(roi_height) /
              static_cast<opmath_t>(pooled_height))); // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0)
        ? sampling_ratio
        : static_cast<int>(sycl::ceil(
              static_cast<opmath_t>(roi_width) /
              static_cast<opmath_t>(pooled_width)));

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    const int input_offset = (roi_batch_ind * channels + c) * height * width;

    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(
            height,
            width,
            y,
            x,
            w1,
            w2,
            w3,
            w4,
            x_low,
            x_high,
            y_low,
            y_high,
            index);

        T g1 = grad_output_this_bin * w1 / count;
        T g2 = grad_output_this_bin * w2 / count;
        T g3 = grad_output_this_bin * w3 / count;
        T g4 = grad_output_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(
              (sycl_global_ptr<T>)(grad_input + input_offset + y_low * width +
                                   x_low),
              static_cast<T>(g1));

          atomicAdd(
              (sycl_global_ptr<T>)(grad_input + input_offset + y_low * width +
                                   x_high),
              static_cast<T>(g2));
          atomicAdd(
              (sycl_global_ptr<T>)(grad_input + input_offset + y_high * width +
                                   x_low),
              static_cast<T>(g3));
          atomicAdd(
              (sycl_global_ptr<T>)(grad_input + input_offset + y_high * width +
                                   x_high),
              static_cast<T>(g4));
        } // if
      } // ix
    } // iy
  } // XPU_KERNEL_LOOP
}

Tensor roi_align_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  at::Tensor output = at::zeros(
      {num_rois, channels, pooled_height, pooled_width}, input.options());

  if (output.numel() == 0) {
    return output;
  }

  auto input_ = input.contiguous();
  auto rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      input.scalar_type(),
      "roi_align_forward_kernel_xpu",
      [&] {
        int64_t local_range =
            at::xpu::getKernelMaxWorkGroupSize<roi_align_forward_kernel_fn<scalar_t>>();
        int items_per_roi = pooled_height * pooled_width * channels;
        if (items_per_roi < local_range) {
          constexpr int simd_len = 32;
          local_range = std::min(
              local_range,
              int64_t(items_per_roi + simd_len - 1) / simd_len * simd_len);
        }
        int wgs_per_roi = (items_per_roi + local_range - 1) / local_range;
        int64_t global_range = wgs_per_roi * num_rois;
        sycl_kernel_submit<roi_align_forward_kernel_fn<scalar_t>>(
            global_range * local_range,
            local_range,
            at::xpu::getCurrentSYCLQueue(),
            0,
            input_.const_data_ptr<scalar_t>(),
            (scalar_t)spatial_scale,
            (int)items_per_roi,
            (int)wgs_per_roi,
            (int)channels,
            (int)height,
            (int)width,
            (int)pooled_height,
            (int)pooled_width,
            (int)sampling_ratio,
            aligned,
            rois_.const_data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>());
      });
  return output;
}

Tensor roi_align_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t sampling_ratio,
    bool aligned) {
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

  at::globalContext().alertNotDeterministic("roi_align_backward_kernel_xpu");

  auto rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      grad.scalar_type(),
      "roi_align_backward_kernel_xpu",
      [&] {
        sycl_kernel_submit<roi_align_backward_kernel_fn<scalar_t>>(
            global_range * local_range,
            local_range,
            at::xpu::getCurrentSYCLQueue(),
            0,
            (int)grad.numel(),
            grad.const_data_ptr<scalar_t>(),
            (scalar_t)spatial_scale,
            (int)channels,
            (int)height,
            (int)width,
            (int)pooled_height,
            (int)pooled_width,
            (int)sampling_ratio,
            aligned,
            grad_input.data_ptr<scalar_t>(),
            rois_.const_data_ptr<scalar_t>(),
            n_stride,
            c_stride,
            h_stride,
            w_stride,
            (int)grad_input.numel());
      });
  return grad_input;
}

} // namespace at::native::xpu

// clang-format off
DISABLE_RETURN_TYPE_WARNING_END
// clang-format on
