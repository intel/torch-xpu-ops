/*
 * Copyright 2020-2026 Intel Corporation
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

#include <ATen/AccumulateType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Pool.h>

#include <ATen/native/xpu/sycl/AveragePool2dKernels.h>
#include <ATen/native/xpu/sycl/IntegerDivider.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <comm/Runtime.h>
#include <comm/SYCLContext.h>
#include <comm/SYCLHelpers.h>

namespace at::native {
namespace xpu {

// Grid-stride cap for the vectorized channels-last kernels (VEC/WTILE/HWTILE).
// Bounding the launch to a large-but-finite multiple of the device's resident
// thread capacity keeps the grid size sane for huge inputs while preserving
// the oversubscription memory-bound kernels need to hide DRAM latency: a cap
// of 1x (no oversubscription, matching the scalar fallback kernels' cap)
// regresses small-spatial/large-batch shapes by up to 1.86x in benchmarking;
// 32x recovers that (and is within noise of fully uncapped on all shapes
// tested) while still bounding the launch instead of scaling unbounded with
// input size.
static int64_t vectorized_grid_stride_cap() {
  return syclMaxWorkItemsPerTile() * 32;
}

template <typename scalar_t, typename accscalar_t, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void avg_pool2d_kernel_impl(
    scalar_t* top_data,
    const scalar_t* bottom_data,
    const int total_elements,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int divisor_override,
    bool count_include_pad,
    bool use_divisor) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  XPU_KERNEL_LOOP(item, index, total_elements) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;

    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = sycl::min(hstart + kernel_h, static_cast<int>(height + pad_h));
    int wend = sycl::min(wstart + kernel_w, static_cast<int>(width + pad_w));
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = sycl::max(hstart, 0);
    wstart = sycl::max(wstart, 0);
    hend = sycl::min(hend, static_cast<int>(height));
    wend = sycl::min(wend, static_cast<int>(width));

    if (hstart >= hend || wstart >= wend) {
      top_data[index] = scalar_t(0);
      return;
    }

    accscalar_t aveval = accscalar_t(0);
    const scalar_t* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;

    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    int divide_factor;
    if (use_divisor) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (hend - hstart) * (wend - wstart);
      }
    }
    top_data[index] = static_cast<scalar_t>(aveval / divide_factor);
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void avg_pool2d_channels_last_kernel_impl(
    scalar_t* top_data,
    const scalar_t* bottom_data,
    index_t total_elements,
    index_t channels,
    index_t height,
    index_t width,
    int pooled_height,
    int pooled_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int divisor_override,
    bool count_include_pad,
    bool use_divisor,
    at::detail::IntDivider<unsigned int> div_channels,
    at::detail::IntDivider<unsigned int> div_pooled_width,
    at::detail::IntDivider<unsigned int> div_pooled_height) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  XPU_KERNEL_LOOP(item, index, total_elements) {
    // Magic-number division replaces chained %/`/` for the index
    // decomposition below.
    auto dm_c = div_channels.divmod(index);
    const int c = dm_c.mod;
    auto dm_w = div_pooled_width.divmod(dm_c.div);
    const int pw = dm_w.mod;
    auto dm_h = div_pooled_height.divmod(dm_w.div);
    const int ph = dm_h.mod;
    const int n = dm_h.div;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = sycl::min(hstart + kernel_h, static_cast<int>(height + pad_h));
    int wend = sycl::min(wstart + kernel_w, static_cast<int>(width + pad_w));
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = sycl::max(hstart, 0);
    wstart = sycl::max(wstart, 0);
    hend = sycl::min(hend, static_cast<int>(height));
    wend = sycl::min(wend, static_cast<int>(width));

    if (hstart >= hend || wstart >= wend) {
      top_data[index] = scalar_t(0);
      return;
    }

    accscalar_t aveval = accscalar_t(0);
    const scalar_t* const bottom_slice =
        bottom_data + n * channels * height * width + c;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[(h * width + w) * channels];
      }
    }
    int divide_factor;
    if (use_divisor) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (hend - hstart) * (wend - wstart);
      }
    }
    top_data[index] = static_cast<scalar_t>(aveval / divide_factor);
  }
}

// Each work-item processes vec_size channels at once using aligned_vector
// loads.
template <
    typename scalar_t,
    typename accscalar_t,
    typename vec_t,
    int vec_size,
    typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void avg_pool2d_channels_last_vec_kernel_impl(
    vec_t* output_vec,
    const vec_t* input_vec,
    index_t total_elements,
    index_t num_vec_channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int divisor_override,
    bool count_include_pad,
    bool use_divisor,
    at::detail::IntDivider<unsigned int> div_num_vec_channels,
    at::detail::IntDivider<unsigned int> div_pooled_width,
    at::detail::IntDivider<unsigned int> div_pooled_height,
    index_t nvc_mask,
    int nvc_shift) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  // total_elements = N * oH * oW * (C / vec_size)
  for (index_t idx = item.get_global_linear_id(); idx < total_elements;
       idx += item.get_local_range(0) * item.get_group_range(0)) {
    // Decompose: idx -> (plane, pw, ph, n)
    // Use bitshift for power-of-2 num_vec_channels, else IntDivider
    index_t plane, spatial_idx;
    if (nvc_shift > 0) {
      plane = idx & nvc_mask;
      spatial_idx = idx >> nvc_shift;
    } else {
      auto dm_c = div_num_vec_channels.divmod(idx);
      plane = dm_c.mod;
      spatial_idx = dm_c.div;
    }
    auto dm_w = div_pooled_width.divmod(spatial_idx);
    const index_t pw = dm_w.mod;
    auto dm_h = div_pooled_height.divmod(dm_w.div);
    const index_t ph = dm_h.mod;
    const index_t n = dm_h.div;

    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = sycl::min(hstart + kernel_h, height + pad_h);
    int wend = sycl::min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = sycl::max(hstart, 0);
    wstart = sycl::max(wstart, 0);
    hend = sycl::min(hend, height);
    wend = sycl::min(wend, width);

    if (hstart >= hend || wstart >= wend) {
      vec_t zero_vec;
#pragma unroll
      for (int i = 0; i < vec_size; i++)
        zero_vec.val[i] = scalar_t(0);
      output_vec[idx] = zero_vec;
      continue;
    }

    accscalar_t aveval[vec_size];

    // input layout: N, H, W, C (channels-last)
    // input_vec layout: N, H, W, C/vec_size
    const index_t batch_offset = n * height * width * num_vec_channels;
    const index_t row_stride = width * num_vec_channels;

    // Fast path: interior pixels with full 3x3 window.
    // All 9 loads issued upfront so HW memory pipeline can overlap them.
    if (kernel_h == 3 && kernel_w == 3 && (hend - hstart) == 3 &&
        (wend - wstart) == 3) {
      const index_t base = batch_offset + hstart * row_stride +
          wstart * num_vec_channels + plane;
      vec_t v00 = input_vec[base];
      vec_t v01 = input_vec[base + num_vec_channels];
      vec_t v02 = input_vec[base + 2 * num_vec_channels];
      vec_t v10 = input_vec[base + row_stride];
      vec_t v11 = input_vec[base + row_stride + num_vec_channels];
      vec_t v12 = input_vec[base + row_stride + 2 * num_vec_channels];
      vec_t v20 = input_vec[base + 2 * row_stride];
      vec_t v21 = input_vec[base + 2 * row_stride + num_vec_channels];
      vec_t v22 = input_vec[base + 2 * row_stride + 2 * num_vec_channels];
#pragma unroll
      for (int i = 0; i < vec_size; i++) {
        aveval[i] = static_cast<accscalar_t>(v00.val[i]) +
            static_cast<accscalar_t>(v01.val[i]) +
            static_cast<accscalar_t>(v02.val[i]) +
            static_cast<accscalar_t>(v10.val[i]) +
            static_cast<accscalar_t>(v11.val[i]) +
            static_cast<accscalar_t>(v12.val[i]) +
            static_cast<accscalar_t>(v20.val[i]) +
            static_cast<accscalar_t>(v21.val[i]) +
            static_cast<accscalar_t>(v22.val[i]);
      }
    } else if (
        kernel_h == 2 && kernel_w == 2 && (hend - hstart) == 2 &&
        (wend - wstart) == 2) {
      // Fast path: full 2x2 window (k2s2 interior pixels).
      // All 4 loads issued upfront for memory pipeline overlap.
      const index_t base = batch_offset + hstart * row_stride +
          wstart * num_vec_channels + plane;
      vec_t v00 = input_vec[base];
      vec_t v01 = input_vec[base + num_vec_channels];
      vec_t v10 = input_vec[base + row_stride];
      vec_t v11 = input_vec[base + row_stride + num_vec_channels];
#pragma unroll
      for (int i = 0; i < vec_size; i++) {
        aveval[i] = static_cast<accscalar_t>(v00.val[i]) +
            static_cast<accscalar_t>(v01.val[i]) +
            static_cast<accscalar_t>(v10.val[i]) +
            static_cast<accscalar_t>(v11.val[i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < vec_size; i++)
        aveval[i] = accscalar_t(0);
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          index_t load_idx =
              batch_offset + (h * width + w) * num_vec_channels + plane;
          vec_t val = input_vec[load_idx];
#pragma unroll
          for (int i = 0; i < vec_size; i++) {
            aveval[i] += static_cast<accscalar_t>(val.val[i]);
          }
        }
      }
    }

    int divide_factor;
    if (use_divisor) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (hend - hstart) * (wend - wstart);
      }
    }

    vec_t out_vec;
#pragma unroll
    for (int i = 0; i < vec_size; i++) {
      out_vec.val[i] = static_cast<scalar_t>(aveval[i] / divide_factor);
    }
    output_vec[idx] = out_vec;
  }
}

// W-tile kernel: each work-item processes 2 adjacent output positions in W
// direction, sharing overlapping input columns for k=3 s=1.
// Reduces total loads by 33% (12 loads for 2 outputs vs 18 independently).
template <
    typename scalar_t,
    typename accscalar_t,
    typename vec_t,
    int vec_size,
    typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void avg_pool2d_channels_last_vec_wtile_kernel_impl(
    vec_t* output_vec,
    const vec_t* input_vec,
    index_t total_elements,
    index_t num_vec_channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int stride_h,
    int pad_h,
    int pad_w,
    int divisor_override,
    bool count_include_pad,
    bool use_divisor,
    at::detail::IntDivider<unsigned int> div_num_vec_channels,
    at::detail::IntDivider<unsigned int> div_pooled_width_half,
    at::detail::IntDivider<unsigned int> div_pooled_height,
    index_t nvc_mask,
    int nvc_shift) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  // total_elements = N * oH * ceil(oW/2) * num_vec_channels
  for (index_t idx = item.get_global_linear_id(); idx < total_elements;
       idx += item.get_local_range(0) * item.get_group_range(0)) {
    // Index decomposition: idx -> (plane, pw_pair, ph, n)
    index_t plane, spatial_idx;
    if (nvc_shift > 0) {
      plane = idx & nvc_mask;
      spatial_idx = idx >> nvc_shift;
    } else {
      auto dm_c = div_num_vec_channels.divmod(idx);
      plane = dm_c.mod;
      spatial_idx = dm_c.div;
    }
    auto dm_w = div_pooled_width_half.divmod(spatial_idx);
    const index_t pw_pair = dm_w.mod;
    auto dm_h = div_pooled_height.divmod(dm_w.div);
    const index_t ph = dm_h.mod;
    const index_t n = dm_h.div;

    const int pw0 = pw_pair * 2;
    const int pw1 = pw0 + 1;

    // Compute H bounds (same for both outputs)
    int hstart = ph * stride_h - pad_h;
    int hend = sycl::min(hstart + 3, height + pad_h);
    const int pool_h_full = hend - hstart;
    hstart = sycl::max(hstart, 0);
    hend = sycl::min(hend, height);

    // Compute W bounds for pw0
    int wstart0 = pw0 - pad_w;
    int wend0 = sycl::min(wstart0 + 3, width + pad_w);
    const int pool_w0_full = wend0 - wstart0;
    wstart0 = sycl::max(wstart0, 0);
    wend0 = sycl::min(wend0, width);

    // Compute W bounds for pw1
    int wstart1 = pw1 - pad_w;
    int wend1 = sycl::min(wstart1 + 3, width + pad_w);
    const int pool_w1_full = wend1 - wstart1;
    wstart1 = sycl::max(wstart1, 0);
    wend1 = sycl::min(wend1, width);

    const index_t batch_offset = n * height * width * num_vec_channels;

    accscalar_t acc0[vec_size], acc1[vec_size];

    // Fast path: both outputs have full 3x3 windows (interior pixels).
    // Unroll all 12 loads (3 rows x 4 cols) upfront.
    if ((hend - hstart) == 3 && (wend0 - wstart0) == 3 &&
        (wend1 - wstart1) == 3) {
      const index_t row_stride = width * num_vec_channels;
      const index_t base = batch_offset + hstart * row_stride +
          wstart0 * num_vec_channels + plane;
      // 3 rows, 4 columns: wstart0, wstart0+1, wstart0+2, wstart0+3
      // wstart0+0 = unique to pw0
      // wstart0+1, wstart0+2 = shared (pw0 and pw1)
      // wstart0+3 = unique to pw1
      vec_t r0c0 = input_vec[base];
      vec_t r0c1 = input_vec[base + num_vec_channels];
      vec_t r0c2 = input_vec[base + 2 * num_vec_channels];
      vec_t r0c3 = input_vec[base + 3 * num_vec_channels];
      vec_t r1c0 = input_vec[base + row_stride];
      vec_t r1c1 = input_vec[base + row_stride + num_vec_channels];
      vec_t r1c2 = input_vec[base + row_stride + 2 * num_vec_channels];
      vec_t r1c3 = input_vec[base + row_stride + 3 * num_vec_channels];
      vec_t r2c0 = input_vec[base + 2 * row_stride];
      vec_t r2c1 = input_vec[base + 2 * row_stride + num_vec_channels];
      vec_t r2c2 = input_vec[base + 2 * row_stride + 2 * num_vec_channels];
      vec_t r2c3 = input_vec[base + 2 * row_stride + 3 * num_vec_channels];
#pragma unroll
      for (int i = 0; i < vec_size; i++) {
        accscalar_t c0 = static_cast<accscalar_t>(r0c0.val[i]) +
            static_cast<accscalar_t>(r1c0.val[i]) +
            static_cast<accscalar_t>(r2c0.val[i]);
        accscalar_t c1 = static_cast<accscalar_t>(r0c1.val[i]) +
            static_cast<accscalar_t>(r1c1.val[i]) +
            static_cast<accscalar_t>(r2c1.val[i]);
        accscalar_t c2 = static_cast<accscalar_t>(r0c2.val[i]) +
            static_cast<accscalar_t>(r1c2.val[i]) +
            static_cast<accscalar_t>(r2c2.val[i]);
        accscalar_t c3 = static_cast<accscalar_t>(r0c3.val[i]) +
            static_cast<accscalar_t>(r1c3.val[i]) +
            static_cast<accscalar_t>(r2c3.val[i]);
        acc0[i] = c0 + c1 + c2; // cols 0,1,2
        acc1[i] = c1 + c2 + c3; // cols 1,2,3
      }
    } else {
      // Generic path for edge pixels
#pragma unroll
      for (int i = 0; i < vec_size; i++) {
        acc0[i] = accscalar_t(0);
        acc1[i] = accscalar_t(0);
      }

      // Shared and unique column ranges for data reuse
      const int shared_wstart = sycl::max(wstart0, wstart1);
      const int shared_wend = sycl::min(wend0, wend1);

      for (int h = hstart; h < hend; ++h) {
        const index_t row_base = batch_offset + h * width * num_vec_channels;

        // Columns unique to pw0 (left of shared region)
        for (int w = wstart0; w < shared_wstart; ++w) {
          vec_t val = input_vec[row_base + w * num_vec_channels + plane];
#pragma unroll
          for (int i = 0; i < vec_size; i++)
            acc0[i] += static_cast<accscalar_t>(val.val[i]);
        }

        // Shared columns (add to both accumulators)
        for (int w = shared_wstart; w < shared_wend; ++w) {
          vec_t val = input_vec[row_base + w * num_vec_channels + plane];
#pragma unroll
          for (int i = 0; i < vec_size; i++) {
            acc0[i] += static_cast<accscalar_t>(val.val[i]);
            acc1[i] += static_cast<accscalar_t>(val.val[i]);
          }
        }

        // Columns unique to pw1 (right of shared region)
        for (int w = shared_wend; w < wend1; ++w) {
          vec_t val = input_vec[row_base + w * num_vec_channels + plane];
#pragma unroll
          for (int i = 0; i < vec_size; i++)
            acc1[i] += static_cast<accscalar_t>(val.val[i]);
        }
      }
    }

    // Compute divide factors
    int divide0, divide1;
    if (use_divisor) {
      divide0 = divide1 = divisor_override;
    } else if (count_include_pad) {
      divide0 = pool_h_full * pool_w0_full;
      divide1 = pool_h_full * pool_w1_full;
    } else {
      divide0 = (hend - hstart) * (wend0 - wstart0);
      divide1 = (hend - hstart) * (wend1 - wstart1);
    }

    // Write output for pw0
    const index_t out_idx0 =
        n * pooled_height * pooled_width * num_vec_channels +
        ph * pooled_width * num_vec_channels + pw0 * num_vec_channels + plane;
    vec_t out0;
#pragma unroll
    for (int i = 0; i < vec_size; i++)
      out0.val[i] = static_cast<scalar_t>(acc0[i] / divide0);
    output_vec[out_idx0] = out0;

    // Write output for pw1 (bounds check for odd pooled_width)
    if (pw1 < pooled_width) {
      const index_t out_idx1 = out_idx0 + num_vec_channels;
      vec_t out1;
#pragma unroll
      for (int i = 0; i < vec_size; i++)
        out1.val[i] = static_cast<scalar_t>(acc1[i] / divide1);
      output_vec[out_idx1] = out1;
    }
  }
}

// HW-tile kernel: each work-item processes a 2x2 block of adjacent outputs
// (ph0,pw0), (ph0,pw1), (ph1,pw0), (ph1,pw1) for k=3 s=1.
// Loads a 4x4 input grid (16 loads) shared across 4 outputs = 4 loads/output
// vs 9 in flat. Used only for vec_size==2 shapes where WTile subgroup
// alignment is poor (num_vc not a multiple of subgroup width).
template <
    typename scalar_t,
    typename accscalar_t,
    typename vec_t,
    int vec_size,
    typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void avg_pool2d_channels_last_vec_hwtile_kernel_impl(
    vec_t* output_vec,
    const vec_t* input_vec,
    index_t total_elements,
    index_t num_vec_channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int stride_h,
    int pad_h,
    int pad_w,
    int divisor_override,
    bool count_include_pad,
    bool use_divisor,
    at::detail::IntDivider<unsigned int> div_num_vec_channels,
    at::detail::IntDivider<unsigned int> div_pooled_width_half,
    at::detail::IntDivider<unsigned int> div_pooled_height_half,
    index_t nvc_mask,
    int nvc_shift) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  // total_elements = N * ceil(oH/2) * ceil(oW/2) * num_vec_channels
  for (index_t idx = item.get_global_linear_id(); idx < total_elements;
       idx += item.get_local_range(0) * item.get_group_range(0)) {
    index_t plane, spatial_idx;
    if (nvc_shift > 0) {
      plane = idx & nvc_mask;
      spatial_idx = idx >> nvc_shift;
    } else {
      auto dm_c = div_num_vec_channels.divmod(idx);
      plane = dm_c.mod;
      spatial_idx = dm_c.div;
    }
    auto dm_w = div_pooled_width_half.divmod(spatial_idx);
    const index_t pw_pair = dm_w.mod;
    auto dm_h = div_pooled_height_half.divmod(dm_w.div);
    const index_t ph_pair = dm_h.mod;
    const index_t n = dm_h.div;

    const int pw0 = pw_pair * 2;
    const int pw1 = pw0 + 1;
    const int ph0 = ph_pair * 2;
    const int ph1 = ph0 + 1;

    int hstart0 = ph0 * stride_h - pad_h;
    int hend0 = sycl::min(hstart0 + 3, height + pad_h);
    const int pool_h0_full = hend0 - hstart0;
    hstart0 = sycl::max(hstart0, 0);
    hend0 = sycl::min(hend0, height);

    int hstart1 = ph1 * stride_h - pad_h;
    int hend1 = sycl::min(hstart1 + 3, height + pad_h);
    const int pool_h1_full = hend1 - hstart1;
    hstart1 = sycl::max(hstart1, 0);
    hend1 = sycl::min(hend1, height);

    int wstart0 = pw0 - pad_w;
    int wend0 = sycl::min(wstart0 + 3, width + pad_w);
    const int pool_w0_full = wend0 - wstart0;
    wstart0 = sycl::max(wstart0, 0);
    wend0 = sycl::min(wend0, width);

    int wstart1 = pw1 - pad_w;
    int wend1 = sycl::min(wstart1 + 3, width + pad_w);
    const int pool_w1_full = wend1 - wstart1;
    wstart1 = sycl::max(wstart1, 0);
    wend1 = sycl::min(wend1, width);

    const index_t batch_offset = n * height * width * num_vec_channels;

    accscalar_t acc00[vec_size], acc01[vec_size];
    accscalar_t acc10[vec_size], acc11[vec_size];

    // Fast path: all 4 outputs are interior (full 3x3 window, ph1 valid).
    // Load a 4x4 grid (16 loads), then combine with column partial sums.
    if (ph1 < pooled_height && (hend0 - hstart0) == 3 &&
        (hend1 - hstart1) == 3 && (wend0 - wstart0) == 3 &&
        (wend1 - wstart1) == 3) {
      const index_t row_stride = width * num_vec_channels;
      const index_t base = batch_offset + hstart0 * row_stride +
          wstart0 * num_vec_channels + plane;
      vec_t r0c0 = input_vec[base];
      vec_t r0c1 = input_vec[base + num_vec_channels];
      vec_t r0c2 = input_vec[base + 2 * num_vec_channels];
      vec_t r0c3 = input_vec[base + 3 * num_vec_channels];
      vec_t r1c0 = input_vec[base + row_stride];
      vec_t r1c1 = input_vec[base + row_stride + num_vec_channels];
      vec_t r1c2 = input_vec[base + row_stride + 2 * num_vec_channels];
      vec_t r1c3 = input_vec[base + row_stride + 3 * num_vec_channels];
      vec_t r2c0 = input_vec[base + 2 * row_stride];
      vec_t r2c1 = input_vec[base + 2 * row_stride + num_vec_channels];
      vec_t r2c2 = input_vec[base + 2 * row_stride + 2 * num_vec_channels];
      vec_t r2c3 = input_vec[base + 2 * row_stride + 3 * num_vec_channels];
      vec_t r3c0 = input_vec[base + 3 * row_stride];
      vec_t r3c1 = input_vec[base + 3 * row_stride + num_vec_channels];
      vec_t r3c2 = input_vec[base + 3 * row_stride + 2 * num_vec_channels];
      vec_t r3c3 = input_vec[base + 3 * row_stride + 3 * num_vec_channels];
#pragma unroll
      for (int i = 0; i < vec_size; i++) {
        // Column sums for rows 0..2 (used by ph0 outputs)
        accscalar_t c0_012 = static_cast<accscalar_t>(r0c0.val[i]) +
            static_cast<accscalar_t>(r1c0.val[i]) +
            static_cast<accscalar_t>(r2c0.val[i]);
        accscalar_t c1_012 = static_cast<accscalar_t>(r0c1.val[i]) +
            static_cast<accscalar_t>(r1c1.val[i]) +
            static_cast<accscalar_t>(r2c1.val[i]);
        accscalar_t c2_012 = static_cast<accscalar_t>(r0c2.val[i]) +
            static_cast<accscalar_t>(r1c2.val[i]) +
            static_cast<accscalar_t>(r2c2.val[i]);
        accscalar_t c3_012 = static_cast<accscalar_t>(r0c3.val[i]) +
            static_cast<accscalar_t>(r1c3.val[i]) +
            static_cast<accscalar_t>(r2c3.val[i]);
        // Column sums for rows 1..3 (used by ph1 outputs)
        accscalar_t c0_123 = static_cast<accscalar_t>(r1c0.val[i]) +
            static_cast<accscalar_t>(r2c0.val[i]) +
            static_cast<accscalar_t>(r3c0.val[i]);
        accscalar_t c1_123 = static_cast<accscalar_t>(r1c1.val[i]) +
            static_cast<accscalar_t>(r2c1.val[i]) +
            static_cast<accscalar_t>(r3c1.val[i]);
        accscalar_t c2_123 = static_cast<accscalar_t>(r1c2.val[i]) +
            static_cast<accscalar_t>(r2c2.val[i]) +
            static_cast<accscalar_t>(r3c2.val[i]);
        accscalar_t c3_123 = static_cast<accscalar_t>(r1c3.val[i]) +
            static_cast<accscalar_t>(r2c3.val[i]) +
            static_cast<accscalar_t>(r3c3.val[i]);
        acc00[i] = c0_012 + c1_012 + c2_012; // ph0,pw0: rows 0..2, cols 0..2
        acc01[i] = c1_012 + c2_012 + c3_012; // ph0,pw1: rows 0..2, cols 1..3
        acc10[i] = c0_123 + c1_123 + c2_123; // ph1,pw0: rows 1..3, cols 0..2
        acc11[i] = c1_123 + c2_123 + c3_123; // ph1,pw1: rows 1..3, cols 1..3
      }
    } else {
      // Generic slow path: each output computed independently.
#pragma unroll
      for (int i = 0; i < vec_size; i++)
        acc00[i] = acc01[i] = acc10[i] = acc11[i] = accscalar_t(0);

      for (int h = hstart0; h < hend0; ++h) {
        const index_t rbase =
            batch_offset + h * width * num_vec_channels + plane;
        for (int w = wstart0; w < wend0; ++w) {
          vec_t v = input_vec[rbase + w * num_vec_channels];
#pragma unroll
          for (int i = 0; i < vec_size; i++)
            acc00[i] += static_cast<accscalar_t>(v.val[i]);
        }
        for (int w = wstart1; w < wend1; ++w) {
          vec_t v = input_vec[rbase + w * num_vec_channels];
#pragma unroll
          for (int i = 0; i < vec_size; i++)
            acc01[i] += static_cast<accscalar_t>(v.val[i]);
        }
      }
      for (int h = hstart1; h < hend1; ++h) {
        const index_t rbase =
            batch_offset + h * width * num_vec_channels + plane;
        for (int w = wstart0; w < wend0; ++w) {
          vec_t v = input_vec[rbase + w * num_vec_channels];
#pragma unroll
          for (int i = 0; i < vec_size; i++)
            acc10[i] += static_cast<accscalar_t>(v.val[i]);
        }
        for (int w = wstart1; w < wend1; ++w) {
          vec_t v = input_vec[rbase + w * num_vec_channels];
#pragma unroll
          for (int i = 0; i < vec_size; i++)
            acc11[i] += static_cast<accscalar_t>(v.val[i]);
        }
      }
    }

    // Divide factors for ph0 outputs
    int div00, div01;
    if (use_divisor) {
      div00 = div01 = divisor_override;
    } else if (count_include_pad) {
      div00 = pool_h0_full * pool_w0_full;
      div01 = pool_h0_full * pool_w1_full;
    } else {
      div00 = (hend0 - hstart0) * (wend0 - wstart0);
      div01 = (hend0 - hstart0) * (wend1 - wstart1);
    }

    const index_t out_base =
        n * pooled_height * pooled_width * num_vec_channels +
        ph0 * pooled_width * num_vec_channels;
    const index_t out00 = out_base + pw0 * num_vec_channels + plane;

    vec_t o00;
#pragma unroll
    for (int i = 0; i < vec_size; i++)
      o00.val[i] = static_cast<scalar_t>(acc00[i] / div00);
    output_vec[out00] = o00;

    if (pw1 < pooled_width) {
      vec_t o01;
#pragma unroll
      for (int i = 0; i < vec_size; i++)
        o01.val[i] = static_cast<scalar_t>(acc01[i] / div01);
      output_vec[out00 + num_vec_channels] = o01;
    }

    // Write ph1 outputs only if ph1 is in bounds
    if (ph1 < pooled_height) {
      int div10, div11;
      if (use_divisor) {
        div10 = div11 = divisor_override;
      } else if (count_include_pad) {
        div10 = pool_h1_full * pool_w0_full;
        div11 = pool_h1_full * pool_w1_full;
      } else {
        div10 = (hend1 - hstart1) * (wend0 - wstart0);
        div11 = (hend1 - hstart1) * (wend1 - wstart1);
      }
      const index_t out10 = out_base + pooled_width * num_vec_channels +
          pw0 * num_vec_channels + plane;
      vec_t o10;
#pragma unroll
      for (int i = 0; i < vec_size; i++)
        o10.val[i] = static_cast<scalar_t>(acc10[i] / div10);
      output_vec[out10] = o10;

      if (pw1 < pooled_width) {
        vec_t o11;
#pragma unroll
        for (int i = 0; i < vec_size; i++)
          o11.val[i] = static_cast<scalar_t>(acc11[i] / div11);
        output_vec[out10 + num_vec_channels] = o11;
      }
    }
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
void launch_avg_pool2d_channels_last_kernel(
    const int total_elements,
    const Tensor& input,
    const index_t channels,
    const index_t height,
    const index_t width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const Tensor& output,
    const int divisor_override,
    const bool count_include_pad,
    const bool use_divisor) {
  scalar_t* top_data = output.mutable_data_ptr<scalar_t>();
  const scalar_t* bottom_data = input.const_data_ptr<scalar_t>();

  auto& queue = at::xpu::getCurrentSYCLQueue();
  const int64_t group_size =
      static_cast<int64_t>(syclMaxWorkItemsPerSubSlice());

  // Vectorized path: pick largest vec_size that (1) divides channels,
  // (2) maintains enough work-items for good EU occupancy, and
  // (3) passes alignment check.
  const index_t spatial_out =
      static_cast<index_t>(pooled_height) * static_cast<index_t>(pooled_width);
  const index_t batch_count = total_elements / (channels * spatial_out);
  // Device-derived occupancy floor instead of a fixed constant, so the
  // threshold scales with the actual GPU's EU/thread capacity. The divisor
  // is chosen to reproduce the empirically-tuned value on BMG exactly
  // (syclMaxWorkItemsPerTile() == 40960 on BMG, 40960 / 5 == 8192).
  const int64_t min_work_items = syclMaxWorkItemsPerTile() / 5;

  // Determine max vec_size from pointer alignment alone (bypass conservative
  // preferred_vector_width which caps at 4 on BMG). The hardware supports
  // wider loads; the preferred_width is a SYCL hint, not a hard limit.
  int max_vs = 8; // start at 8, we'll validate alignment
  {
    uint64_t addr_in = reinterpret_cast<uint64_t>(bottom_data);
    uint64_t addr_out = reinterpret_cast<uint64_t>(top_data);
    constexpr int align8 = sizeof(scalar_t) * 8; // 32 bytes for float
    constexpr int align4 = sizeof(scalar_t) * 4; // 16 bytes for float
    if (addr_in % align8 != 0 || addr_out % align8 != 0) {
      if (addr_in % align4 == 0 && addr_out % align4 == 0)
        max_vs = 4;
      else
        max_vs = 1;
    }
  }

  // For k=3,s=1 shapes that qualify for W-tile, prefer vec4 so W-tile can
  // activate (W-tile + vec4 is better than plain vec8 for large shapes due
  // to 33% load sharing). Only allow vec8 when W-tile won't be used.
  const bool wtile_eligible =
      (kernel_h == 3 && kernel_w == 3 && stride_w == 1 && pooled_width >= 2);
  if (wtile_eligible && max_vs > 4) {
    // Check if vec4 + W-tile would meet the wtile threshold
    const index_t num_vc4 = channels / 4;
    const index_t pw_half = (static_cast<index_t>(pooled_width) + 1) / 2;
    const index_t wtile_items_v4 =
        batch_count * static_cast<index_t>(pooled_height) * pw_half * num_vc4;
    if (wtile_items_v4 >= 8 * min_work_items) {
      // W-tile would activate with vec4 -> cap at vec4 to get load sharing
      max_vs = 4;
    }
  }

  // Don't allow vec8 if it would reduce work-items below the minimum
  // occupancy threshold.
  if (max_vs > 4 && channels % 8 == 0) {
    index_t vec8_items = batch_count * spatial_out * (channels / 8);
    if (vec8_items < min_work_items) {
      max_vs = 4;
    }
  }

  int vec_size = 1;
  for (int vs = max_vs; vs > 1; vs /= 2) {
    if (channels % vs != 0)
      continue;
    index_t vec_work_items = batch_count * spatial_out * (channels / vs);
    if (vec_work_items >= min_work_items) {
      vec_size = vs;
      break;
    }
  }

  // W-tile path for k=3, stride_w=1: each WI handles 2 adjacent pw outputs
  // with shared input column reuse (33% fewer loads).
  // Only activate when halved work-items still provide enough occupancy.
  const bool use_wtile = [&]() {
    if (kernel_h != 3 || kernel_w != 3 || stride_w != 1 || pooled_width < 2)
      return false;
    const index_t num_vc = channels / vec_size;
    const index_t pooled_width_half =
        (static_cast<index_t>(pooled_width) + 1) / 2;
    const index_t wtile_items = batch_count *
        static_cast<index_t>(pooled_height) * pooled_width_half * num_vc;
    // W-tile halves the work-item count for a smaller per-output saving than
    // a larger vec_size, so it needs more headroom above the occupancy floor
    // to be worth activating (8x reproduces the tuned 65536 on BMG).
    return wtile_items >= 8 * min_work_items;
  }();

  // Bitshift-based division is used in place of IntDivider when
  // num_vec_channels is a power of 2 (cheaper than magic-number divmod).
  auto nvc_shift_for = [](index_t num_vec_channels) {
    int shift = 0;
    if ((num_vec_channels & (num_vec_channels - 1)) == 0) {
      index_t tmp = num_vec_channels;
      while (tmp > 1) {
        shift++;
        tmp >>= 1;
      }
    }
    return shift;
  };

#define LAUNCH_AVG_POOL2D_CL_VEC(VS)                                          \
  {                                                                           \
    using vec_t = memory::aligned_vector<scalar_t, VS>;                       \
    const index_t num_vec_channels = channels / VS;                           \
    const index_t vec_total = (total_elements / channels) * num_vec_channels; \
    const int64_t gr =                                                        \
        xpuKernelLoopGroupRange(                                              \
            vec_total, group_size, vectorized_grid_stride_cap()) *            \
        group_size;                                                           \
    sycl_kernel_submit<avg_pool2d_channels_last_vec_kernel_impl<              \
        scalar_t,                                                             \
        accscalar_t,                                                          \
        vec_t,                                                                \
        VS,                                                                   \
        index_t>>(                                                            \
        gr,                                                                   \
        group_size,                                                           \
        queue,                                                                \
        0,                                                                    \
        reinterpret_cast<vec_t*>(top_data),                                   \
        reinterpret_cast<const vec_t*>(bottom_data),                          \
        vec_total,                                                            \
        num_vec_channels,                                                     \
        static_cast<int>(height),                                             \
        static_cast<int>(width),                                              \
        pooled_height,                                                        \
        pooled_width,                                                         \
        kernel_h,                                                             \
        kernel_w,                                                             \
        stride_h,                                                             \
        stride_w,                                                             \
        pad_h,                                                                \
        pad_w,                                                                \
        divisor_override,                                                     \
        count_include_pad,                                                    \
        use_divisor,                                                          \
        at::detail::IntDivider<unsigned int>(                                 \
            static_cast<unsigned int>(num_vec_channels)),                     \
        at::detail::IntDivider<unsigned int>(                                 \
            static_cast<unsigned int>(pooled_width)),                         \
        at::detail::IntDivider<unsigned int>(                                 \
            static_cast<unsigned int>(pooled_height)),                        \
        static_cast<index_t>(num_vec_channels - 1),                           \
        nvc_shift_for(num_vec_channels));                                     \
  }

#define LAUNCH_AVG_POOL2D_CL_WTILE(VS)                                 \
  {                                                                    \
    using vec_t = memory::aligned_vector<scalar_t, VS>;                \
    const index_t num_vec_channels = channels / VS;                    \
    const index_t pooled_width_half =                                  \
        (static_cast<index_t>(pooled_width) + 1) / 2;                  \
    const index_t wtile_total = batch_count *                          \
        static_cast<index_t>(pooled_height) * pooled_width_half *      \
        num_vec_channels;                                              \
    const int64_t gr =                                                 \
        xpuKernelLoopGroupRange(                                       \
            wtile_total, group_size, vectorized_grid_stride_cap()) *   \
        group_size;                                                    \
    sycl_kernel_submit<avg_pool2d_channels_last_vec_wtile_kernel_impl< \
        scalar_t,                                                      \
        accscalar_t,                                                   \
        vec_t,                                                         \
        VS,                                                            \
        index_t>>(                                                     \
        gr,                                                            \
        group_size,                                                    \
        queue,                                                         \
        0,                                                             \
        reinterpret_cast<vec_t*>(top_data),                            \
        reinterpret_cast<const vec_t*>(bottom_data),                   \
        wtile_total,                                                   \
        num_vec_channels,                                              \
        static_cast<int>(height),                                      \
        static_cast<int>(width),                                       \
        pooled_height,                                                 \
        pooled_width,                                                  \
        stride_h,                                                      \
        pad_h,                                                         \
        pad_w,                                                         \
        divisor_override,                                              \
        count_include_pad,                                             \
        use_divisor,                                                   \
        at::detail::IntDivider<unsigned int>(                          \
            static_cast<unsigned int>(num_vec_channels)),              \
        at::detail::IntDivider<unsigned int>(                          \
            static_cast<unsigned int>(pooled_width_half)),             \
        at::detail::IntDivider<unsigned int>(                          \
            static_cast<unsigned int>(pooled_height)),                 \
        static_cast<index_t>(num_vec_channels - 1),                    \
        nvc_shift_for(num_vec_channels));                              \
  }

#define LAUNCH_AVG_POOL2D_CL_HWTILE(VS)                                 \
  {                                                                     \
    using vec_t = memory::aligned_vector<scalar_t, VS>;                 \
    const index_t num_vec_channels = channels / VS;                     \
    const index_t pooled_width_half =                                   \
        (static_cast<index_t>(pooled_width) + 1) / 2;                   \
    const index_t pooled_height_half =                                  \
        (static_cast<index_t>(pooled_height) + 1) / 2;                  \
    const index_t hwtile_total = batch_count * pooled_height_half *     \
        pooled_width_half * num_vec_channels;                           \
    const int64_t gr =                                                  \
        xpuKernelLoopGroupRange(                                        \
            hwtile_total, group_size, vectorized_grid_stride_cap()) *   \
        group_size;                                                     \
    sycl_kernel_submit<avg_pool2d_channels_last_vec_hwtile_kernel_impl< \
        scalar_t,                                                       \
        accscalar_t,                                                    \
        vec_t,                                                          \
        VS,                                                             \
        index_t>>(                                                      \
        gr,                                                             \
        group_size,                                                     \
        queue,                                                          \
        0,                                                              \
        reinterpret_cast<vec_t*>(top_data),                             \
        reinterpret_cast<const vec_t*>(bottom_data),                    \
        hwtile_total,                                                   \
        num_vec_channels,                                               \
        static_cast<int>(height),                                       \
        static_cast<int>(width),                                        \
        pooled_height,                                                  \
        pooled_width,                                                   \
        stride_h,                                                       \
        pad_h,                                                          \
        pad_w,                                                          \
        divisor_override,                                               \
        count_include_pad,                                              \
        use_divisor,                                                    \
        at::detail::IntDivider<unsigned int>(                           \
            static_cast<unsigned int>(num_vec_channels)),               \
        at::detail::IntDivider<unsigned int>(                           \
            static_cast<unsigned int>(pooled_width_half)),              \
        at::detail::IntDivider<unsigned int>(                           \
            static_cast<unsigned int>(pooled_height_half)),             \
        static_cast<index_t>(num_vec_channels - 1),                     \
        nvc_shift_for(num_vec_channels));                               \
  }

  if (use_wtile) {
    if (vec_size == 2) {
      // HWTILE's fast path reuses one shared 4-row input grid, assuming
      // ph1's window starts exactly 1 row after ph0's (rows 1..3 of the
      // loaded grid). That assumption only holds when stride_h == 1;
      // otherwise fall back to plain VEC to avoid computing wrong sums.
      if (stride_h == 1) {
        LAUNCH_AVG_POOL2D_CL_HWTILE(2);
      } else {
        LAUNCH_AVG_POOL2D_CL_VEC(2);
      }
    } else {
      switch (vec_size) {
        case 8:
          LAUNCH_AVG_POOL2D_CL_WTILE(8);
          break;
        case 4:
          LAUNCH_AVG_POOL2D_CL_WTILE(4);
          break;
        case 1:
          LAUNCH_AVG_POOL2D_CL_WTILE(1);
          break;
        default:
          break;
      }
    }
  } else {
    switch (vec_size) {
      case 8:
        LAUNCH_AVG_POOL2D_CL_VEC(8);
        break;
      case 4:
        LAUNCH_AVG_POOL2D_CL_VEC(4);
        break;
      case 2:
        LAUNCH_AVG_POOL2D_CL_VEC(2);
        break;
      default: {
        // Scalar fallback: no vec_size evenly divides channels.
        const int64_t global_range =
            xpuKernelLoopGroupRange(total_elements, group_size) * group_size;
        sycl_kernel_submit<avg_pool2d_channels_last_kernel_impl<
            scalar_t,
            accscalar_t,
            index_t>>(
            global_range,
            group_size,
            queue,
            0,
            top_data,
            bottom_data,
            total_elements,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            divisor_override,
            count_include_pad,
            use_divisor,
            at::detail::IntDivider<unsigned int>(
                static_cast<unsigned int>(channels)),
            at::detail::IntDivider<unsigned int>(
                static_cast<unsigned int>(pooled_width)),
            at::detail::IntDivider<unsigned int>(
                static_cast<unsigned int>(pooled_height)));
        break;
      }
    }
  }
#undef LAUNCH_AVG_POOL2D_CL_VEC
#undef LAUNCH_AVG_POOL2D_CL_WTILE
#undef LAUNCH_AVG_POOL2D_CL_HWTILE
}

template <typename scalar_t, typename accscalar_t, typename index_t>
void launch_avg_pool2d_kernel(
    const int total_elements,
    const Tensor& input,
    const index_t channels,
    const index_t height,
    const index_t width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const Tensor& output,
    const int divisor_override,
    const bool count_include_pad,
    const bool use_divisor) {
  scalar_t* top_data = output.mutable_data_ptr<scalar_t>();
  const scalar_t* bottom_data = input.const_data_ptr<scalar_t>();

  auto& queue = at::xpu::getCurrentSYCLQueue();
  const int64_t group_size =
      static_cast<int64_t>(syclMaxWorkItemsPerSubSlice());
  const int64_t global_range =
      xpuKernelLoopGroupRange(total_elements, group_size) * group_size;

  sycl_kernel_submit<avg_pool2d_kernel_impl<scalar_t, accscalar_t, index_t>>(
      global_range,
      group_size,
      queue,
      0,
      top_data,
      bottom_data,
      total_elements,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      divisor_override,
      count_include_pad,
      use_divisor);
}

template <typename scalar_t, typename accscalar_t, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void avg_pool2d_channels_last_backward_kernel_impl(
    const scalar_t* top_data,
    scalar_t* bottom_data,
    int64_t total_elements,
    int64_t channels,
    int64_t height,
    int64_t width,
    int pooled_height,
    int pooled_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int divisor_override,
    bool count_include_pad,
    bool use_divisor) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  XPU_KERNEL_LOOP_TYPE(item, index, total_elements, index_t) {
    const int c = index % channels;
    const int w = (index / channels) % width + pad_w;
    const int h = (index / channels / width) % height + pad_h;
    const int n = index / channels / width / height;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = sycl::min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = sycl::min(w / stride_w + 1, pooled_width);
    accscalar_t gradient = accscalar_t(0);
    const scalar_t* const top_slice =
        top_data + n * channels * pooled_height * pooled_width + c;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend =
            sycl::min(hstart + kernel_h, static_cast<int>(height + pad_h));
        int wend =
            sycl::min(wstart + kernel_w, static_cast<int>(width + pad_w));
        int pool_size = (hend - hstart) * (wend - wstart);
        hstart = sycl::max(hstart, 0);
        wstart = sycl::max(wstart, 0);
        hend = sycl::min(hend, static_cast<int>(height));
        wend = sycl::min(wend, static_cast<int>(width));
        if (hstart >= hend || wstart >= wend) {
          continue;
        }
        int divide_factor;
        if (use_divisor) {
          divide_factor = divisor_override;
        } else {
          if (count_include_pad) {
            divide_factor = pool_size;
          } else {
            divide_factor = (hend - hstart) * (wend - wstart);
          }
        }
        gradient +=
            top_slice[(ph * pooled_width + pw) * channels] / divide_factor;
      }
    }
    bottom_data[index] = static_cast<scalar_t>(gradient);
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void avg_pool2d_backward_kernel_impl(
    const scalar_t* top_data,
    scalar_t* bottom_data,
    int64_t total_elements,
    int64_t channels,
    int64_t height,
    int64_t width,
    int pooled_height,
    int pooled_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int divisor_override,
    bool count_include_pad,
    bool use_divisor) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  XPU_KERNEL_LOOP_TYPE(item, index, total_elements, index_t) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = sycl::min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = sycl::min(w / stride_w + 1, pooled_width);
    accscalar_t gradient = accscalar_t(0);
    const scalar_t* const top_data_slice =
        top_data + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend =
            sycl::min(hstart + kernel_h, static_cast<int>(height + pad_h));
        int wend =
            sycl::min(wstart + kernel_w, static_cast<int>(width + pad_w));
        int pool_size = (hend - hstart) * (wend - wstart);
        hstart = sycl::max(hstart, 0);
        wstart = sycl::max(wstart, 0);
        hend = sycl::min(hend, static_cast<int>(height));
        wend = sycl::min(wend, static_cast<int>(width));
        if (hstart >= hend || wstart >= wend) {
          continue;
        }
        int divide_factor;
        if (use_divisor) {
          divide_factor = divisor_override;
        } else {
          if (count_include_pad) {
            divide_factor = pool_size;
          } else {
            divide_factor = (hend - hstart) * (wend - wstart);
          }
        }
        gradient += top_data_slice[ph * pooled_width + pw] / divide_factor;
      }
    }
    bottom_data[index] = static_cast<scalar_t>(gradient);
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
void launch_avg_pool2d_backward_channels_last_kernel(
    const index_t total_elements,
    const Tensor& grad_output,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const Tensor& grad_input,
    const int divisor_override,
    bool count_include_pad,
    bool use_divisor) {
  const scalar_t* top_data = grad_output.const_data_ptr<scalar_t>();
  scalar_t* bottom_data = grad_input.mutable_data_ptr<scalar_t>();

  auto& queue = at::xpu::getCurrentSYCLQueue();
  const int64_t group_size =
      static_cast<int64_t>(syclMaxWorkItemsPerSubSlice());
  const int64_t global_range =
      xpuKernelLoopGroupRange(total_elements, group_size) * group_size;

  sycl_kernel_submit<avg_pool2d_channels_last_backward_kernel_impl<
      scalar_t,
      accscalar_t,
      index_t>>(
      global_range,
      group_size,
      queue,
      0,
      top_data,
      bottom_data,
      total_elements,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      divisor_override,
      count_include_pad,
      use_divisor);
}

template <typename scalar_t, typename accscalar_t, typename index_t>
void launch_avg_pool2d_backward_kernel(
    const index_t total_elements,
    const Tensor& grad_output,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const Tensor& grad_input,
    const int divisor_override,
    bool count_include_pad,
    bool use_divisor) {
  const scalar_t* top_data = grad_output.const_data_ptr<scalar_t>();
  scalar_t* bottom_data = grad_input.mutable_data_ptr<scalar_t>();

  auto& queue = at::xpu::getCurrentSYCLQueue();
  const int64_t group_size =
      static_cast<int64_t>(syclMaxWorkItemsPerSubSlice());
  const int64_t global_range =
      xpuKernelLoopGroupRange(total_elements, group_size) * group_size;

  sycl_kernel_submit<
      avg_pool2d_backward_kernel_impl<scalar_t, accscalar_t, index_t>>(
      global_range,
      group_size,
      queue,
      0,
      top_data,
      bottom_data,
      total_elements,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      divisor_override,
      count_include_pad,
      use_divisor);
}

void avg_pool2d_kernel(
    const Tensor& input_,
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
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  int64_t outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW_, padW_, dW_, 1, ceil_mode);
  int64_t outputHeight =
      pooling_output_shape<int64_t>(inputHeight, kH_, padH_, dH_, 1, ceil_mode);
  const auto memory_format = input_.suggest_memory_format();

  Tensor input = input_.contiguous(memory_format);
  const auto count = safe_downcast<int32_t, int64_t>(output.numel());

  bool use_divisor = divisor_override.has_value();
  const auto divisor_override_value =
      use_divisor ? divisor_override.value() : 0;
  if (count != 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "avg_pool2d_xpu", [&] {
          using accscalar_t = acc_type_device<scalar_t, kXPU>;
          AT_DISPATCH_INDEX_TYPES(
              at::native::canUse32BitIndexMath(output, INT_MAX)
                  ? ScalarType::Int
                  : ScalarType::Long,
              "avg_pool2d_xpu",
              [&] {
                switch (memory_format) {
                  case MemoryFormat::ChannelsLast: {
                    output.unsafeGetTensorImpl()->empty_tensor_restride(
                        MemoryFormat::ChannelsLast);
                    launch_avg_pool2d_channels_last_kernel<
                        scalar_t,
                        accscalar_t,
                        index_t>(
                        count,
                        input,
                        nInputPlane,
                        inputHeight,
                        inputWidth,
                        outputHeight,
                        outputWidth,
                        kH_,
                        kW_,
                        dH_,
                        dW_,
                        padH_,
                        padW_,
                        output,
                        divisor_override_value,
                        count_include_pad,
                        use_divisor);
                    break;
                  }
                  case MemoryFormat::Contiguous: {
                    launch_avg_pool2d_kernel<scalar_t, accscalar_t, index_t>(
                        count,
                        input,
                        nInputPlane,
                        inputHeight,
                        inputWidth,
                        outputHeight,
                        outputWidth,
                        kH_,
                        kW_,
                        dH_,
                        dW_,
                        padH_,
                        padW_,
                        output,
                        divisor_override_value,
                        count_include_pad,
                        use_divisor);
                    break;
                  }
                  default:
                    TORCH_CHECK(
                        false,
                        "Unsupported memory format. Supports only "
                        "ChannelsLast, Contiguous");
                }
              });
        });
  }
}

void avg_pool2d_backward_kernel(
    const Tensor& gradOutput_,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override,
    const Tensor& gradInput) {
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);

  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  const auto memory_format = input_.suggest_memory_format();
  const Tensor input = input_.contiguous(memory_format);
  const Tensor gradOutput = gradOutput_.contiguous(memory_format);

  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  int64_t outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  int64_t outputHeight =
      pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);

  const auto count = input.numel();
  if (count == 0) {
    return;
  }
  bool use_divisor = divisor_override.has_value();
  const auto divisor_override_value =
      use_divisor ? divisor_override.value() : 0;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, input.scalar_type(), "avg_pool2d_backward_xpu", [&] {
        using accscalar_t = acc_type_device<scalar_t, kXPU>;

        AT_DISPATCH_INDEX_TYPES(
            at::native::canUse32BitIndexMath(input, INT_MAX) ? ScalarType::Int
                                                             : ScalarType::Long,
            "avg_pool2d_backward_xpu",
            [&] {
              switch (memory_format) {
                case MemoryFormat::ChannelsLast: {
                  gradInput.unsafeGetTensorImpl()->empty_tensor_restride(
                      MemoryFormat::ChannelsLast);
                  launch_avg_pool2d_backward_channels_last_kernel<
                      scalar_t,
                      accscalar_t,
                      index_t>(
                      count,
                      gradOutput,
                      nInputPlane,
                      inputHeight,
                      inputWidth,
                      outputHeight,
                      outputWidth,
                      kH,
                      kW,
                      dH,
                      dW,
                      padH,
                      padW,
                      gradInput,
                      divisor_override_value,
                      count_include_pad,
                      use_divisor);
                  break;
                }
                case MemoryFormat::Contiguous: {
                  launch_avg_pool2d_backward_kernel<
                      scalar_t,
                      accscalar_t,
                      index_t>(
                      count,
                      gradOutput,
                      nInputPlane,
                      inputHeight,
                      inputWidth,
                      outputHeight,
                      outputWidth,
                      kH,
                      kW,
                      dH,
                      dW,
                      padH,
                      padW,
                      gradInput,
                      divisor_override_value,
                      count_include_pad,
                      use_divisor);
                  break;
                }
                default:
                  TORCH_CHECK(
                      false,
                      "Unsupported memory format. Supports only "
                      "ChannelsLast, Contiguous");
              }
            });
      });
}

} // namespace xpu
} // namespace at::native
