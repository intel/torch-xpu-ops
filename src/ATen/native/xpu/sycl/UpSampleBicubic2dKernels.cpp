/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/xpu/XPUContext.h>
#include <comm/Macros.h>
// clang-format off
DISABLE_RETURN_TYPE_WARNING_BEGIN
// clang-format on

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/ceil_div.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/xpu/UpSample.h>
#include <comm/SYCLContext.h>
#include <comm/SYCLHelpers.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/UpSampleBicubic2dKernels.h>

namespace at::native::xpu {

constexpr int kBicubicInterpolationTaps = 4;
// Seven consecutive output centers share ten source rows instead of 28.
// This tile improved the measured 240-to-224 resize; it is not a shape guard.
constexpr int kBicubicOutputRowsPerThread = 7;
constexpr int kBicubicReusedInputRows =
    kBicubicOutputRowsPerThread + kBicubicInterpolationTaps - 1;
// Cap repeated geometry work while exposing independent batches. Eight is the
// current measured-workload tuning choice, not a hardware or correctness limit.
constexpr int kBicubicMaxBatchPartitions = 8;

template <typename scalar_t, typename accscalar_t, typename index_t>
inline accscalar_t bicubic_interpolate_row(
    const scalar_t* source_row,
    const index_t* column_offsets,
    const accscalar_t* weights) {
  return static_cast<accscalar_t>(source_row[column_offsets[0]]) * weights[0] +
      static_cast<accscalar_t>(source_row[column_offsets[1]]) * weights[1] +
      static_cast<accscalar_t>(source_row[column_offsets[2]]) * weights[2] +
      static_cast<accscalar_t>(source_row[column_offsets[3]]) * weights[3];
}

template <typename scalar_t, typename accscalar_t, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void upsample_bicubic2d_kernel(
    GenericPackedTensorAccessor<scalar_t, 4, DefaultPtrTraits, index_t> output,
    GenericPackedTensorAccessor<const scalar_t, 4, DefaultPtrTraits, index_t>
        input,
    bool channels_last,
    bool align_corners,
    accscalar_t height_scale,
    accscalar_t width_scale) {
  const int64_t work_item = sycl::ext::oneapi::this_work_item::get_nd_item<1>()
                                .get_global_linear_id();
  const index_t row_width = output.size(3) * output.size(1);
  const index_t tile_count = at::ceil_div(
      int64_t(output.size(2)), int64_t(kBicubicOutputRowsPerThread));
  const index_t partition_size = tile_count * row_width;
  const index_t batch_step =
      sycl::min(input.size(0), index_t(kBicubicMaxBatchPartitions));
  if (work_item >= int64_t(partition_size) * batch_step) {
    return;
  }
  const index_t batch_start = work_item / partition_size;
  const index_t tile_lane = work_item % partition_size;
  // Match adjacent lanes to the faster output dimension, without requiring
  // either input or output to be dense.
  const index_t output_y =
      (channels_last ? tile_lane / row_width
                     : (tile_lane / output.size(3)) % tile_count) *
      kBicubicOutputRowsPerThread;
  const index_t output_x = channels_last
      ? (tile_lane % row_width) / output.size(1)
      : tile_lane % output.size(3);
  const index_t channel = channels_last
      ? tile_lane % output.size(1)
      : tile_lane / (tile_count * output.size(3));
  const int valid_rows = sycl::min(
      index_t(kBicubicOutputRowsPerThread), output.size(2) - output_y);
  const accscalar_t real_x = area_pixel_compute_source_index(
      width_scale, output_x, align_corners, true);
  const int input_x = floorf(real_x);
  accscalar_t width_weights[kBicubicInterpolationTaps];
  get_cubic_upsampling_coefficients(width_weights, real_x - input_x);
  index_t column_offsets[kBicubicInterpolationTaps];
#pragma unroll
  for (int column = 0; column < kBicubicInterpolationTaps; ++column) {
    column_offsets[column] =
        sycl::clamp(
            input_x - 1 + column, 0, static_cast<int>(input.size(3)) - 1) *
        input.stride(3);
  }
  const int first_center = floorf(area_pixel_compute_source_index(
      height_scale, output_y, align_corners, true));
  bool consecutive = true;
#pragma unroll
  for (int row = 1; row < kBicubicOutputRowsPerThread; ++row) {
    if (row < valid_rows) {
      const int center = floorf(area_pixel_compute_source_index(
          height_scale, output_y + row, align_corners, true));
      consecutive = consecutive && center == first_center + row;
    }
  }
  if (!consecutive) {
#pragma unroll 1
    for (int row = 0; row < kBicubicOutputRowsPerThread; ++row) {
      if (row < valid_rows) {
        const accscalar_t real_y = area_pixel_compute_source_index(
            height_scale, output_y + row, align_corners, true);
        const int center = floorf(real_y);
        accscalar_t weights[kBicubicInterpolationTaps];
        get_cubic_upsampling_coefficients(weights, real_y - center);
        for (index_t batch = batch_start; batch < input.size(0);
             batch += batch_step) {
          const scalar_t* source = input.data() + batch * input.stride(0) +
              channel * input.stride(1);
          accscalar_t horizontal[kBicubicInterpolationTaps];
#pragma unroll
          for (int tap = 0; tap < kBicubicInterpolationTaps; ++tap) {
            const scalar_t* source_row = source +
                sycl::clamp(
                    center - 1 + tap, 0, static_cast<int>(input.size(2)) - 1) *
                    input.stride(2);
            horizontal[tap] = bicubic_interpolate_row<scalar_t, accscalar_t>(
                source_row, column_offsets, width_weights);
          }
          output[batch][channel][output_y + row][output_x] =
              static_cast<scalar_t>(
                  horizontal[0] * weights[0] + horizontal[1] * weights[1] +
                  horizontal[2] * weights[2] + horizontal[3] * weights[3]);
        }
      }
    }
    return;
  }
  accscalar_t height_weights[kBicubicOutputRowsPerThread]
                            [kBicubicInterpolationTaps];
#pragma unroll
  for (int row = 0; row < kBicubicOutputRowsPerThread; ++row) {
    const int valid_row = sycl::min(row, valid_rows - 1);
    const accscalar_t real_y = area_pixel_compute_source_index(
        height_scale, output_y + valid_row, align_corners, true);
    get_cubic_upsampling_coefficients(
        height_weights[row], real_y - (first_center + valid_row));
  }
  for (index_t batch = batch_start; batch < input.size(0);
       batch += batch_step) {
    const scalar_t* source =
        input.data() + batch * input.stride(0) + channel * input.stride(1);
    scalar_t* destination = output.data() + batch * output.stride(0) +
        output_y * output.stride(2) + output_x * output.stride(3) +
        channel * output.stride(1);
    accscalar_t horizontal[kBicubicReusedInputRows];
#pragma unroll
    for (int row = 0; row < kBicubicReusedInputRows; ++row) {
      const scalar_t* source_row = source +
          sycl::clamp(
              first_center - 1 + row, 0, static_cast<int>(input.size(2)) - 1) *
              input.stride(2);
      horizontal[row] = bicubic_interpolate_row<scalar_t, accscalar_t>(
          source_row, column_offsets, width_weights);
    }
#pragma unroll
    for (int row = 0; row < kBicubicOutputRowsPerThread; ++row) {
      if (row < valid_rows) {
        destination[row * output.stride(2)] = static_cast<scalar_t>(
            horizontal[row] * height_weights[row][0] +
            horizontal[row + 1] * height_weights[row][1] +
            horizontal[row + 2] * height_weights[row][2] +
            horizontal[row + 3] * height_weights[row][3]);
      }
    }
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
static void launch_upsample_bicubic2d_out_kernel(
    GenericPackedTensorAccessor<scalar_t, 4, DefaultPtrTraits, index_t> output,
    GenericPackedTensorAccessor<const scalar_t, 4, DefaultPtrTraits, index_t>
        input,
    bool align_corners,
    const accscalar_t height_scale,
    const accscalar_t width_scale) {
  const int64_t work_items =
      at::ceil_div(
          int64_t(output.size(2)), int64_t(kBicubicOutputRowsPerThread)) *
      output.size(3) * output.size(1) *
      std::min<int64_t>(input.size(0), kBicubicMaxBatchPartitions);
  const int64_t group_size = at::xpu::getKernelMaxWorkGroupSize<
      upsample_bicubic2d_kernel<scalar_t, accscalar_t, index_t>>();
  const bool channels_last = output.stride(1) < output.stride(3);
  sycl_kernel_submit<upsample_bicubic2d_kernel<scalar_t, accscalar_t, index_t>>(
      at::ceil_div(work_items, group_size) * group_size,
      group_size,
      getCurrentSYCLQueue(),
      0,
      output,
      input,
      channels_last,
      align_corners,
      height_scale,
      width_scale);
}

void upsample_bicubic2d_kernel(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});

  int output_height = output_size[0];
  int output_width = output_size[1];

  int input_height = input.size(2);
  int input_width = input.size(3);

  if (output.numel() == 0) {
    return;
  }
  if (input_height == output_height && input_width == output_width) {
    output.copy_(input);
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "upsample_bicubic2d_xpu",
      [&] {
        // Get scaling factors
        using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);

        AT_DISPATCH_INDEX_TYPES(
            canUse32BitIndexMath(input) && canUse32BitIndexMath(output)
                ? ScalarType::Int
                : ScalarType::Long,
            "upsample_bicubic2d_index",
            [&] {
              launch_upsample_bicubic2d_out_kernel<
                  scalar_t,
                  accscalar_t,
                  index_t>(
                  output.generic_packed_accessor<
                      scalar_t,
                      4,
                      DefaultPtrTraits,
                      index_t>(),
                  input.generic_packed_accessor<
                      const scalar_t,
                      4,
                      DefaultPtrTraits,
                      index_t>(),
                  align_corners,
                  rheight,
                  rwidth);
            });
      });
}

template <typename scalar_t, typename accscalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void upsample_bicubic2d_backward_kernel(
    int num_elements,
    accscalar_t height_scale,
    accscalar_t width_scale,
    bool align_corners,
    PackedTensorAccessor64<scalar_t, 4> in_data,
    const PackedTensorAccessor64<const scalar_t, 4> out_data) {
  sycl::nd_item<1> item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  int index = item.get_global_linear_id();
  const int batchsize = in_data.size(0);
  const int channels = in_data.size(1);
  const int input_height = in_data.size(2);
  const int input_width = in_data.size(3);
  const int output_height = out_data.size(2);
  const int output_width = out_data.size(3);
  if (index >= num_elements) {
    return;
  }
  const int output_x = index % output_width;
  const int output_y = index / output_width;
  // special case: output_xust copy
  if (input_height == output_height && input_width == output_width) {
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const scalar_t val = out_data[n][c][output_y][output_x];
        in_data[n][c][output_y][output_x] = val;
      }
    }
    return;
  }

  accscalar_t real_x = area_pixel_compute_source_index(
      width_scale, output_x, align_corners, /*cubic=*/true);
  int input_x = floorf(real_x);
  accscalar_t t_x = real_x - input_x;

  accscalar_t real_y = area_pixel_compute_source_index(
      height_scale, output_y, align_corners, /*cubic=*/true);
  int input_y = floorf(real_y);
  accscalar_t t_y = real_y - input_y;

  accscalar_t x_coeffs[4];
  accscalar_t y_coeffs[4];

  get_cubic_upsampling_coefficients(x_coeffs, t_x);
  get_cubic_upsampling_coefficients(y_coeffs, t_y);

  for (int n = 0; n < batchsize; n++) {
    for (int c = 0; c < channels; ++c) {
      scalar_t out_value = out_data[n][c][output_y][output_x];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          upsample_increment_value_bounded<scalar_t, accscalar_t>(
              in_data,
              n,
              c,
              input_height,
              input_width,
              input_y - 1 + i,
              input_x - 1 + j,
              out_value * y_coeffs[i] * x_coeffs[j]);
        }
      }
    }
  }
}

template <typename scalar_t, typename accscalar_t>
static void launch_upsample_bicubic2d_backward_out_kernel(
    int64_t num_elements,
    const accscalar_t height_scale,
    const accscalar_t width_scale,
    const bool align_corners,
    PackedTensorAccessor64<scalar_t, 4> idata,
    const PackedTensorAccessor64<const scalar_t, 4> odata) {
  int64_t wg_size = at::xpu::getKernelMaxWorkGroupSize<
      upsample_bicubic2d_backward_kernel<scalar_t, accscalar_t>>();
  int64_t num_wg = at::ceil_div(num_elements, wg_size);
  auto queue = getCurrentSYCLQueue();

  sycl_kernel_submit<upsample_bicubic2d_backward_kernel<scalar_t, accscalar_t>>(
      num_wg * wg_size,
      wg_size,
      queue,
      0,
      (int)num_elements,
      height_scale,
      width_scale,
      align_corners,
      idata,
      odata);
}

void upsample_bicubic2d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(__func__, {grad_output_arg, grad_input_arg});

  int output_height = output_size[0];
  int output_width = output_size[1];

  int input_height = input_size[2];
  int input_width = input_size[3];

  Tensor grad_output = grad_output_.contiguous();

  grad_input.zero_();
  const int num_elements = output_height * output_width;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_output.scalar_type(),
      "upsample_bicubic2d_backward_out_frame",
      [&] {
        using accscalar_t = at::acc_type_device<scalar_t, kXPU>;

        auto idata = grad_input.packed_accessor64<scalar_t, 4>();
        auto odata = grad_output.packed_accessor64<const scalar_t, 4>();

        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);

        launch_upsample_bicubic2d_backward_out_kernel<scalar_t, accscalar_t>(
            num_elements, rheight, rwidth, align_corners, idata, odata);
      });
}

} // namespace at::native::xpu

// clang-format off
DISABLE_RETURN_TYPE_WARNING_END
// clang-format on
