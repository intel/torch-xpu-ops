/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <comm/Macros.h>
// clang-format off
DISABLE_RETURN_TYPE_WARNING_BEGIN
// clang-format on

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/xpu/UpSample.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <comm/SYCLContext.h>
#include <comm/SYCLHelpers.h>

#include <ATen/native/xpu/sycl/UpSampleLinear1dKernels.h>

namespace at::native::xpu {
template <typename scalar_t, typename accscalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void upsample_linear1d_kernel(
    const int n,
    const accscalar_t rwidth,
    const bool align_corners,
    const PackedTensorAccessor64<const scalar_t, 3> idata,
    PackedTensorAccessor64<scalar_t, 3> odata) {
  sycl::nd_item<1> item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  int index =
      item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int width1 = idata.size(2);
  const int width2 = odata.size(2);
  PackedTensorAccessor64<scalar_t, 3> odata_res = odata;

  if (index < n) {
    const int w2 = index % width2;
    // special case: just copy
    if (width1 == width2) {
      const int w1 = w2;
      for (int nc = 0; nc < batchsize; nc++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = idata[nc][c][w1];
          odata_res[nc][c][w2] = val;
        }
      }
      return;
    }

    const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
        rwidth, w2, align_corners, /*cubic=*/false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

    for (int nc = 0; nc < batchsize; nc++) {
      for (int c = 0; c < channels; ++c) {
        const accscalar_t val =
            w0lambda * idata[nc][c][w1] + w1lambda * idata[nc][c][w1 + w1p];
        odata[nc][c][w2] = static_cast<scalar_t>(val);
      }
    }
  }
}

void upsample_linear1d_kernel(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales,
    const Tensor& output) {
  int output_width = output_size[0];
  output.zero_();
  int input_width = input.size(2);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "upsample_linear1d_xpu",
      [&] {
        auto idata = input.packed_accessor64<const scalar_t, 3>();
        auto odata = output.packed_accessor64<scalar_t, 3>();

        using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales);
        const int num_kernels = output_width;
        constexpr auto kfn = upsample_linear1d_kernel<scalar_t, accscalar_t>;
        const auto local_range = syclMaxWorkGroupSize<kfn>();
        auto global_range =
            (num_kernels + local_range - 1) / local_range * local_range;
        sycl_kernel_submit<kfn>(
            global_range,
            local_range,
            getCurrentSYCLQueue(),
            0,
            num_kernels,
            rwidth,
            align_corners,
            idata,
            odata);
      });
}

template <typename scalar_t, typename accscalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void upsample_linear1d_backward_kernel(
    const int n,
    const accscalar_t rwidth,
    const bool align_corners,
    PackedTensorAccessor64<scalar_t, 3> idata,
    const PackedTensorAccessor64<const scalar_t, 3> odata) {
  sycl::nd_item<1> item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  int index =
      item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int width1 = idata.size(2);
  const int width2 = odata.size(2);
  PackedTensorAccessor64<scalar_t, 3> idata_res = idata;

  if (index < n) {
    const int w2 = index % width2;
    if (width1 == width2) {
      const int w1 = w2;
      for (int nc = 0; nc < batchsize; nc++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = odata[nc][c][w1];
          idata_res[nc][c][w2] = val;
        }
      }
      return;
    }
    const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
        rwidth, w2, align_corners, /*cubic=*/false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

    for (int nc = 0; nc < batchsize; nc++) {
      for (int c = 0; c < channels; ++c) {
        const scalar_t d2val = odata[nc][c][w2];
        atomicAdd(
            (sycl_global_ptr<scalar_t>)(&idata_res[nc][c][w1]),
            static_cast<scalar_t>(w0lambda * d2val));
        atomicAdd(
            (sycl_global_ptr<scalar_t>)(&idata_res[nc][c][w1 + w1p]),
            static_cast<scalar_t>(w1lambda * d2val));
      }
    }
  }
}

void upsample_linear1d_backward_kernel(
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales,
    const Tensor& grad_input) {
  globalContext().alertNotDeterministic("upsample_linear1d_backward_out_xpu");

  int output_width = output_size[0];
  int input_width = input_size[2];
  Tensor grad_output = grad_output_.contiguous();
  grad_input.zero_();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_output.scalar_type(),
      "upsample_linear1d_backward_xpu",
      [&] {
        using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
        const int num_kernels = output_width;
        auto idata = grad_input.packed_accessor64<scalar_t, 3>();
        auto odata = grad_output.packed_accessor64<const scalar_t, 3>();
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales);
        constexpr auto kfn =
            upsample_linear1d_backward_kernel<scalar_t, accscalar_t>;
        const auto local_range = syclMaxWorkGroupSize<kfn>();
        auto global_range =
            (num_kernels + local_range - 1) / local_range * local_range;
        sycl_kernel_submit<kfn>(
            global_range,
            local_range,
            getCurrentSYCLQueue(),
            0,
            num_kernels,
            rwidth,
            align_corners,
            idata,
            odata);
      });
}
} // namespace at::native::xpu

// clang-format off
DISABLE_RETURN_TYPE_WARNING_END
// clang-format on
