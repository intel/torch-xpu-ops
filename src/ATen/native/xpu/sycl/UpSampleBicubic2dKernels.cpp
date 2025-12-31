/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning(disable : 4715)
#endif

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/ceil_div.h>
#include <ATen/native/xpu/UpSample.h>
#include <comm/SYCLContext.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/UpSampleBicubic2dKernels.h>

namespace at::native::xpu {

template <typename scalar_t, typename accscalar_t>
struct UpsampleBicubic2dKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto idata = in_data_;
    auto odata = out_data_;
    int global_id = item.get_global_linear_id();
    const int nbatch = idata.size(0);
    const int channels = idata.size(1);
    const int input_height = idata.size(2);
    const int input_width = idata.size(3);
    const int output_height = odata.size(2);
    const int output_width = odata.size(3);
    if (global_id < output_height * output_width) {
      const int output_x = global_id % output_width;
      const int output_y = global_id / output_width;
      if (input_height == output_height && input_width == output_width) {
        for (int n = 0; n < nbatch; n++) {
          for (int c = 0; c < channels; c++) {
            auto val = idata[n][c][output_y][output_x];
            odata[n][c][output_y][output_x] = val;
          }
        }
        return;
      }

      // Interpolation kernel
      accscalar_t real_x = area_pixel_compute_source_index(
          width_scale_, output_x, align_corners_, /*cubic=*/true);
      int in_x = floorf(real_x);
      accscalar_t t_x = real_x - in_x;

      accscalar_t real_y = area_pixel_compute_source_index(
          height_scale_, output_y, align_corners_, /*cubic=*/true);
      int in_y = floorf(real_y);
      accscalar_t t_y = real_y - in_y;
      for (int n = 0; n < nbatch; n++) {
        for (int c = 0; c < channels; c++) {
          accscalar_t coefficients[4];
          for (int k = 0; k < 4; k++) {
            coefficients[k] = cubic_interp1d<scalar_t, accscalar_t>(
                upsample_get_value_bounded<scalar_t>(
                    idata,
                    n,
                    c,
                    input_width,
                    input_height,
                    in_x - 1,
                    in_y - 1 + k),
                upsample_get_value_bounded<scalar_t>(
                    idata,
                    n,
                    c,
                    input_width,
                    input_height,
                    in_x + 0,
                    in_y - 1 + k),
                upsample_get_value_bounded<scalar_t>(
                    idata,
                    n,
                    c,
                    input_width,
                    input_height,
                    in_x + 1,
                    in_y - 1 + k),
                upsample_get_value_bounded<scalar_t>(
                    idata,
                    n,
                    c,
                    input_width,
                    input_height,
                    in_x + 2,
                    in_y - 1 + k),
                t_x);
          }

          odata[n][c][output_y][output_x] =
              static_cast<scalar_t>(cubic_interp1d<scalar_t, accscalar_t>(
                  coefficients[0],
                  coefficients[1],
                  coefficients[2],
                  coefficients[3],
                  t_y));
        }
      }
    }
  }
  UpsampleBicubic2dKernelFunctor(
      PackedTensorAccessor64<scalar_t, 4> out_data,
      const PackedTensorAccessor64<const scalar_t, 4> in_data,
      int64_t onum,
      bool align_corners,
      const accscalar_t height_scale,
      const accscalar_t width_scale)
      : out_data_(out_data),
        in_data_(in_data),
        onum_(onum),
        align_corners_(align_corners),
        height_scale_(height_scale),
        width_scale_(width_scale) {}

 private:
  PackedTensorAccessor64<scalar_t, 4> out_data_;
  const PackedTensorAccessor64<const scalar_t, 4> in_data_;
  int64_t onum_;
  bool align_corners_;
  const accscalar_t height_scale_;
  const accscalar_t width_scale_;
};

template <typename scalar_t, typename accscalar_t>
static void upsample_bicubic2d_out_template(
    PackedTensorAccessor64<scalar_t, 4> odata,
    const PackedTensorAccessor64<const scalar_t, 4> idata,
    int64_t onum,
    bool align_corners,
    const accscalar_t height_scale,
    const accscalar_t width_scale) {
  UpsampleBicubic2dKernelFunctor<scalar_t, accscalar_t> kfn(
      odata, idata, onum, align_corners, height_scale, width_scale);

  int64_t wg_size = syclMaxWorkGroupSize(kfn);
  int64_t num_wg = at::ceil_div(onum, wg_size);
  auto queue = getCurrentSYCLQueue();

  sycl_kernel_submit(num_wg * wg_size, wg_size, queue, kfn);
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

  output.zero_();

  const int num_output_elements = output_height * output_width;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "upsample_bicubic2d_xpu",
      [&] {
        auto idata = input.packed_accessor64<const scalar_t, 4>();
        auto odata = output.packed_accessor64<scalar_t, 4>();

        // Get scaling factors
        using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);

        upsample_bicubic2d_out_template<scalar_t, accscalar_t>(
            odata, idata, num_output_elements, align_corners, rheight, rwidth);
      });
}

template <typename scalar_t, typename accscalar_t>
struct UpsampleBicubic2dBackwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto idata = in_data_;
    auto odata = out_data_;
    int index = item.get_global_linear_id();
    const int batchsize = idata.size(0);
    const int channels = idata.size(1);
    const int input_height = idata.size(2);
    const int input_width = idata.size(3);
    const int output_height = odata.size(2);
    const int output_width = odata.size(3);
    if (index >= num_elements_) {
      return;
    }
    const int output_x = index % output_width;
    const int output_y = index / output_width;
    // special case: output_xust copy
    if (input_height == output_height && input_width == output_width) {
      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = odata[n][c][output_y][output_x];
          idata[n][c][output_y][output_x] = val;
        }
      }
      return;
    }

    accscalar_t real_x = area_pixel_compute_source_index(
        width_scale_, output_x, align_corners_, /*cubic=*/true);
    int input_x = floorf(real_x);
    accscalar_t t_x = real_x - input_x;

    accscalar_t real_y = area_pixel_compute_source_index(
        height_scale_, output_y, align_corners_, /*cubic=*/true);
    int input_y = floorf(real_y);
    accscalar_t t_y = real_y - input_y;

    accscalar_t x_coeffs[4];
    accscalar_t y_coeffs[4];

    get_cubic_upsampling_coefficients(x_coeffs, t_x);
    get_cubic_upsampling_coefficients(y_coeffs, t_y);

    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        scalar_t out_value = odata[n][c][output_y][output_x];
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 4; j++) {
            upsample_increment_value_bounded<scalar_t, accscalar_t>(
                idata,
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
  UpsampleBicubic2dBackwardKernelFunctor(
      const int num_elements,
      const accscalar_t height_scale,
      const accscalar_t width_scale,
      const bool align_corners,
      PackedTensorAccessor64<scalar_t, 4> in_data,
      const PackedTensorAccessor64<const scalar_t, 4> out_data)
      : num_elements_(num_elements),
        height_scale_(height_scale),
        width_scale_(width_scale),
        align_corners_(align_corners),
        in_data_(in_data),
        out_data_(out_data) {}

 private:
  const int num_elements_;
  const accscalar_t height_scale_;
  const accscalar_t width_scale_;
  const bool align_corners_;
  PackedTensorAccessor64<scalar_t, 4> in_data_;
  const PackedTensorAccessor64<const scalar_t, 4> out_data_;
};

template <typename scalar_t, typename accscalar_t>
static void upsample_bicubic2d_backward_out_template(
    int64_t num_elements,
    const accscalar_t height_scale,
    const accscalar_t width_scale,
    const bool align_corners,
    PackedTensorAccessor64<scalar_t, 4> idata,
    const PackedTensorAccessor64<const scalar_t, 4> odata) {
  UpsampleBicubic2dBackwardKernelFunctor<scalar_t, accscalar_t> kfn(
      num_elements, height_scale, width_scale, align_corners, idata, odata);

  int64_t wg_size = syclMaxWorkGroupSize(kfn);
  int64_t num_wg = at::ceil_div(num_elements, wg_size);
  auto queue = getCurrentSYCLQueue();

  sycl_kernel_submit(num_wg * wg_size, wg_size, queue, kfn);
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

        upsample_bicubic2d_backward_out_template<scalar_t, accscalar_t>(
            num_elements, rheight, rwidth, align_corners, idata, odata);
      });
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
#ifdef _MSC_VER
  #pragma warning(pop)
#endif

