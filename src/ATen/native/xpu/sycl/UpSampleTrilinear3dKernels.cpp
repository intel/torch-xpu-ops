/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/ceil_div.h>

#include <ATen/native/xpu/UpSample.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/UpSampleTrilinear3dKernels.h>

namespace at::native::xpu {

template <typename scalar_t, typename accscalar_t>
struct UpsampleTrilinear3dKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto odata = odata_;
    int index = item.get_global_linear_id();

    const int batchsize = idata_.size(0);
    const int channels = idata_.size(1);
    const int depth1 = idata_.size(2);
    const int height1 = idata_.size(3);
    const int width1 = idata_.size(4);
    const int depth2 = odata_.size(2);
    const int height2 = odata_.size(3);
    const int width2 = odata_.size(4);

    if (index < n_) {
      const int w2 = (index % (height2 * width2)) % width2; // 0:width2-1
      const int h2 = (index % (height2 * width2)) / width2; // 0:height2-1
      const int t2 = index / (height2 * width2); // 0:depth2-1

      if (depth1 == depth2 && height1 == height2 && width1 == width2) {
        const int t1 = t2;
        const int h1 = h2;
        const int w1 = w2;

        for (int n = 0; n < batchsize; n++) {
          for (int c = 0; c < channels; ++c) {
            const scalar_t val = idata_[n][c][t1][h1][w1];
            odata[n][c][t2][h2][w2] = val;
          }
        }
        return;
      }
      //
      const accscalar_t t1r = area_pixel_compute_source_index<accscalar_t>(
          rdepth_, t2, align_corners_, /*cubic=*/false);
      const int t1 = t1r;
      const int t1p = (t1 < depth1 - 1) ? 1 : 0;
      const accscalar_t t1lambda = t1r - t1;
      const accscalar_t t0lambda = static_cast<accscalar_t>(1) - t1lambda;
      //
      const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
          rheight_, h2, align_corners_, /*cubic=*/false);
      const int h1 = h1r;
      const int h1p = (h1 < height1 - 1) ? 1 : 0;
      const accscalar_t h1lambda = h1r - h1;
      const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
      //
      const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
          rwidth_, w2, align_corners_, /*cubic=*/false);
      const int w1 = w1r;
      const int w1p = (w1 < width1 - 1) ? 1 : 0;
      const accscalar_t w1lambda = w1r - w1;
      const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
      //
      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const accscalar_t val = t0lambda *
                  (h0lambda *
                       (w0lambda * idata_[n][c][t1][h1][w1] +
                        w1lambda * idata_[n][c][t1][h1][w1 + w1p]) +
                   h1lambda *
                       (w0lambda * idata_[n][c][t1][h1 + h1p][w1] +
                        w1lambda * idata_[n][c][t1][h1 + h1p][w1 + w1p])) +
              t1lambda *
                  (h0lambda *
                       (w0lambda * idata_[n][c][t1 + t1p][h1][w1] +
                        w1lambda * idata_[n][c][t1 + t1p][h1][w1 + w1p]) +
                   h1lambda *
                       (w0lambda * idata_[n][c][t1 + t1p][h1 + h1p][w1] +
                        w1lambda * idata_[n][c][t1 + t1p][h1 + h1p][w1 + w1p]));
          odata[n][c][t2][h2][w2] = static_cast<scalar_t>(val);
        }
      }
    }
  }
  UpsampleTrilinear3dKernelFunctor(
      const int n,
      const accscalar_t rdepth,
      const accscalar_t rheight,
      const accscalar_t rwidth,
      const bool align_corners,
      const PackedTensorAccessor64<const scalar_t, 5> idata,
      PackedTensorAccessor64<scalar_t, 5> odata)
      : n_(n),
        rdepth_(rdepth),
        rheight_(rheight),
        rwidth_(rwidth),
        align_corners_(align_corners),
        idata_(idata),
        odata_(odata) {}

 private:
  const int n_;
  const accscalar_t rdepth_;
  const accscalar_t rheight_;
  const accscalar_t rwidth_;
  const bool align_corners_;
  const PackedTensorAccessor64<const scalar_t, 5> idata_;
  PackedTensorAccessor64<scalar_t, 5> odata_;
};

template <typename scalar_t, typename accscalar_t>
void launch_upsample_trilinear3d_kernel(
    const int n,
    const accscalar_t rdepth,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    const PackedTensorAccessor64<const scalar_t, 5> idata_acc,
    PackedTensorAccessor64<scalar_t, 5> odata_acc) {
  UpsampleTrilinear3dKernelFunctor<scalar_t, accscalar_t> kfn(
      n, rdepth, rheight, rwidth, align_corners, idata_acc, odata_acc);

  int64_t wg_size = syclMaxWorkGroupSize(kfn);
  int num_group = at::ceil_div(n, (int)wg_size);
  auto queue = getCurrentSYCLQueue();

  sycl_kernel_submit(
      sycl::range<1>(num_group * wg_size), sycl::range<1>(wg_size), queue, kfn);
}

inline size_t idx_3d(
    const size_t nc,
    const size_t depth,
    const size_t height,
    const size_t width,
    const size_t z,
    const size_t y,
    const size_t x) {
  return ((nc * depth + z) * height + y) * width + x;
}

template <typename scalar_t, typename accscalar_t>
struct UpsampleTrilinear3dBackwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int index = item.get_global_linear_id();

    const int batchsize = idata_.size(0);
    const int channels = idata_.size(1);
    const int depth1 = idata_.size(2);
    const int height1 = idata_.size(3);
    const int width1 = idata_.size(4);
    const int depth2 = odata_.size(2);
    const int height2 = odata_.size(3);
    const int width2 = odata_.size(4);

    if (index < n_) {
      const int w2 = (index % (height2 * width2)) % width2; // 0:width2-1
      const int h2 = (index % (height2 * width2)) / width2; // 0:height2-1
      const int t2 = index / (height2 * width2); // 0:depth2-1
      // special case: just copy
      auto idata = idata_;
      if (depth1 == depth2 && height1 == height2 && width1 == width2) {
        const int t1 = t2;
        const int h1 = h2;
        const int w1 = w2;

        for (int n = 0; n < batchsize; n++) {
          for (int c = 0; c < channels; ++c) {
            const scalar_t val = odata_[n][c][t1][h1][w1];
            idata[n][c][t2][h2][w2] = val;
          }
        }
        return;
      }

      //
      const accscalar_t t1r = area_pixel_compute_source_index<accscalar_t>(
          rdepth_, t2, align_corners_, /*cubic=*/false);
      const int t1 = t1r;
      const int t1p = (t1 < depth1 - 1) ? 1 : 0;
      const accscalar_t t1lambda = t1r - t1;
      const accscalar_t t0lambda = static_cast<accscalar_t>(1) - t1lambda;
      //
      const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
          rheight_, h2, align_corners_, /*cubic=*/false);
      const int h1 = h1r;
      const int h1p = (h1 < height1 - 1) ? 1 : 0;
      const accscalar_t h1lambda = h1r - h1;
      const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
      //
      const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
          rwidth_, w2, align_corners_, /*cubic=*/false);
      const int w1 = w1r;
      const int w1p = (w1 < width1 - 1) ? 1 : 0;
      const accscalar_t w1lambda = w1r - w1;
      const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
      //
      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t d2val = odata_[n][c][t2][h2][w2];
          const size_t nc = n * channels + c;

          atomicAdd(
              (sycl_global_ptr<
                  scalar_t>)(idata_ptr_ + idx_3d(nc, depth1, height1, width1, t1, h1, w1)),
              static_cast<scalar_t>(t0lambda * h0lambda * w0lambda * d2val));

          atomicAdd(
              (sycl_global_ptr<
                  scalar_t>)(idata_ptr_ + idx_3d(nc, depth1, height1, width1, t1, h1, w1 + w1p)),

              static_cast<scalar_t>(t0lambda * h0lambda * w1lambda * d2val));
          atomicAdd(
              (sycl_global_ptr<
                  scalar_t>)(idata_ptr_ + idx_3d(nc, depth1, height1, width1, t1, h1 + h1p, w1)),

              static_cast<scalar_t>(t0lambda * h1lambda * w0lambda * d2val));
          atomicAdd(
              (sycl_global_ptr<scalar_t>)(sycl_global_ptr<
                                          scalar_t>)(idata_ptr_ + idx_3d(nc, depth1, height1, width1, t1, h1 + h1p, w1 + w1p)),

              static_cast<scalar_t>(t0lambda * h1lambda * w1lambda * d2val));
          atomicAdd(
              (sycl_global_ptr<
                  scalar_t>)(idata_ptr_ + idx_3d(nc, depth1, height1, width1, t1 + t1p, h1, w1)),

              static_cast<scalar_t>(t1lambda * h0lambda * w0lambda * d2val));
          atomicAdd(
              (sycl_global_ptr<
                  scalar_t>)(idata_ptr_ + idx_3d(nc, depth1, height1, width1, t1 + t1p, h1, w1 + w1p)),

              static_cast<scalar_t>(t1lambda * h0lambda * w1lambda * d2val));
          atomicAdd(
              (sycl_global_ptr<
                  scalar_t>)(idata_ptr_ + idx_3d(nc, depth1, height1, width1, t1 + t1p, h1 + h1p, w1)),

              static_cast<scalar_t>(t1lambda * h1lambda * w0lambda * d2val));
          atomicAdd(
              (sycl_global_ptr<
                  scalar_t>)(idata_ptr_ + idx_3d(nc, depth1, height1, width1, t1 + t1p, h1 + h1p, w1 + w1p)),

              static_cast<scalar_t>(t1lambda * h1lambda * w1lambda * d2val));
        }
      }
    }
  }

  UpsampleTrilinear3dBackwardKernelFunctor(
      const size_t n,
      const accscalar_t rdepth,
      const accscalar_t rheight,
      const accscalar_t rwidth,
      const bool align_corners,
      PackedTensorAccessor64<scalar_t, 5> idata,
      const PackedTensorAccessor64<const scalar_t, 5> odata,
      scalar_t* idata_ptr)
      : n_(n),
        rdepth_(rdepth),
        rheight_(rheight),
        rwidth_(rwidth),
        align_corners_(align_corners),
        idata_(idata),
        odata_(odata),
        idata_ptr_(idata_ptr) {}

 private:
  const size_t n_;
  const accscalar_t rdepth_;
  const accscalar_t rheight_;
  const accscalar_t rwidth_;
  const bool align_corners_;
  PackedTensorAccessor64<scalar_t, 5> idata_;
  const PackedTensorAccessor64<const scalar_t, 5> odata_;
  scalar_t* idata_ptr_;
};

template <typename scalar_t, typename accscalar_t>
void launch_upsample_trilinear3d_backward_kernel(
    const size_t num_kernels,
    const accscalar_t rdepth,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    PackedTensorAccessor64<scalar_t, 5> idata,
    const PackedTensorAccessor64<const scalar_t, 5> odata,
    scalar_t* idata_ptr) {
  UpsampleTrilinear3dBackwardKernelFunctor<scalar_t, accscalar_t> kfn(
      num_kernels,
      rdepth,
      rheight,
      rwidth,
      align_corners,
      idata,
      odata,
      idata_ptr);

  int64_t wg_size = syclMaxWorkGroupSize(kfn);
  int num_group = at::ceil_div((int64_t)num_kernels, (int64_t)wg_size);
  auto queue = getCurrentSYCLQueue();

  sycl_kernel_submit(
      sycl::range<1>(num_group * wg_size), sycl::range<1>(wg_size), queue, kfn);
}

void upsample_trilinear3d_out_kernel(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});

  int output_depth = output_size[0];
  int output_height = output_size[1];
  int output_width = output_size[2];

  int input_depth = input.size(2);
  int input_height = input.size(3);
  int input_width = input.size(4);

  const int num_kernels = output_depth * output_height * output_width;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "upsample_trilinear3d_xpu",
      [&] {
        using accscalar_t = acc_type_device<scalar_t, kXPU>;
        auto idata_acc = input.packed_accessor64<const scalar_t, 5>();
        auto odata_acc = output.packed_accessor64<scalar_t, 5>();

        const accscalar_t rdepth = area_pixel_compute_scale<accscalar_t>(
            input_depth, output_depth, align_corners, scales_d);
        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);

        launch_upsample_trilinear3d_kernel<scalar_t, accscalar_t>(
            num_kernels,
            rdepth,
            rheight,
            rwidth,
            align_corners,
            idata_acc,
            odata_acc);
      });
}

void upsample_trilinear3d_backward_out_kernel(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  TensorArg grad_input_arg{grad_input_, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(__func__, {grad_output_arg, grad_input_arg});

  int output_depth = output_size[0];
  int output_height = output_size[1];
  int output_width = output_size[2];

  int input_depth = input_size[2];
  int input_height = input_size[3];
  int input_width = input_size[4];

  Tensor grad_output = grad_output_.contiguous();

  Tensor grad_input = grad_input_.contiguous();

  grad_input.zero_();

  const int num_kernels = output_depth * output_height * output_width;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_output_.scalar_type(),
      "upsample_trilinear3d_backward_xpu",
      [&] {
        using accscalar_t = acc_type_device<scalar_t, kXPU>;

        auto idata = grad_input.packed_accessor64<scalar_t, 5>();
        auto odata = grad_output.packed_accessor64<const scalar_t, 5>();
        scalar_t* idata_ptr = grad_input.mutable_data_ptr<scalar_t>();

        const accscalar_t rdepth = area_pixel_compute_scale<accscalar_t>(
            input_depth, output_depth, align_corners, scales_d);
        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);

        launch_upsample_trilinear3d_backward_kernel<scalar_t, accscalar_t>(
            num_kernels,
            rdepth,
            rheight,
            rwidth,
            align_corners,
            idata,
            odata,
            idata_ptr);

        if (!grad_input_.is_contiguous()) {
          grad_input_.copy_(grad_input);
        }
      });
}

} // namespace at::native::xpu
