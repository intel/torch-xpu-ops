#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/ceil_div.h>
#include <comm/SYCLContext.h>
#include "Atomics.h"
#include "UpSample.h"

namespace at::native::xpu {

template <typename scalar_t, typename accscalar_t>
struct UpsampleBilinear2dOutFrameKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int index = item.get_global_linear_id();

    if (index < n) {
      const int output_x = index % output_width;
      const int output_y = index / output_width;

      const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
          rheight, output_y, align_corners, /*cubic=*/false);
      const int h1 = h1r;
      const int h1p = (h1 < input_height - 1) ? 1 : 0;
      const accscalar_t h1lambda = h1r - h1;
      const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;

      const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
          rwidth, output_x, align_corners, /*cubic=*/false);
      const int w1 = w1r;
      const int w1p = (w1 < input_width - 1) ? 1 : 0;
      const accscalar_t w1lambda = w1r - w1;
      const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
      auto odata = out_data_acc;
      for (int n = 0; n < nbatch; n++) {
        for (int c = 0; c < channels; ++c) {
          const accscalar_t val = h0lambda *
                  (w0lambda * in_data_acc[n][c][h1][w1] +
                   w1lambda * in_data_acc[n][c][h1][w1 + w1p]) +
              h1lambda *
                  (w0lambda * in_data_acc[n][c][h1 + h1p][w1] +
                   w1lambda * in_data_acc[n][c][h1 + h1p][w1 + w1p]);
          odata[n][c][output_y][output_x] = static_cast<scalar_t>(val);
        }
      }
    }
  }
  UpsampleBilinear2dOutFrameKernelFunctor(
      const int n_,
      const accscalar_t rheight_,
      const accscalar_t rwidth_,
      const bool align_corners_,
      const PackedTensorAccessor<scalar_t, 4> idata_acc_,
      PackedTensorAccessor<scalar_t, 4> odata_acc_,
      int64_t input_height_,
      int64_t input_width_,
      int64_t output_height_,
      int64_t output_width_,
      int64_t nbatch_,
      int64_t channels_)
      : n(n_),
        rheight(rheight_),
        rwidth(rwidth_),
        align_corners(align_corners_),
        in_data_acc(idata_acc_),
        out_data_acc(odata_acc_),
        input_height(input_height_),
        input_width(input_width_),
        output_height(output_height_),
        output_width(output_width_),
        nbatch(nbatch_),
        channels(channels_) {}

 private:
  const int n;
  const accscalar_t rheight;
  const accscalar_t rwidth;
  const bool align_corners;
  const PackedTensorAccessor<scalar_t, 4> in_data_acc;
  PackedTensorAccessor<scalar_t, 4> out_data_acc;
  int64_t input_height;
  int64_t input_width;
  int64_t output_height;
  int64_t output_width;
  int64_t nbatch;
  int64_t channels;
};

template <typename scalar_t, typename accscalar_t>
void upsample_bilinear2d_out_frame(
    const int n,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    const PackedTensorAccessor<scalar_t, 4> idata_acc,
    PackedTensorAccessor<scalar_t, 4> odata_acc,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels) {
  auto queue = getCurrentSYCLQueue();
  int64_t wg_size = std::min(syclMaxWorkGroupSize(), (int64_t)1024);
  int num_group = at::ceil_div(n, (int)wg_size);

  UpsampleBilinear2dOutFrameKernelFunctor<scalar_t, accscalar_t> kfn(
      n,
      rheight,
      rwidth,
      align_corners,
      idata_acc,
      odata_acc,
      input_height,
      input_width,
      output_height,
      output_width,
      nbatch,
      channels);

  sycl_kernel_submit(
      sycl::range<1>(num_group * wg_size), sycl::range<1>(wg_size), queue, kfn);
}

size_t idx(
    const size_t nc,
    const size_t height,
    const size_t width,
    const size_t y,
    const size_t x) {
  return (nc * height + y) * width + x;
}

template <typename scalar_t, typename accscalar_t>
struct UpsampleBilinear2dBackwardOutFrameKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    for (size_t index =
             item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);
         index < o_numel;
         index += item.get_local_range(0) * item.get_group_range(0)) {
      size_t index_temp = index;
      const int w2 = index_temp % output_width;
      index_temp /= output_width;
      const int h2 = index_temp % output_height;
      const size_t nc = index_temp / output_height;

      const accscalar_t h1r = area_pixel_compute_source_index<scalar_t>(
          rheight, h2, align_corners, /*cubic=*/false);
      const int h1 = h1r;
      const int h1p = (h1 < input_height - 1) ? 1 : 0;
      const accscalar_t h1lambda = h1r - h1;
      const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;

      const accscalar_t w1r = area_pixel_compute_source_index<scalar_t>(
          rwidth, w2, align_corners, /*cubic=*/false);
      const int w1 = w1r;
      const int w1p = (w1 < input_width - 1) ? 1 : 0;
      const accscalar_t w1lambda = w1r - w1;
      const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

      const scalar_t d2val = out_data[index];

      atomicAdd(
          (sycl_global_ptr<
              scalar_t>)(in_data + idx(nc, input_height, input_width, h1, w1)),
          static_cast<scalar_t>(h0lambda * w0lambda * d2val));

      atomicAdd(
          (sycl_global_ptr<
              scalar_t>)(in_data + idx(nc, input_height, input_width, h1, w1 + w1p)),
          static_cast<scalar_t>(h0lambda * w1lambda * d2val));

      atomicAdd(
          (sycl_global_ptr<
              scalar_t>)(in_data + idx(nc, input_height, input_width, h1 + h1p, w1)),
          static_cast<scalar_t>(h1lambda * w0lambda * d2val));

      atomicAdd(
          (sycl_global_ptr<
              scalar_t>)(in_data + idx(nc, input_height, input_width, h1 + h1p, w1 + w1p)),
          static_cast<scalar_t>(h1lambda * w1lambda * d2val));
    }
  }
  UpsampleBilinear2dBackwardOutFrameKernelFunctor(
      const size_t nc_,
      int64_t input_height_,
      int64_t input_width_,
      int64_t output_height_,
      int64_t output_width_,
      int64_t nbatch_,
      int64_t channels_,
      const accscalar_t rheight_,
      const accscalar_t rwidth_,
      const bool align_corners_,
      scalar_t* in_data_,
      const scalar_t* out_data_,
      const size_t o_numel_,
      const size_t i_numel_)
      : nc(nc_),
        input_height(input_height_),
        input_width(input_width_),
        output_height(output_height_),
        output_width(output_width_),
        nbatch(nbatch_),
        channels(channels_),
        rheight(rheight_),
        rwidth(rwidth_),
        align_corners(align_corners_),
        in_data(in_data_),
        out_data(out_data_),
        o_numel(o_numel_),
        i_numel(i_numel_) {}

 private:
  const size_t nc;
  int64_t input_height;
  int64_t input_width;
  int64_t output_height;
  int64_t output_width;
  int64_t nbatch;
  int64_t channels;
  const accscalar_t rheight;
  const accscalar_t rwidth;
  const bool align_corners;
  scalar_t* in_data;
  const scalar_t* out_data;
  const size_t o_numel;
  const size_t i_numel;
};

template <typename scalar_t, typename accscalar_t>
void upsample_bilinear2d_backward_out_frame(
    const size_t nc,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    scalar_t* idata,
    const scalar_t* odata) {
  auto queue = getCurrentSYCLQueue();
  int64_t wg_size = std::min(syclMaxWorkGroupSize(), (int64_t)1024);

  const size_t o_numel = nc * output_width * output_height;
  const size_t i_numel = nc * input_width * input_height;

  const size_t num_kernels = nc * output_width * output_height;
  int num_group = at::ceil_div((int64_t)num_kernels, (int64_t)wg_size);

  UpsampleBilinear2dBackwardOutFrameKernelFunctor<scalar_t, accscalar_t> kfn(
      nc,
      input_height,
      input_width,
      output_height,
      output_width,
      nbatch,
      channels,
      rheight,
      rwidth,
      align_corners,
      idata,
      odata,
      o_numel,
      i_numel);
  sycl_kernel_submit(
      sycl::range<1>(num_group * wg_size), sycl::range<1>(wg_size), queue, kfn);
}

void upsample_bilinear2d_out_kernel(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input.size(0);
  int channels = input.size(1);
  int input_height = input.size(2);
  int input_width = input.size(3);

  if (input.sizes() == output.sizes()) {
    output.copy_(input);
    return;
  }
  const int num_kernels = output_height * output_width;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "upsample_bilinear2d_out_kernel",
      [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        auto idata_acc = input.packed_accessor64<scalar_t, 4>();
        auto odata_acc = output.packed_accessor64<scalar_t, 4>();

        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);
        // TODO:a faster kernel for channel last
        upsample_bilinear2d_out_frame<scalar_t, accscalar_t>(
            num_kernels,
            rheight,
            rwidth,
            align_corners,
            idata_acc,
            odata_acc,
            input_height,
            input_width,
            output_height,
            output_width,
            nbatch,
            channels);
      });
}

void upsample_bilinear2d_backward_out_kernel(
    Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(__func__, {grad_output_arg, grad_input_arg});

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input_size[0];
  int channels = input_size[1];
  int input_height = input_size[2];
  int input_width = input_size[3];

  if (grad_input.numel() == 0) {
    return;
  }

  grad_input.zero_();

  if (grad_output_.sizes() == grad_input.sizes()) {
    grad_input.copy_(grad_output_);
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_output_.scalar_type(),
      "upsample_bilinear2d_backward_out_kernel",
      [&] {
        using accscalar_t = acc_type<scalar_t, true>;

        Tensor grad_input_c = grad_input.is_contiguous()
            ? grad_input
            : at::zeros(grad_input.sizes(), grad_input.options());
        Tensor grad_output = grad_output_.contiguous();

        scalar_t* idata = grad_input_c.data_ptr<scalar_t>();
        scalar_t* odata = grad_output.data_ptr<scalar_t>();

        const accscalar_t rheight = area_pixel_compute_scale<scalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<scalar_t>(
            input_width, output_width, align_corners, scales_w);
        // TODO: a faster kernel for channel last
        upsample_bilinear2d_backward_out_frame<scalar_t, accscalar_t>(
            nbatch * channels,
            input_height,
            input_width,
            output_height,
            output_width,
            nbatch,
            channels,
            rheight,
            rwidth,
            align_corners,
            idata,
            odata);
        if (!grad_input.is_contiguous()) {
          grad_input.copy_(grad_input_c);
        }
      });
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop