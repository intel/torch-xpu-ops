#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/xpu/UpSample.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <comm/SYCLContext.h>
#include <ATen/Context.h>
#include <ATen/core/TensorBase.h>

#include <ATen/native/xpu/sycl/UpSampleLinear1dKernels.h>

namespace at::native::xpu {
template <typename scalar_t, typename accscalar_t>
struct UpsampleLinear1dKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int index =
        item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);

    const int batchsize = idata_.size(0);
    const int channels = idata_.size(1);
    const int width1 = idata_.size(2);
    const int width2 = odata_.size(2);
    PackedTensorAccessor64<scalar_t, 3> odata_res = odata_;

    if (index < n_) {
      const int w2 = index % width2;
      // special case: just copy
      if (width1 == width2) {
        const int w1 = w2;
        for (int n = 0; n < batchsize; n++) {
          for (int c = 0; c < channels; ++c) {
            const scalar_t val = idata_[n][c][w1];
            odata_res[n][c][w2] = val;
          }
        }
        return;
      }

      const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
          rwidth_, w2, align_corners_, /*cubic=*/false);
      const int w1 = w1r;
      const int w1p = (w1 < width1 - 1) ? 1 : 0;
      const accscalar_t w1lambda = w1r - w1;
      const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const accscalar_t val =
              w0lambda * idata_[n][c][w1] + w1lambda * idata_[n][c][w1 + w1p];
          odata_res[n][c][w2] = static_cast<scalar_t>(val);
        }
      }
    }
  }
  UpsampleLinear1dKernelFunctor(
      const int n,
      const accscalar_t rwidth,
      const bool align_corners,
      const PackedTensorAccessor64<const scalar_t, 3> idata,
      PackedTensorAccessor64<scalar_t, 3> odata)
      : n_(n),
        rwidth_(rwidth),
        align_corners_(align_corners),
        idata_(idata),
        odata_(odata) {}

 private:
  const int n_;
  const accscalar_t rwidth_;
  const bool align_corners_;
  const PackedTensorAccessor64<const scalar_t, 3> idata_;
  PackedTensorAccessor64<scalar_t, 3> odata_;
};

void upsample_linear1d_kernel(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales,
    const Tensor& output) {
  int output_width = output_size[0];
  output.zero_();
  int input_width = input.size(2);

  AT_ASSERT(input_width > 0 && output_width > 0);

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
        UpsampleLinear1dKernelFunctor<scalar_t, accscalar_t> kfn(
            num_kernels, rwidth, align_corners, idata, odata);
        const auto local_range = syclMaxWorkGroupSize(kfn);
        auto global_range =
            (num_kernels + local_range - 1) / local_range * local_range;
        sycl_kernel_submit(
            global_range, local_range, getCurrentSYCLQueue(), kfn);
      });
}

template <typename scalar_t, typename accscalar_t>
struct UpsampleLinear1dBackwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int index =
        item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);

    const int batchsize = idata_.size(0);
    const int channels = idata_.size(1);
    const int width1 = idata_.size(2);
    const int width2 = odata_.size(2);
    PackedTensorAccessor64<scalar_t, 3> idata_res = idata_;
    if (index < n_) {
      const int w2 = index % width2;
      if (width1 == width2) {
        const int w1 = w2;
        for (int n = 0; n < batchsize; n++) {
          for (int c = 0; c < channels; ++c) {
            const scalar_t val = odata_[n][c][w1];
            idata_res[n][c][w2] = val;
          }
        }
        return;
      }
      const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
          rwidth_, w2, align_corners_, /*cubic=*/false);
      const int w1 = w1r;
      const int w1p = (w1 < width1 - 1) ? 1 : 0;
      const accscalar_t w1lambda = w1r - w1;
      const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t d2val = odata_[n][c][w2];
          atomicAdd(
              (sycl_global_ptr<scalar_t>)(&idata_res[n][c][w1]),
              static_cast<scalar_t>(w0lambda * d2val));
          atomicAdd(
              (sycl_global_ptr<scalar_t>)(&idata_res[n][c][w1 + w1p]),
              static_cast<scalar_t>(w1lambda * d2val));
        }
      }
    }
  }
  UpsampleLinear1dBackwardKernelFunctor(
      const int n,
      const accscalar_t rwidth,
      const bool align_corners,
      PackedTensorAccessor64<scalar_t, 3> idata,
      const PackedTensorAccessor64<const scalar_t, 3> odata)
      : n_(n),
        rwidth_(rwidth),
        align_corners_(align_corners),
        idata_(idata),
        odata_(odata) {}

 private:
  const int n_;
  const accscalar_t rwidth_;
  const bool align_corners_;
  PackedTensorAccessor64<scalar_t, 3> idata_;
  const PackedTensorAccessor64<const scalar_t, 3> odata_;
};

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
        UpsampleLinear1dBackwardKernelFunctor<scalar_t, accscalar_t> kfn(
            num_kernels, rwidth, align_corners, idata, odata);
        const auto local_range = syclMaxWorkGroupSize(kfn);
        auto global_range =
            (num_kernels + local_range - 1) / local_range * local_range;
        sycl_kernel_submit(
            global_range, local_range, getCurrentSYCLQueue(), kfn);
      });
}
} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
