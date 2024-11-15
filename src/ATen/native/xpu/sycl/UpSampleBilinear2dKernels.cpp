#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/ceil_div.h>
#include <comm/xpu_aten.h>
#include <ATen/native/xpu/sycl/LaunchUtils.h>

#include <ATen/native/xpu/UpSample.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/UpSampleBilinear2dKernels.h>

namespace at::native::xpu {

template <typename scalar_t, typename accscalar_t>
struct UpsampleBilinear2dKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int index = item.get_global_linear_id();

    if (index < n_) {
      const int output_x = index % output_width_;
      const int output_y = index / output_width_;

      const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
          rheight_, output_y, align_corners_, /*cubic=*/false);
      const int h1 = h1r;
      const int h1p = (h1 < input_height_ - 1) ? 1 : 0;
      const accscalar_t h1lambda = h1r - h1;
      const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;

      const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
          rwidth_, output_x, align_corners_, /*cubic=*/false);
      const int w1 = w1r;
      const int w1p = (w1 < input_width_ - 1) ? 1 : 0;
      const accscalar_t w1lambda = w1r - w1;
      const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
      auto odata = out_data_acc_;
      for (int n = 0; n < nbatch_; n++) {
        for (int c = 0; c < channels_; ++c) {
          const accscalar_t val = h0lambda *
                  (w0lambda * in_data_acc_[n][c][h1][w1] +
                   w1lambda * in_data_acc_[n][c][h1][w1 + w1p]) +
              h1lambda *
                  (w0lambda * in_data_acc_[n][c][h1 + h1p][w1] +
                   w1lambda * in_data_acc_[n][c][h1 + h1p][w1 + w1p]);
          odata[n][c][output_y][output_x] = static_cast<scalar_t>(val);
        }
      }
    }
  }
  UpsampleBilinear2dKernelFunctor(
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
      int64_t channels)
      : n_(n),
        rheight_(rheight),
        rwidth_(rwidth),
        align_corners_(align_corners),
        in_data_acc_(idata_acc),
        out_data_acc_(odata_acc),
        input_height_(input_height),
        input_width_(input_width),
        output_height_(output_height),
        output_width_(output_width),
        nbatch_(nbatch),
        channels_(channels) {}

 private:
  const int n_;
  const accscalar_t rheight_;
  const accscalar_t rwidth_;
  const bool align_corners_;
  const PackedTensorAccessor<scalar_t, 4> in_data_acc_;
  PackedTensorAccessor<scalar_t, 4> out_data_acc_;
  int64_t input_height_;
  int64_t input_width_;
  int64_t output_height_;
  int64_t output_width_;
  int64_t nbatch_;
  int64_t channels_;
};

template <typename scalar_t, typename accscalar_t>
void launch_upsample_bilinear2d_kernel(
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
  UpsampleBilinear2dKernelFunctor<scalar_t, accscalar_t> kfn(
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

  int64_t wg_size = syclMaxWorkGroupSize(kfn);
  int num_group = at::ceil_div(n, (int)wg_size);
  auto queue = getCurrentSYCLQueue();

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
struct UpsampleBilinear2dBackwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    for (size_t index =
             item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);
         index < o_numel_;
         index += item.get_local_range(0) * item.get_group_range(0)) {
      size_t index_temp = index;
      const int w2 = index_temp % output_width_;
      index_temp /= output_width_;
      const int h2 = index_temp % output_height_;
      const size_t nc = index_temp / output_height_;

      const accscalar_t h1r = area_pixel_compute_source_index<scalar_t>(
          rheight_, h2, align_corners_, /*cubic=*/false);
      const int h1 = h1r;
      const int h1p = (h1 < input_height_ - 1) ? 1 : 0;
      const accscalar_t h1lambda = h1r - h1;
      const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;

      const accscalar_t w1r = area_pixel_compute_source_index<scalar_t>(
          rwidth_, w2, align_corners_, /*cubic=*/false);
      const int w1 = w1r;
      const int w1p = (w1 < input_width_ - 1) ? 1 : 0;
      const accscalar_t w1lambda = w1r - w1;
      const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

      const scalar_t d2val = out_data_[index];

      atomicAdd(
          (sycl_global_ptr<
              scalar_t>)(in_data_ + idx(nc, input_height_, input_width_, h1, w1)),
          static_cast<scalar_t>(h0lambda * w0lambda * d2val));

      atomicAdd(
          (sycl_global_ptr<
              scalar_t>)(in_data_ + idx(nc, input_height_, input_width_, h1, w1 + w1p)),
          static_cast<scalar_t>(h0lambda * w1lambda * d2val));

      atomicAdd(
          (sycl_global_ptr<
              scalar_t>)(in_data_ + idx(nc, input_height_, input_width_, h1 + h1p, w1)),
          static_cast<scalar_t>(h1lambda * w0lambda * d2val));

      atomicAdd(
          (sycl_global_ptr<
              scalar_t>)(in_data_ + idx(nc, input_height_, input_width_, h1 + h1p, w1 + w1p)),
          static_cast<scalar_t>(h1lambda * w1lambda * d2val));
    }
  }
  UpsampleBilinear2dBackwardKernelFunctor(
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
      scalar_t* in_data,
      const scalar_t* out_data,
      const size_t o_numel,
      const size_t i_numel)
      : nc_(nc),
        input_height_(input_height),
        input_width_(input_width),
        output_height_(output_height),
        output_width_(output_width),
        nbatch_(nbatch),
        channels_(channels),
        rheight_(rheight),
        rwidth_(rwidth),
        align_corners_(align_corners),
        in_data_(in_data),
        out_data_(out_data),
        o_numel_(o_numel),
        i_numel_(i_numel) {}

 private:
  const size_t nc_;
  int64_t input_height_;
  int64_t input_width_;
  int64_t output_height_;
  int64_t output_width_;
  int64_t nbatch_;
  int64_t channels_;
  const accscalar_t rheight_;
  const accscalar_t rwidth_;
  const bool align_corners_;
  scalar_t* in_data_;
  const scalar_t* out_data_;
  const size_t o_numel_;
  const size_t i_numel_;
};

template <typename scalar_t, typename accscalar_t>
void launch_upsample_bilinear2d_backward_kernel(
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
  const size_t o_numel = nc * output_width * output_height;
  const size_t i_numel = nc * input_width * input_height;

  const size_t num_kernels = nc * output_width * output_height;

  UpsampleBilinear2dBackwardKernelFunctor<scalar_t, accscalar_t> kfn(
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

  int64_t wg_size = syclMaxWorkGroupSize(kfn);
  int num_group = at::ceil_div((int64_t)num_kernels, (int64_t)wg_size);
  auto queue = getCurrentSYCLQueue();

  sycl_kernel_submit(
      sycl::range<1>(num_group * wg_size), sycl::range<1>(wg_size), queue, kfn);
}

void upsample_bilinear2d_out_kernel(
    const Tensor& output,
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
      "upsample_bilinear2d_xpu",
      [&] {
        using accscalar_t = acc_type_device<scalar_t, kXPU>;
        auto idata_acc = input.packed_accessor64<scalar_t, 4>();
        auto odata_acc = output.packed_accessor64<scalar_t, 4>();

        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);

        // TODO:a faster kernel for channel last
        launch_upsample_bilinear2d_kernel<scalar_t, accscalar_t>(
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
    const Tensor& grad_input,
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
      "upsample_bilinear2d_backward_xpu",
      [&] {
        using accscalar_t = acc_type_device<scalar_t, kXPU>;

        // TODO: using PackedTensorAccessor instead of copy
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
        launch_upsample_bilinear2d_backward_kernel<scalar_t, accscalar_t>(
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

template <typename scalar_t, typename accscalar_t, typename InterpFilter>
struct UpsampleGen2dAaKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<3> item) const {
    const int output_x = item.get_global_id(2);
    const int output_y = item.get_global_id(1);

    if (output_x >= output_width_ || output_y >= output_height_) {
      return;
    }

    const int interp_height = (int)ceilf(support_h_) * 2 + 1;
    const int interp_width = (int)ceilf(support_w_) * 2 + 1;

    // threadIdx.x + blockIdx.x * blockDim.x;
    // item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);
    scalar_t* wx = shared_ + interp_width * item.get_local_id(2);
    scalar_t* wy = shared_ + interp_width * item.get_local_range(2) + interp_height * item.get_local_id(1);
    const int offset = interp_width * item.get_local_range(2) + interp_height * item.get_local_range(1);
    scalar_t *buffer2 = shared_ + offset + interp_height * (item.get_local_id(2) + item.get_local_id(1) * item.get_local_range(2));

    int xmin, xsize, ymin, ysize;
    accscalar_t xcenter, ycenter;
    upsample_antialias::_compute_weights_span(
      output_x, input_width_, width_scale_, support_w_, xmin, xsize, xcenter);
    upsample_antialias::_compute_weights_span(
      output_y, input_height_, height_scale_, support_h_, ymin, ysize, ycenter);

    if (item.get_local_id(1) == 0)
    {
      // All threadIdx.y have the same wx weights
      upsample_antialias::_compute_weights<scalar_t, accscalar_t>(
          wx,
          width_scale_,
          interp_width,
          interp_filter_,
          xmin - xcenter,
          xsize);
    }

    if (item.get_local_id(2) == 0)
    {
      // All threadIdx.x have the same wy weights
      upsample_antialias::_compute_weights<scalar_t, accscalar_t>(
          wy,
          height_scale_,
          interp_height,
          interp_filter_,
          ymin - ycenter,
          ysize);
    }

    item.barrier(sycl_local_fence);

    const scalar_t * buffer1;

    // Parallelized across batch/channels
    for (int i = item.get_group(0); i < batchsize_ * channels_; i += item.get_global_range(0)) {
      int n = i / channels_;
      int c = i % channels_;
      // interpolate on y-axis for ymin to ymin + ysize
      for (int y = 0; y < ysize; y++) {
        buffer1 = &(idata_[n][c][ymin + y][xmin]);
        buffer2[y] = static_cast<scalar_t>(
            upsample_antialias::interpolate_aa_single_dim<scalar_t, accscalar_t>(
                buffer1, wx, xsize));
      }
      odata_[n][c][output_y][output_x] = static_cast<scalar_t>(
          upsample_antialias::interpolate_aa_single_dim<scalar_t, accscalar_t>(
              buffer2, wy, ysize));
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<scalar_t>(local_size_, cgh);
  }

  UpsampleGen2dAaKernelFunctor(
      const accscalar_t height_scale,
      const accscalar_t width_scale,
      const PackedTensorAccessor<scalar_t, 4> idata,
      PackedTensorAccessor<scalar_t, 4> odata,
      const InterpFilter & interp_filter,
      int64_t input_height,
      int64_t input_width,
      int64_t output_height,
      int64_t output_width,
      int64_t batchsize,
      int64_t channels,
      const accscalar_t support_h,
      const accscalar_t support_w,
      int64_t local_size)
      : height_scale_(height_scale),
        width_scale_(width_scale),
        idata_(idata),
        odata_(odata),
        interp_filter_(interp_filter),
        input_height_(input_height),
        input_width_(input_width),
        output_height_(output_height),
        output_width_(output_width),
        batchsize_(batchsize),
        channels_(channels),
        support_h_(support_h),
        support_w_(support_w),
        local_size_(local_size) {}

 private:
  const accscalar_t height_scale_;
  const accscalar_t width_scale_;
  const PackedTensorAccessor<scalar_t, 4> idata_;
  PackedTensorAccessor<scalar_t, 4> odata_;
  const InterpFilter & interp_filter_;
  int64_t input_height_;
  int64_t input_width_;
  int64_t output_height_;
  int64_t output_width_;
  int64_t batchsize_;
  int64_t channels_;
  const accscalar_t support_h_;
  const accscalar_t support_w_;
  int64_t local_size_;
  sycl_local_acc_t<accscalar_t, 1> shared_;
};

template<typename InterpFilter>
void upsample_gen2d_aa_out_kernel(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TensorArg input_arg{input_, "input_", 1}, output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});

  // TODO: remove this when the kernel is updated to support the channels_last memory format.
  auto output_c = output.is_contiguous() ? output : at::empty(output.sizes(), output.options());
  auto input = input_.contiguous();
  int output_height = output_size[0];
  int output_width = output_size[1];
  int input_height = input.size(2);
  int input_width = input.size(3);
  int nbatch = input.size(0);
  int channels = input.size(1);

  auto sharedMemPerBlock = syclLocalMemSize();
  auto total_threads = syclMaxWorkItemsPerTile();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "upsample_gen2d_aa_xpu",
      [&] {
        using accscalar_t = acc_type_device<scalar_t, kXPU>;
        auto idata = input.packed_accessor64<const scalar_t, 4>();
        auto odata = output_c.packed_accessor64<scalar_t, 4>();

        const accscalar_t height_scale = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t width_scale = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);

        auto interp_filter = InterpFilter();
        const accscalar_t support_h = static_cast<accscalar_t>(
            (height_scale >= 1.0) ? (interp_filter.size * 0.5) * height_scale : interp_filter.size * 0.5);
        const accscalar_t support_w = static_cast<accscalar_t>(
            (width_scale >= 1.0) ? (interp_filter.size * 0.5) * width_scale : interp_filter.size * 0.5);
        const int interp_height = (int)ceilf(support_h) * 2 + 1;
        const int interp_width = (int)ceilf(support_w) * 2 + 1;

        // TODO: test if 256 really works better
        int maxThreadsPerBlock = std::min(syclMaxWorkGroupSize<UpsampleGen2dAaKernelFunctor<scalar_t, accscalar_t, InterpFilter>>(), 256);
        int block_x = syclMaxSubGroupSize();  // TODO: tune subgroup size (and fix it), or merge x,y
	
	int numer = sharedMemPerBlock * 1.0 / sizeof(scalar_t) - interp_width * block_x;
        int denom = interp_height * (block_x + 1);
        int block_y = lastPow2((unsigned int) (numer / denom));
        block_y = std::min<int>(maxThreadsPerBlock / block_x, block_y);

        int grid_x = std::min<int>(total_threads, (output_width + block_x - 1) / block_x * block_x);
        int grid_y = std::min<int>(total_threads / grid_x, (output_height + block_y - 1) / block_y * block_y);
        int grid_z = std::min<int>(total_threads / grid_x / grid_y, nbatch * channels);

        int64_t weights_per_block = interp_width * block_x + interp_height * block_y;
        weights_per_block += interp_height * block_y * block_x;
        int64_t shmem_size = weights_per_block * sizeof(scalar_t);
        TORCH_CHECK(
            shmem_size <= sharedMemPerBlock,
            "Provided interpolation parameters can not be handled with current algorithm implementation. ",
            "Please reduce the scale factor. Too much shared memory required: ",
            shmem_size, " vs ", sharedMemPerBlock);

        UpsampleGen2dAaKernelFunctor<scalar_t, accscalar_t, InterpFilter> kfn(
            height_scale,
            width_scale,
            idata,
            odata,
            interp_filter,
            input_height,
            input_width,
            output_height,
            output_width,
            nbatch,
            channels,
            support_h,
            support_w,
            shmem_size);

	auto queue = getCurrentSYCLQueue();
        sycl_kernel_submit(
          sycl::range<3>(grid_z, grid_y, grid_x),
          sycl::range<3>(1, block_y, block_x),
          queue,
          kfn);
      });

  if (!output.is_contiguous()) {
       output.copy_(output_c);
  }
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
