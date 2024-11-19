#include <ATen/AccumulateType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/LaunchUtils.h>
#include <ATen/native/xpu/sycl/UpSampleNearest2dKernels.h>
#include <comm/Runtime.h>
#include <comm/SYCLContext.h>
#include <comm/SYCLHelpers.h>

namespace at::native {
namespace xpu {

template <typename scalar_t, typename accscalar_t, typename index_bw_op_t>
struct UpsampleNearest2dBackwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int dst_idx = item.get_global_linear_id();
    if (dst_idx >= dim_c_ * dst_dim_h_ * dst_dim_w_)
      return;

    int dst_c_stride = dst_dim_h_ * dst_dim_w_;
    int src_c_stride = src_dim_h_ * src_dim_w_;

    int c = (dst_idx / (dst_c_stride)) % dim_c_;

    int dst_y = (dst_idx / dst_dim_w_) % dst_dim_h_;
    // note that we do not want to clamp src_y to src_dim_y, since we might
    // intentionally want to skip in case of scale_factor < 1.0
    int src_y = index_bw_op_(height_scale_, dst_y, src_dim_h_);
    int src_y_up = index_bw_op_(height_scale_, dst_y + 1, src_dim_h_);

    int dst_x = dst_idx % dst_dim_w_;
    // note that we do not want to clamp src_x to src_dim_w_, since we might
    // intentionally want to skip in case of scale_factor < 1.0
    int src_x = index_bw_op_(width_scale_, dst_x, src_dim_w_);
    int src_x_up = index_bw_op_(width_scale_, dst_x + 1, src_dim_w_);

    for (int b = 0; b < dim_b_; b++) {
      accscalar_t grad = 0;
      for (int y = src_y; y < src_y_up; y++) {
        for (int x = src_x; x < src_x_up; x++) {
          int src_idx =
              b * dim_c_ * src_c_stride + c * src_c_stride + y * src_dim_w_ + x;
          grad += grad_o_[src_idx];
        }
      }
      grad_i_[dst_idx] = grad;
      dst_idx += dim_c_ * dst_c_stride;
    }
  }
  UpsampleNearest2dBackwardKernelFunctor(
      size_t n,
      const scalar_t* grad_o,
      size_t dim_b,
      size_t dim_c,
      size_t src_dim_h,
      size_t src_dim_w,
      size_t dst_dim_h,
      size_t dst_dim_w,
      scalar_t* grad_i,
      float height_scale,
      float width_scale,
      index_bw_op_t index_bw_op)
      : n_(n),
        grad_o_(grad_o),
        dim_b_(dim_b),
        dim_c_(dim_c),
        src_dim_h_(src_dim_h),
        src_dim_w_(src_dim_w),
        dst_dim_h_(dst_dim_h),
        dst_dim_w_(dst_dim_w),
        grad_i_(grad_i),
        height_scale_(height_scale),
        width_scale_(width_scale),
        index_bw_op_(index_bw_op) {}

 private:
  size_t n_;
  const scalar_t* grad_o_;
  size_t dim_b_;
  size_t dim_c_;
  size_t src_dim_h_;
  size_t src_dim_w_;
  size_t dst_dim_h_;
  size_t dst_dim_w_;
  scalar_t* grad_i_;
  float height_scale_;
  float width_scale_;
  index_bw_op_t index_bw_op_;
};

template <typename scalar_t, typename accscalar_t, typename index_bw_op_t>
void upsample_nearest2d_backward_frame(
    size_t n,
    const scalar_t* grad_o,
    size_t dim_b,
    size_t dim_c,
    size_t src_dim_h,
    size_t src_dim_w,
    size_t dst_dim_h,
    size_t dst_dim_w,
    scalar_t* grad_i,
    float height_scale,
    float width_scale,
    index_bw_op_t index_bw_op) {
  auto& queue = at::xpu::getCurrentSYCLQueue();
  auto work_group_size = syclMaxWorkItemsPerEU();
  int global_range =
      (n + work_group_size - 1) / work_group_size * work_group_size;
  auto caller = UpsampleNearest2dBackwardKernelFunctor<
      scalar_t,
      accscalar_t,
      index_bw_op_t>(
      n,
      grad_o,
      dim_b,
      dim_c,
      src_dim_h,
      src_dim_w,
      dst_dim_h,
      dst_dim_w,
      grad_i,
      height_scale,
      width_scale,
      index_bw_op);
  sycl_kernel_submit(global_range, work_group_size, queue, caller);
}

template <typename scalar_t, typename accscalar_t, typename index_bw_op_t>
struct UpsampleNearest2dBackwardChannelsLastKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int index = item.get_global_linear_id();

    if (index < gi_numel_) {
      const int c = index % channels_;
      const int w2 = (index / channels_) % width2_;
      const int h2 = (index / channels_ / width2_) % height2_;
      const int n = index / channels_ / width2_ / height2_;

      int h1 = index_bw_op_(height_scale_, h2, height1_);
      int h1_up = index_bw_op_(height_scale_, h2 + 1, height1_);

      int w1 = index_bw_op_(width_scale_, w2, width1_);
      int w1_up = index_bw_op_(width_scale_, w2 + 1, width1_);

      accscalar_t grad = 0;
      for (int ih = h1; ih < h1_up; ih++) {
        for (int iw = w1; iw < w1_up; iw++) {
          grad += go_[idx_cl(n, ih, iw, c, height1_, width1_, channels_)];
        }
      }
      gi_[index] = static_cast<scalar_t>(grad);
    }
  }
  UpsampleNearest2dBackwardChannelsLastKernelFunctor(
      const scalar_t* go,
      scalar_t* gi,
      const size_t height1,
      const size_t width1,
      const size_t height2,
      const size_t width2,
      const size_t channels,
      const float height_scale,
      const float width_scale,
      const size_t gi_numel,
      index_bw_op_t index_bw_op)
      : go_(go),
        gi_(gi),
        height1_(height1),
        width1_(width1),
        height2_(height2),
        width2_(width2),
        channels_(channels),
        height_scale_(height_scale),
        width_scale_(width_scale),
        gi_numel_(gi_numel),
        index_bw_op_(index_bw_op) {}

 private:
  const scalar_t* go_;
  scalar_t* gi_;
  const size_t height1_;
  const size_t width1_;
  const size_t height2_;
  const size_t width2_;
  const size_t channels_;
  const float height_scale_;
  const float width_scale_;
  const size_t gi_numel_;
  index_bw_op_t index_bw_op_;
};

template <typename scalar_t, typename accscalar_t, typename index_bw_op_t>
void upsample_nearest2d_backward_channels_last_frame(
    const scalar_t* go,
    scalar_t* gi,
    const size_t height1,
    const size_t width1,
    const size_t height2,
    const size_t width2,
    const size_t channels,
    const float height_scale,
    const float width_scale,
    const size_t gi_numel,
    index_bw_op_t index_bw_op) {
  auto& queue = at::xpu::getCurrentSYCLQueue();
  auto work_group_size = syclMaxWorkItemsPerEU();
  int global_range =
      (gi_numel + work_group_size - 1) / work_group_size * work_group_size;
  auto caller = UpsampleNearest2dBackwardChannelsLastKernelFunctor<
      scalar_t,
      accscalar_t,
      index_bw_op_t>(
      go,
      gi,
      height1,
      width1,
      height2,
      width2,
      channels,
      height_scale,
      width_scale,
      gi_numel,
      index_bw_op);
  sycl_kernel_submit(global_range, work_group_size, queue, caller);
}

void upsample_nearest2d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    bool is_exact) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1};
  TensorArg grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(__func__, {grad_input_arg, grad_output_arg});
  if (grad_input.numel() == 0) {
    return;
  }

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input_size[0];
  int channels = input_size[1];
  int input_height = input_size[2];
  int input_width = input_size[3];

  const float height_scale = compute_scales_value_backwards<float>(
      scales_h, output_height, input_height);
  const float width_scale = compute_scales_value_backwards<float>(
      scales_w, output_width, input_width);

  auto memory_format = grad_output_.suggest_memory_format();

  if (grad_output_.sizes() == grad_input.sizes()) {
    grad_input.copy_(grad_output_);
    return;
  }

  if (memory_format == at::MemoryFormat::ChannelsLast && channels >= 4 &&
      grad_input.is_contiguous(memory_format)) {
    Tensor grad_output =
        grad_output_.contiguous(at::MemoryFormat::ChannelsLast);

    TORCH_CHECK(
        grad_input.numel() < std::numeric_limits<int>::max(),
        "upsample_nearest_channels_last only supports grad_input tensors with less than INT_MAX elements");
    TORCH_CHECK(
        grad_output.numel() < std::numeric_limits<int>::max(),
        "upsample_nearest_channels_last only supports grad_output tensors with less than INT_MAX elements");

    AT_DISPATCH_FLOATING_TYPES_AND3(
        ScalarType::BFloat16,
        ScalarType::Half,
        ScalarType::Byte,
        grad_output.scalar_type(),
        "upsample_nearest2d_backward_channels_last_xpu",
        [&] {
          using accscalar_t = acc_type_device<scalar_t, kXPU>;

          const scalar_t* go = grad_output.const_data_ptr<scalar_t>();
          scalar_t* gi = grad_input.mutable_data_ptr<scalar_t>();
          if (is_exact) {
            upsample_nearest2d_backward_channels_last_frame<
                scalar_t,
                accscalar_t>(
                go,
                gi,
                output_height,
                output_width,
                input_height,
                input_width,
                channels,
                height_scale,
                width_scale,
                grad_input.numel(),
                NearestExactBwIndexOp());
          } else {
            upsample_nearest2d_backward_channels_last_frame<
                scalar_t,
                accscalar_t>(
                go,
                gi,
                output_height,
                output_width,
                input_height,
                input_width,
                channels,
                height_scale,
                width_scale,
                grad_input.numel(),
                NearestBwIndexOp());
          }
        });
  } else {
    // This is needed for non-contiguous tensors.
    Tensor grad_input_c = grad_input.is_contiguous()
        ? grad_input
        : at::empty(grad_input.sizes(), grad_input.options());
    Tensor grad_output = grad_output_.contiguous();
    unsigned int n = grad_input.numel() / nbatch;

    // upsample_nearest2d meta call makes sure `nbatch != 0`
    AT_DISPATCH_FLOATING_TYPES_AND3(
        ScalarType::BFloat16,
        ScalarType::Half,
        ScalarType::Byte,
        grad_output.scalar_type(),
        "upsample_nearest2d_backward_xpu",
        [&] {
          using accscalar_t = acc_type_device<scalar_t, kXPU>;

          auto idata = grad_input_c.mutable_data_ptr<scalar_t>();
          auto odata = grad_output.const_data_ptr<scalar_t>();
          if (is_exact) {
            upsample_nearest2d_backward_frame<scalar_t, accscalar_t>(
                n,
                odata,
                nbatch,
                channels,
                output_height,
                output_width,
                input_height,
                input_width,
                idata,
                height_scale,
                width_scale,
                NearestExactBwIndexOp());

          } else {
            upsample_nearest2d_backward_frame<scalar_t, accscalar_t>(
                n,
                odata,
                nbatch,
                channels,
                output_height,
                output_width,
                input_height,
                input_width,
                idata,
                height_scale,
                width_scale,
                NearestBwIndexOp());
          }
        });

    if (!grad_input.is_contiguous()) {
      grad_input.copy_(grad_input_c);
    }
  }
}

template <typename scalar_t, typename index_op_t>
struct UpsampleNearest2dKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    size_t nc_idx = item.get_global_id(0);
    int h2 = item.get_global_id(1);
    int w2 = item.get_global_id(2);

    if (w2 >= width2_ || h2 >= height2_) {
      return;
    }

    int nc_range = item.get_global_range(0);

    const size_t h1 =
        height1_ == height2_ ? h2 : index_op_(height_scale_, h2, height1_);
    const size_t w1 =
        width1_ == width2_ ? w2 : index_op_(width_scale_, w2, width1_);

    size_t src_index = (nc_idx * height1_ + h1) * width1_ + w1;
    size_t src_index_stride = nc_range * width1_ * height1_;
    size_t dst_index = (nc_idx * height2_ + h2) * width2_ + w2;
    size_t dst_index_stride = nc_range * width2_ * height2_;

    // iterating over
    while (nc_idx < nc_) {
      odata_[dst_index] = idata_[src_index];
      dst_index += dst_index_stride;
      src_index += src_index_stride;
      nc_idx += nc_range;
    }
  }
  UpsampleNearest2dKernelFunctor(
      const scalar_t* idata,
      scalar_t* odata,
      const size_t nc,
      const size_t height1,
      const size_t width1,
      const size_t height2,
      const size_t width2,
      float height_scale,
      float width_scale,
      index_op_t index_op)
      : idata_(idata),
        odata_(odata),
        nc_(nc),
        height1_(height1),
        width1_(width1),
        height2_(height2),
        width2_(width2),
        height_scale_(height_scale),
        width_scale_(width_scale),
        index_op_(index_op) {}

 private:
  const scalar_t* idata_;
  scalar_t* odata_;
  const size_t nc_;
  const size_t height1_;
  const size_t width1_;
  const size_t height2_;
  const size_t width2_;
  float height_scale_;
  float width_scale_;
  index_op_t index_op_;
};

template <typename scalar_t, typename index_op_t>
void upsample_nearest2d_frame(
    const scalar_t* idata,
    scalar_t* odata,
    const size_t nc,
    const size_t height1, // input height
    const size_t width1,
    const size_t height2, // output height
    const size_t width2,
    float height_scale,
    float width_scale,
    index_op_t index_op) {
  auto& queue = at::xpu::getCurrentSYCLQueue();

  auto work_group_size = syclMaxWorkItemsPerEU();
  int local_x = std::min<int>(lastPow2(width2), work_group_size);
  int local_y = std::min<int>(lastPow2(height2), work_group_size / local_x);
  int local_z = std::min<int>(nc, work_group_size / local_x / local_y);

  int global_x = (width2 + local_x - 1) / local_x * local_x;
  int global_y = (height2 + local_y - 1) / local_y * local_y;
  int z_plane = local_z * 4;
  int global_z = (nc + z_plane - 1) / z_plane * z_plane;

  auto kfn = UpsampleNearest2dKernelFunctor<scalar_t, index_op_t>(
      idata,
      odata,
      nc,
      height1,
      width1,
      height2,
      width2,
      height_scale,
      width_scale,
      index_op);

  sycl_kernel_submit(
      sycl::range<3>(global_z, global_y, global_x),
      sycl::range<3>(local_z, local_y, local_x),
      queue,
      kfn);
}

template <typename scalar_t, typename index_op_t>
struct UpsampleNearest2dChannelsLastKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    const int index = item.get_global_linear_id();

    if (index < out_numel_) {
      const int c = index % channels_;
      const int w2 = (index / channels_) % width2_;
      const int h2 = (index / channels_ / width2_) % height2_;
      const int n = index / channels_ / width2_ / height2_;

      const size_t h1 =
          height1_ == height2_ ? h2 : index_op_(height_scale_, h2, height1_);
      const size_t w1 =
          width1_ == width2_ ? w2 : index_op_(width_scale_, w2, width1_);

      odata_[index] =
          idata_[idx_cl(n, h1, w1, c, height1_, width1_, channels_)];
    }
  }
  UpsampleNearest2dChannelsLastKernelFunctor(
      const scalar_t* idata,
      scalar_t* odata,
      const size_t channels,
      const size_t height1,
      const size_t width1,
      const size_t height2,
      const size_t width2,
      float height_scale,
      float width_scale,
      const size_t out_numel,
      index_op_t index_op)
      : idata_(idata),
        odata_(odata),
        channels_(channels),
        height1_(height1),
        width1_(width1),
        height2_(height2),
        width2_(width2),
        height_scale_(height_scale),
        width_scale_(width_scale),
        out_numel_(out_numel),
        index_op_(index_op) {}

 private:
  const scalar_t* idata_;
  scalar_t* odata_;
  const size_t channels_;
  const size_t height1_;
  const size_t width1_;
  const size_t height2_;
  const size_t width2_;
  float height_scale_;
  float width_scale_;
  const size_t out_numel_;
  index_op_t index_op_;
};

template <typename scalar_t, typename index_op_t>
void upsample_nearest2d_channels_last_frame(
    const scalar_t* idata,
    scalar_t* odata,
    const size_t channels,
    const size_t height1,
    const size_t width1,
    const size_t height2,
    const size_t width2,
    float height_scale,
    float width_scale,
    const size_t out_numel,
    index_op_t index_op) {
  auto& queue = at::xpu::getCurrentSYCLQueue();

  auto work_group_size = syclMaxWorkItemsPerEU();
  int global_range =
      (out_numel + work_group_size - 1) / work_group_size * work_group_size;

  auto kfn = UpsampleNearest2dChannelsLastKernelFunctor<scalar_t, index_op_t>(
      idata,
      odata,
      channels,
      height1,
      width1,
      height2,
      width2,
      height_scale,
      width_scale,
      out_numel,
      index_op);

  sycl_kernel_submit(global_range, work_group_size, queue, kfn);
}

void upsample_nearest2d_kernel(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    bool is_exact) {
  TensorArg input_arg{input_, "input_", 1};
  TensorArg output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});
  if (input_.numel() == 0) {
    return;
  }

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input_.size(0);
  int channels = input_.size(1);
  int input_height = input_.size(2);
  int input_width = input_.size(3);

  const float height_scale =
      compute_scales_value<float>(scales_h, input_height, output_height);
  const float width_scale =
      compute_scales_value<float>(scales_w, input_width, output_width);

  const auto memory_format = input_.suggest_memory_format();

  if (input_.sizes() == output.sizes()) {
    output.copy_(input_);
    return;
  }
  // heuristic: only use channels_last path when it's faster than the
  // contiguous path
  if (memory_format == at::MemoryFormat::ChannelsLast && channels >= 4 &&
      output.is_contiguous(memory_format)) {
    at::Tensor input = input_.contiguous(at::MemoryFormat::ChannelsLast);

    TORCH_CHECK(
        input.numel() < std::numeric_limits<int64_t>::max(),
        "upsample_nearest_nhwc only supports input tensors with less than 2^63 - 1 elements");
    TORCH_CHECK(
        output.numel() < std::numeric_limits<int64_t>::max(),
        "upsample_nearest_nhwc only supports output tensors with less than 2^63 - 1 elements");

    AT_DISPATCH_FLOATING_TYPES_AND3(
        ScalarType::BFloat16,
        ScalarType::Half,
        ScalarType::Byte,
        input.scalar_type(),
        "upsample_nearest2d_channels_last_xpu",
        [&] {
          const scalar_t* idata = input.const_data_ptr<scalar_t>();
          scalar_t* odata = output.mutable_data_ptr<scalar_t>();
          if (is_exact) {
            upsample_nearest2d_channels_last_frame<scalar_t>(
                idata,
                odata,
                channels,
                input_height,
                input_width,
                output_height,
                output_width,
                height_scale,
                width_scale,
                output.numel(),
                NearestExactIndexOp());
          } else {
            upsample_nearest2d_channels_last_frame<scalar_t>(
                idata,
                odata,
                channels,
                input_height,
                input_width,
                output_height,
                output_width,
                height_scale,
                width_scale,
                output.numel(),
                NearestIndexOp());
          }
        });
  } else {
    // This is needed for non-contiguous tensors.
    Tensor output_c = output.is_contiguous()
        ? output
        : at::empty(output.sizes(), output.options());
    Tensor input = input_.contiguous();

    int nc = nbatch * channels;

    AT_DISPATCH_FLOATING_TYPES_AND3(
        ScalarType::BFloat16,
        ScalarType::Half,
        ScalarType::Byte,
        input.scalar_type(),
        "upsample_nearest2d_xpu",
        [&] {
          auto idata = input.const_data_ptr<scalar_t>();
          auto odata = output_c.mutable_data_ptr<scalar_t>();
          if (is_exact) {
            upsample_nearest2d_frame<scalar_t>(
                idata,
                odata,
                nc,
                input_height,
                input_width,
                output_height,
                output_width,
                height_scale,
                width_scale,
                NearestExactIndexOp());
          } else {
            upsample_nearest2d_frame<scalar_t>(
                idata,
                odata,
                nc,
                input_height,
                input_width,
                output_height,
                output_width,
                height_scale,
                width_scale,
                NearestIndexOp());
          }
        });

    if (!output.is_contiguous()) {
      output.copy_(output_c);
    }
  }
}
} // namespace xpu
} // namespace at::native
