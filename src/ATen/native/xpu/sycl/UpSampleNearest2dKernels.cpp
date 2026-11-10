/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/AccumulateType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/IntegerDivider.h>
#include <ATen/native/xpu/sycl/LaunchUtils.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <ATen/native/xpu/sycl/UpSampleNearest2dKernels.h>
#include <ATen/xpu/XPUContext.h>
#include <comm/Runtime.h>
#include <comm/SYCLContext.h>
#include <comm/SYCLHelpers.h>

namespace at::native {
namespace xpu {

// Packed accesses require dense channels-last tensors with pairs of channels.
inline bool can_use_upsample_nearest2d_channels_last_packed(
    const Tensor& input,
    const Tensor& output,
    bool is_exact) {
  if (is_exact || input.dim() != 4 || output.dim() != 4 || input.numel() == 0 ||
      output.numel() == 0 || input.numel() >= INT_MAX ||
      output.numel() >= INT_MAX) {
    return false;
  }
  if ((input.scalar_type() != kHalf && input.scalar_type() != kBFloat16) ||
      output.scalar_type() != input.scalar_type()) {
    return false;
  }
  if (!input.is_contiguous(MemoryFormat::ChannelsLast) ||
      !output.is_contiguous(MemoryFormat::ChannelsLast)) {
    return false;
  }
  return input.size(0) == output.size(0) && input.size(1) == output.size(1) &&
      input.size(1) >= 16 && input.size(1) % 2 == 0;
}

// Explicit scales must agree with the integer replication implied by the sizes.
inline int upsample_nearest2d_packed_scale(
    const Tensor& input,
    const Tensor& output,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  if (output.size(2) % input.size(2) || output.size(3) % input.size(3)) {
    return 0;
  }
  const int height_scale = output.size(2) / input.size(2);
  const int width_scale = output.size(3) / input.size(3);
  if (height_scale != width_scale ||
      (height_scale != 2 && height_scale != 4 && height_scale != 8) ||
      (scales_h && *scales_h != height_scale) ||
      (scales_w && *scales_w != width_scale)) {
    return 0;
  }
  return height_scale;
}

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
  int64_t work_group_size = at::xpu::getDeviceMaxSubGroupSize() *
      at::xpu::getDeviceEUCountPerXeCore();
  int64_t global_range =
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

template <typename scalar_t, int scale>
struct UpsampleNearest2dBackwardChannelsLastPackedKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    for (int index = item.get_global_linear_id(); index < gi_numel_;
         index += item.get_global_range(0)) {
      const uint32_t pixel = channels_divider_.div(index);
      const uint32_t channel = index - pixel * channels_divider_.divisor;
      const uint32_t batch_height = width_divider_.div(pixel);
      const uint32_t width2 = pixel - batch_height * width_divider_.divisor;
      const uint32_t input_index =
          (batch_height * scale * width_divider_.divisor + width2) * scale *
              channels_divider_.divisor +
          channel;
      memory::aligned_vector<float, 2> grad{{0.f, 0.f}};
#pragma unroll
      for (int height_offset = 0; height_offset < scale; ++height_offset) {
#pragma unroll
        for (int width_offset = 0; width_offset < scale; ++width_offset) {
          const auto values =
              sycl::bit_cast<memory::aligned_vector<scalar_t, 2>>(
                  go_[input_index +
                      (height_offset * width_divider_.divisor * scale +
                       width_offset) *
                          channels_divider_.divisor]);
#pragma unroll
          for (int lane = 0; lane < 2; ++lane)
            grad[lane] += static_cast<float>(values[lane]);
        }
      }
      const memory::aligned_vector<scalar_t, 2> values{
          {static_cast<scalar_t>(grad[0]), static_cast<scalar_t>(grad[1])}};
      gi_[index] = sycl::bit_cast<uint32_t>(values);
    }
  }

  UpsampleNearest2dBackwardChannelsLastPackedKernelFunctor(
      const uint32_t* go,
      uint32_t* gi,
      int channels,
      int width2,
      int gi_numel)
      : go_(go),
        gi_(gi),
        gi_numel_(gi_numel),
        channels_divider_(channels),
        width_divider_(width2) {}

 private:
  const uint32_t* go_;
  uint32_t* gi_;
  int gi_numel_;
  at::detail::IntDivider<uint32_t> channels_divider_;
  at::detail::IntDivider<uint32_t> width_divider_;
};

template <typename scalar_t>
void upsample_nearest2d_backward_channels_last_packed_frame(
    const scalar_t* go,
    scalar_t* gi,
    int channels,
    int width2,
    int gi_numel,
    int scale) {
  auto& queue = at::xpu::getCurrentSYCLQueue();
  constexpr int work_group_size = 256;
  const int groups = std::min<int>(
      (gi_numel + work_group_size - 1) / work_group_size,
      at::xpu::getDeviceHWThreads());
  const auto* packed_go = reinterpret_cast<const uint32_t*>(go);
  auto* packed_gi = reinterpret_cast<uint32_t*>(gi);
  auto launch = [&](auto caller) {
    sycl_kernel_submit(
        groups * work_group_size, work_group_size, queue, caller);
  };
  if (scale == 2) {
    launch(
        UpsampleNearest2dBackwardChannelsLastPackedKernelFunctor<scalar_t, 2>(
            packed_go, packed_gi, channels, width2, gi_numel));
  } else if (scale == 4) {
    launch(
        UpsampleNearest2dBackwardChannelsLastPackedKernelFunctor<scalar_t, 4>(
            packed_go, packed_gi, channels, width2, gi_numel));
  } else {
    launch(
        UpsampleNearest2dBackwardChannelsLastPackedKernelFunctor<scalar_t, 8>(
            packed_go, packed_gi, channels, width2, gi_numel));
  }
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
          grad +=
              go_[idx_cl<size_t>(n, ih, iw, c, height1_, width1_, channels_)];
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
  int64_t work_group_size = at::xpu::getDeviceMaxSubGroupSize() *
      at::xpu::getDeviceEUCountPerXeCore();
  int64_t global_range =
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
    std::optional<double> scales_h,
    std::optional<double> scales_w,
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
          if constexpr (
              std::is_same_v<scalar_t, c10::Half> ||
              std::is_same_v<scalar_t, c10::BFloat16>) {
            if (can_use_upsample_nearest2d_channels_last_packed(
                    grad_output_, grad_input, is_exact) &&
                reinterpret_cast<uintptr_t>(go) % sizeof(uint32_t) == 0 &&
                reinterpret_cast<uintptr_t>(gi) % sizeof(uint32_t) == 0) {
              const int scale = upsample_nearest2d_packed_scale(
                  grad_input, grad_output, scales_h, scales_w);
              if (scale != 0) {
                upsample_nearest2d_backward_channels_last_packed_frame<
                    scalar_t>(
                    go,
                    gi,
                    channels / 2,
                    input_width,
                    static_cast<int>(grad_input.numel() / 2),
                    scale);
                return;
              }
            }
          }
          grad_input.zero_();
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
    grad_input.zero_();
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

  int64_t work_group_size = at::xpu::getDeviceMaxSubGroupSize() *
      at::xpu::getDeviceEUCountPerXeCore();
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

template <int scale>
struct UpsampleNearest2dChannelsLastPackedKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    const int stride = item.get_global_range(0) * kPackedValues;
    for (int index = item.get_global_linear_id() * kPackedValues;
         index < out_numel_;
         index += stride) {
      memory::aligned_vector<uint32_t, kPackedValues> values;
      const uint32_t pixel = channels_divider_.div(index);
      const uint32_t channel = index - pixel * channels_divider_.divisor;
      const uint32_t batch_height = width_divider_.div(pixel);
      const uint32_t width2 = pixel - batch_height * width_divider_.divisor;
      const uint32_t batch = height_divider_.div(batch_height);
      const uint32_t height2 = batch_height - batch * height_divider_.divisor;
      const uint32_t input_index =
          ((batch * height1_ + height2 / scale) * width1_ + width2 / scale) *
              channels_divider_.divisor +
          channel;
      // A four-word store can cross one channel boundary. If that next output
      // pixel repeats the same input column (or row), rewind by one input
      // pixel (or row); otherwise consecutive input words are already correct.
      const int delta = width2 == width2_ - 1
          ? ((height2 % scale == scale - 1)
                 ? 0
                 : -width1_ * static_cast<int>(channels_divider_.divisor))
          : ((width2 % scale == scale - 1)
                 ? 0
                 : -static_cast<int>(channels_divider_.divisor));
#pragma unroll
      for (int lane = 0; lane < kPackedValues; ++lane) {
        const int source_index = input_index + lane +
            (channel + lane >= channels_divider_.divisor ? delta : 0);
        values[lane] = index + lane < out_numel_ ? idata_[source_index] : 0;
      }
      if (index + kPackedValues <= out_numel_) {
        *reinterpret_cast<memory::aligned_vector<uint32_t, kPackedValues>*>(
            odata_ + index) = values;
      } else {
#pragma unroll
        for (int lane = 0; lane < kPackedValues; ++lane) {
          if (index + lane < out_numel_)
            odata_[index + lane] = values[lane];
        }
      }
    }
  }

  UpsampleNearest2dChannelsLastPackedKernelFunctor(
      const uint32_t* idata,
      uint32_t* odata,
      int channels,
      int height1,
      int width1,
      int height2,
      int width2,
      int out_numel)
      : idata_(idata),
        odata_(odata),
        height1_(height1),
        width1_(width1),
        width2_(width2),
        out_numel_(out_numel),
        channels_divider_(channels),
        width_divider_(width2),
        height_divider_(height2) {}

 private:
  static constexpr int kPackedValues = 4;
  const uint32_t* idata_;
  uint32_t* odata_;
  int height1_;
  int width1_;
  int width2_;
  int out_numel_;
  at::detail::IntDivider<uint32_t> channels_divider_;
  at::detail::IntDivider<uint32_t> width_divider_;
  at::detail::IntDivider<uint32_t> height_divider_;
};

void upsample_nearest2d_channels_last_packed_frame(
    const uint32_t* idata,
    uint32_t* odata,
    int channels,
    int height1,
    int width1,
    int height2,
    int width2,
    int out_numel,
    int scale) {
  auto& queue = at::xpu::getCurrentSYCLQueue();
  constexpr int work_group_size = 256;
  constexpr int packed_values_per_item = 4;
  const int items =
      (out_numel + packed_values_per_item - 1) / packed_values_per_item;
  const int groups = std::min<int>(
      (items + work_group_size - 1) / work_group_size,
      at::xpu::getDeviceHWThreads());
  auto launch = [&](auto caller) {
    sycl_kernel_submit(
        groups * work_group_size, work_group_size, queue, caller);
  };
  if (scale == 2) {
    launch(UpsampleNearest2dChannelsLastPackedKernelFunctor<2>(
        idata, odata, channels, height1, width1, height2, width2, out_numel));
  } else if (scale == 4) {
    launch(UpsampleNearest2dChannelsLastPackedKernelFunctor<4>(
        idata, odata, channels, height1, width1, height2, width2, out_numel));
  } else {
    launch(UpsampleNearest2dChannelsLastPackedKernelFunctor<8>(
        idata, odata, channels, height1, width1, height2, width2, out_numel));
  }
}

template <typename scalar_t, typename index_op_t>
struct UpsampleNearest2dChannelsLastKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    size_t index = item.get_global_linear_id();
    const size_t stride = item.get_global_range(0);

    for (; index < out_numel_; index += stride) {
      const size_t c = index % channels_;
      const size_t w2 = (index / channels_) % width2_;
      const size_t h2 = (index / channels_ / width2_) % height2_;
      const size_t n = index / channels_ / width2_ / height2_;

      const size_t h1 =
          height1_ == height2_ ? h2 : index_op_(height_scale_, h2, height1_);
      const size_t w1 =
          width1_ == width2_ ? w2 : index_op_(width_scale_, w2, width1_);

      odata_[index] =
          idata_[idx_cl<size_t>(n, h1, w1, c, height1_, width1_, channels_)];
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

  int64_t work_group_size = at::xpu::getDeviceMaxSubGroupSize() *
      at::xpu::getDeviceEUCountPerXeCore();
  int64_t global_range =
      (out_numel + work_group_size - 1) / work_group_size * work_group_size;
  int64_t max_groups = at::xpu::getDeviceMaxWorkItems() / work_group_size;
  max_groups = std::max<int64_t>(1, max_groups);
  global_range = std::min<int64_t>(global_range, max_groups * work_group_size);

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
    std::optional<double> scales_h,
    std::optional<double> scales_w,
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
          if (can_use_upsample_nearest2d_channels_last_packed(
                  input_, output, is_exact) &&
              reinterpret_cast<uintptr_t>(idata) % sizeof(uint32_t) == 0 &&
              reinterpret_cast<uintptr_t>(odata) %
                      alignof(memory::aligned_vector<uint32_t, 4>) ==
                  0) {
            const int scale = upsample_nearest2d_packed_scale(
                input, output, scales_h, scales_w);
            if (scale != 0) {
              upsample_nearest2d_channels_last_packed_frame(
                  reinterpret_cast<const uint32_t*>(idata),
                  reinterpret_cast<uint32_t*>(odata),
                  channels / 2,
                  input_height,
                  input_width,
                  output_height,
                  output_width,
                  static_cast<int>(output.numel() / 2),
                  scale);
              return;
            }
          }
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
