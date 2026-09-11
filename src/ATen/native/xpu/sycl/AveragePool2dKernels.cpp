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
#include <ATen/ceil_div.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Pool.h>

#include <ATen/native/xpu/sycl/AveragePool2dKernels.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <comm/Runtime.h>
#include <comm/SYCLContext.h>
#include <comm/SYCLHelpers.h>

namespace at::native {
namespace xpu {

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
    bool use_divisor) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  XPU_KERNEL_LOOP(item, index, total_elements) {
    const int c = index % channels;
    const int pw = (index / channels) % pooled_width;
    const int ph = (index / channels / pooled_width) % pooled_height;
    const int n = index / channels / pooled_width / pooled_height;
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
  const int64_t global_range =
      xpuKernelLoopGroupRange(total_elements, group_size) * group_size;

  sycl_kernel_submit<
      avg_pool2d_channels_last_kernel_impl<scalar_t, accscalar_t, index_t>>(
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
