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

#include <ATen/AccumulateType.h>
#include <ATen/native/Pool.h>
#include <ATen/native/utils/ParamUtils.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/BatchKernel.h>
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <comm/Runtime.h>
#include <comm/SYCLHelpers.h>

#include <ATen/native/quantized/sycl/QuantizedMaxPool2d.h>
namespace at::native::xpu {

namespace {
void check_maxpool2d_params(
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "Expected 1d or 2d kernel size, got ",
      kernel_size.size());
  TORCH_CHECK(
      stride.empty() || stride.size() == 2,
      "Expected no strides or 2d strides, got",
      stride.size());
  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "Expected 1d or 2d padding, got ",
      padding.size());
  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "Expected 1d or 2d dilation, got ",
      dilation.size());
}
} // anonymous namespace

template <typename scalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void quantized_max_pool2d_kernel_impl(
    scalar_t* output,
    const scalar_t* input,
    int64_t iC,
    int64_t iH,
    int64_t iW,
    int64_t oH,
    int64_t oW,
    int64_t kH,
    int64_t kW,
    int64_t sH,
    int64_t sW,
    int64_t pH,
    int64_t pW,
    int64_t dH,
    int64_t dW,
    int64_t stride,
    BatchKernelConfig cfg) {
  auto item = syclext::this_work_item::get_nd_item<2>();
  auto desc = cfg.get_item_desc(item);

  do {
    if (desc.glb_problem < cfg.problem_) {
      int idx = desc.glb_problem;
      int64_t b{0}, row{0}, col{0};
      b = idx / stride;
      col = idx % oW;
      row = idx / oW % oH;

      int64_t output_base_offset = (b * oW * oH + row * oW + col) * iC;

      // Get the boundary.
      int64_t h_start = row * sH - pH;
      int64_t w_start = col * sW - pW;
      int64_t h_end = std::min(h_start + (kH - 1) * dH + 1, iH);
      int64_t w_end = std::min(w_start + (kW - 1) * dW + 1, iW);
      while (h_start < 0)
        h_start += dH;
      while (w_start < 0)
        w_start += dW;

      // Stock pytorch's cpu implementation use vectorized instructions
      // through channels such as AVX-512. We use for-loop directly.
      int64_t w, h, c;
#pragma unroll
      for (c = 0; c < iC; c++) {
        scalar_t maxVal = at::numeric_limits<scalar_t>::lower_bound();
#pragma unroll
        for (h = h_start; h < h_end; h += dH) {
#pragma unroll
          for (w = w_start; w < w_end; w += dW) {
            int64_t input_base_offset = (b * iW * iH + h * iW + w) * iC;
            scalar_t val = input[input_base_offset + c];
            if ((static_cast<scalar_t>(val) > maxVal) || at::_isnan(val)) {
              maxVal = static_cast<scalar_t>(val);
            }
          }
        }
        output[output_base_offset + c] = static_cast<scalar_t>(maxVal);
      }
    }
  } while (cfg.next(item, desc));
}

template <typename scalar_t>
void launch_quantized_max_pool2d_kernel(
    scalar_t* output,
    const scalar_t* input,
    int64_t nBatch,
    int64_t iC,
    int64_t iH,
    int64_t iW,
    int64_t oH,
    int64_t oW,
    int64_t kH,
    int64_t kW,
    int64_t sH,
    int64_t sW,
    int64_t pH,
    int64_t pW,
    int64_t dH,
    int64_t dW) {
  auto& queue = at::xpu::getCurrentSYCLQueue();
  int outputSize = nBatch * oH * oW;
  int stride = oH * oW;
  BatchKernelConfig cfg = BatchKernelConfig::make_config<
      quantized_max_pool2d_kernel_impl<scalar_t>>(
      1, outputSize, 1, 1, true, {BatchKernelConfig::Policy::pAdaptive});
  sycl_kernel_submit<quantized_max_pool2d_kernel_impl<scalar_t>>(
      cfg.global_size(),
      cfg.group_size(),
      queue,
      0,
      output,
      input,
      iC,
      iH,
      iW,
      oH,
      oW,
      kH,
      kW,
      sH,
      sW,
      pH,
      pW,
      dH,
      dW,
      stride,
      cfg);
}

Tensor quantized_max_pool2d_kernel(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  check_maxpool2d_params(kernel_size, stride, padding, dilation);
  if (stride.empty()) {
    stride = kernel_size;
  }
  Tensor output;
  int ndim = input.dim();
  int64_t kH = kernel_size[0];
  int64_t kW = kernel_size[1];
  int64_t sH = stride[0];
  int64_t sW = stride[1];
  int64_t pH = padding[0];
  int64_t pW = padding[1];
  int64_t dH = dilation[0];
  int64_t dW = dilation[1];

  // Check input dimensions.
  TORCH_CHECK(kH > 0 && kW > 0, "kernel_size should be greater than zero.");
  TORCH_CHECK(sH > 0 && sW > 0, "strides should be greater than zero.");
  TORCH_CHECK(
      dH > 0 && dW > 0,
      "dilation should be greater than zero. "
      "Got (",
      dH,
      ", ",
      dW,
      ")");
  TORCH_CHECK(
      ndim == 3 || ndim == 4, "Expecting the input tensor of rank 3 or 4.");

  int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  int64_t iC = input.size(-3);
  int64_t iH = input.size(-2);
  int64_t iW = input.size(-1);
  TORCH_CHECK(iC > 0 && iH > 0 && iW > 0, "input dimensions must be non-zero.");
  TORCH_CHECK(
      kH / 2 >= pH && kW / 2 >= pW,
      "padding should be smaller than half of kernel_size.");
  int64_t oH = pooling_output_shape(iH, kH, pH, sH, dH, ceil_mode);
  int64_t oW = pooling_output_shape(iW, kW, pW, sW, dW, ceil_mode);
  int64_t oC = iC;

  TORCH_CHECK(
      oH > 0 && oW > 0,
      "Given input size: (",
      iC,
      "x",
      iH,
      "x",
      iW,
      "). Calculated output size: (",
      oC,
      "x",
      oH,
      "x",
      oW,
      "). Output size is too small.");

  std::vector<int64_t> oSizes;
  if (ndim == 3) {
    oSizes = {oC, oH, oW};
  } else {
    oSizes = {nbatch, oC, oH, oW};
  }

  // Create an input
  output = at::empty(
      oSizes,
      input.options()
          .device(c10::kXPU)
          .dtype(input.scalar_type())
          .memory_format(c10::MemoryFormat::ChannelsLast));

  if (input.is_contiguous(c10::MemoryFormat::ChannelsLast)) {
    AT_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "quantized_max_pool2d_xpu", [&]() {
          launch_quantized_max_pool2d_kernel(
              output.data_ptr<scalar_t>(),
              input.const_data_ptr<scalar_t>(),
              nbatch,
              iC,
              iH,
              iW,
              oH,
              oW,
              kH,
              kW,
              sH,
              sW,
              pH,
              pW,
              dH,
              dW);
        });
  } else {
    // If input is uint8 and contiguous memory format,
    // Use the channels_last implementation and convert output back to
    // contiguous.
    auto input_nhwc = input.contiguous(c10::MemoryFormat::ChannelsLast);
    AT_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "quantized_max_pool2d_xpu", [&]() {
          launch_quantized_max_pool2d_kernel(
              output.data_ptr<scalar_t>(),
              input_nhwc.data_ptr<scalar_t>(),
              nbatch,
              iC,
              iH,
              iW,
              oH,
              oW,
              kH,
              kW,
              sH,
              sW,
              pH,
              pW,
              dH,
              dW);
        });
    output = output.contiguous();
  }
  return output;
}

} // namespace at::native::xpu

// clang-format off
DISABLE_RETURN_TYPE_WARNING_END
// clang-format on
