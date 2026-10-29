/*
 * Copyright 2020-2026 Intel Corporation
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once
#include <ATen/native/xpu/sycl/UpSampleNearest2dBmg.h>

namespace at::native::xpu::bmg_nearest {
template <int S>
struct NearestBackwardVector {
  const uint32_t* x;
  uint32_t* y;
  int count;
  Divider dc, dw;
  void operator()(sycl::nd_item<1> it) const {
    for (int i = it.get_global_linear_id(); i < count;
         i += it.get_global_range(0)) {
      uint32_t p = dc.div(i), c = i - p * dc.d, nh = dw.div(p),
               w = p - nh * dw.d;
      uint32_t base = (nh * S * dw.d + w) * S * dc.d + c;
      sycl::float2 acc(0.f);
#pragma unroll
      for (int h = 0; h < S; ++h) {
#pragma unroll
        for (int w = 0; w < S; ++w) {
          auto v = sycl::vec<uint32_t, 1>(x[base + (h * dw.d * S + w) * dc.d])
                       .template as<sycl::half2>();
          acc += v.template convert<float>();
        }
      }
      y[i] = acc.template convert<sycl::half, sycl::rounding_mode::rte>()
                 .template as<sycl::vec<uint32_t, 1>>()[0];
    }
  }
};
template <int S>
void backward_launch(const at::Tensor& x, const at::Tensor& y) {
  int count = y.numel() / 2;
  int groups = std::min((count + 255) / 256, 1280);
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>(groups * 256, 256),
      NearestBackwardVector<S>{
          (const uint32_t*)x.data_ptr(),
          (uint32_t*)y.data_ptr(),
          count,
          Divider(y.size(1) / 2),
          Divider(y.size(3))});
}

inline bool try_backward(
    const Tensor& input,
    const Tensor& output,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    bool is_exact) {
  if (is_exact || input.dim() != 4 || output.dim() != 4 ||
      input.scalar_type() != kHalf || output.scalar_type() != kHalf ||
      !input.is_contiguous(MemoryFormat::ChannelsLast) ||
      !output.is_contiguous(MemoryFormat::ChannelsLast) || input.size(1) < 16 ||
      input.size(1) % 2 || input.numel() == 0 || input.numel() >= INT_MAX ||
      output.numel() == 0 || input.size(0) != output.size(0) ||
      input.size(1) != output.size(1) || input.size(2) % output.size(2) ||
      input.size(3) % output.size(3) ||
      reinterpret_cast<uintptr_t>(input.data_ptr()) % 4 ||
      reinterpret_cast<uintptr_t>(output.data_ptr()) % 4) {
    return false;
  }
  const int sh = input.size(2) / output.size(2),
            sw = input.size(3) / output.size(3);
  if (sh != sw || (sh != 2 && sh != 4 && sh != 8) ||
      (scales_h && *scales_h != sh) || (scales_w && *scales_w != sw)) {
    return false;
  }
  const auto arch =
      c10::xpu::getCurrentXPUStream()
          .queue()
          .get_device()
          .get_info<
              sycl::ext::oneapi::experimental::info::device::architecture>();
  using Architecture = sycl::ext::oneapi::experimental::architecture;
  if (arch != Architecture::intel_gpu_bmg_g21 &&
      arch != Architecture::intel_gpu_bmg_g31)
    return false;
  if (sh == 2)
    backward_launch<2>(input, output);
  else if (sh == 4)
    backward_launch<4>(input, output);
  else
    backward_launch<8>(input, output);
  return true;
}
} // namespace at::native::xpu::bmg_nearest
