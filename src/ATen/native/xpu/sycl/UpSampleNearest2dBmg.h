/*
 * Copyright 2020-2026 Intel Corporation
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once
#include <ATen/core/Tensor.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <climits>

namespace at::native::xpu::bmg_nearest {
struct Divider {
  uint32_t d, m;
  Divider(uint32_t value)
      : d(value), m(value == 1 ? 0 : uint32_t((uint64_t(1) << 32) / value)) {}
  uint32_t div(uint32_t n) const {
    if (d == 1)
      return n;
    uint32_t q = sycl::mul_hi(n, m);
    return q + uint32_t(n - q * d >= d);
  }
};

template <int SH, int SW>
struct NearestVector {
  const uint32_t* in;
  uint32_t* out;
  int h, w, oh, ow, count;
  Divider dc, dw, dh;
  uint32_t source(uint32_t i) const {
    uint32_t pixel = dc.div(i), c = i - pixel * dc.d;
    uint32_t ny = dw.div(pixel), x = pixel - ny * dw.d;
    uint32_t n = dh.div(ny), y = ny - n * dh.d;
    return ((n * h + y / SH) * w + x / SW) * dc.d + c;
  }
  void operator()(sycl::nd_item<1> it) const {
    int step = it.get_global_range(0) * 4;
    for (int i = it.get_global_linear_id() * 4; i < count; i += step) {
      sycl::vec<uint32_t, 4> values;
      uint32_t pixel = dc.div(i), channel = i - pixel * dc.d;
      uint32_t ny = dw.div(pixel), x = pixel - ny * dw.d;
      uint32_t n = dh.div(ny), y = ny - n * dh.d;
      uint32_t s = ((n * h + y / SH) * w + x / SW) * dc.d + channel;
      int delta = x == ow - 1 ? ((y % SH == SH - 1) ? 0 : -w * int(dc.d))
                              : ((x % SW == SW - 1) ? 0 : -int(dc.d));
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int index = s + j + (channel + j >= dc.d ? delta : 0);
        values[j] = i + j < count ? in[index] : 0;
      }
      if (i + 4 <= count)
        *reinterpret_cast<sycl::vec<uint32_t, 4>*>(out + i) = values;
      else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
          if (i + j < count)
            out[i + j] = values[j];
      }
    }
  }
};

template <int SH, int SW>
void vector_launch(const at::Tensor& x, const at::Tensor& y) {
  int count = y.numel() / 2, c = x.size(1) / 2;
  int groups = std::min((count + 1023) / 1024, 1280);
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>(groups * 256, 256),
      NearestVector<SH, SW>{
          reinterpret_cast<const uint32_t*>(x.data_ptr()),
          reinterpret_cast<uint32_t*>(y.data_ptr()),
          int(x.size(2)),
          int(x.size(3)),
          int(y.size(2)),
          int(y.size(3)),
          count,
          Divider(c),
          Divider(y.size(3)),
          Divider(y.size(2))});
}

inline bool try_forward(
    const Tensor& input,
    const Tensor& output,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    bool is_exact) {
  if (is_exact || input.dim() != 4 || output.dim() != 4 ||
      input.scalar_type() != kHalf || output.scalar_type() != kHalf ||
      !input.is_contiguous(MemoryFormat::ChannelsLast) ||
      !output.is_contiguous(MemoryFormat::ChannelsLast) || input.size(1) < 16 ||
      input.size(1) % 2 || input.numel() == 0 || output.numel() >= INT_MAX ||
      output.numel() == 0 || input.size(0) != output.size(0) ||
      input.size(1) != output.size(1) || output.size(2) % input.size(2) ||
      output.size(3) % input.size(3) ||
      reinterpret_cast<uintptr_t>(input.data_ptr()) % 4 ||
      reinterpret_cast<uintptr_t>(output.data_ptr()) % 16) {
    return false;
  }
  const int sh = output.size(2) / input.size(2),
            sw = output.size(3) / input.size(3);
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
      arch != Architecture::intel_gpu_bmg_g31) {
    return false;
  }
  if (sh == 2)
    vector_launch<2, 2>(input, output);
  else if (sh == 4)
    vector_launch<4, 4>(input, output);
  else
    vector_launch<8, 8>(input, output);
  return true;
}
} // namespace at::native::xpu::bmg_nearest
