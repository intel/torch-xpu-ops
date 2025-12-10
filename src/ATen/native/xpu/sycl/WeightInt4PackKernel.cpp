/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/WeightInt4PackKernel.h>

namespace at::native::xpu {

struct WeightToInt4PackKernelFunctor {
  void operator()(sycl::item<1> item) const {
    auto idx = item.get_linear_id();
    int K_div_2 = K_ / 2;
    int K_div_8 = K_ / 8;
    int out_y = idx / K_div_8;
    int out_x = idx % K_div_8;
    int in_y = out_y;
    int in_x = out_x * 4;

    weight_packed_[out_y * K_div_8 + out_x] = 0x00000000;
    for (int i = 0; i < 4; i++) {
      uint32_t low = weight_[in_y * K_div_2 + in_x + i] & 0x0000000F;
      uint32_t high = weight_[in_y * K_div_2 + in_x + i] >> 4;
      uint32_t ele_i = (low) | (high << 4);
      weight_packed_[out_y * K_div_8 + out_x] |= ele_i << (i * 8);
    }
  }

  WeightToInt4PackKernelFunctor(
      uint32_t* weight_packed,
      uint8_t* weight,
      int N,
      int K)
      : weight_packed_(weight_packed), weight_(weight), N_(N), K_(K) {}

 private:
  uint32_t* weight_packed_;
  uint8_t* weight_;
  int N_;
  int K_;
};

void weight_to_int4pack_kernel(
    const Tensor& weight_packed,
    const Tensor& weight,
    int N,
    int K) {
  auto weight_packed_data =
      reinterpret_cast<uint32_t*>(weight_packed.data_ptr());
  const auto weight_data = weight.data_ptr<uint8_t>();
  int K_div_8 = K / 8;
  size_t global_range = N * K_div_8;
  auto fn =
      WeightToInt4PackKernelFunctor(weight_packed_data, weight_data, N, K);
  sycl_kernel_submit(sycl::range<1>(global_range), getCurrentSYCLQueue(), fn);
}

} // namespace at::native::xpu
