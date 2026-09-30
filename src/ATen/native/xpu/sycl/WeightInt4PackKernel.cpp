/*
 * Copyright 2020-2026 Intel Corporation
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

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void weight_to_int4_pack_sub_kernel(
    uint32_t* weight_packed,
    const uint8_t* weight,
    int K,
    int total) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  auto idx = item.get_global_linear_id();
  if (idx >= static_cast<size_t>(total))
    return;

  int K_div_2 = K / 2;
  int K_div_8 = K / 8;
  int out_y = idx / K_div_8;
  int out_x = idx % K_div_8;
  int in_y = out_y;
  int in_x = out_x * 4;

  weight_packed[out_y * K_div_8 + out_x] = 0x00000000;
  for (int i = 0; i < 4; i++) {
    uint32_t low = weight[in_y * K_div_2 + in_x + i] & 0x0000000F;
    uint32_t high = weight[in_y * K_div_2 + in_x + i] >> 4;
    uint32_t ele_i = (low) | (high << 4);
    weight_packed[out_y * K_div_8 + out_x] |= ele_i << (i * 8);
  }
}

void weight_to_int4pack_kernel(
    const Tensor& weight_packed,
    const Tensor& weight,
    int N,
    int K) {
  auto weight_packed_data =
      reinterpret_cast<uint32_t*>(weight_packed.data_ptr());
  const auto weight_data = weight.const_data_ptr<uint8_t>();
  int K_div_8 = K / 8;
  int total = N * K_div_8;
  constexpr auto kptr = weight_to_int4_pack_sub_kernel;
  int64_t local_range = syclMaxWorkGroupSize<kptr>();
  int64_t num_groups = (total + local_range - 1) / local_range;
  int64_t global_range = num_groups * local_range;
  sycl_kernel_submit<kptr>(
      sycl::range<1>(global_range),
      sycl::range<1>(local_range),
      getCurrentSYCLQueue(),
      0,
      weight_packed_data,
      weight_data,
      K,
      total);
}

} // namespace at::native::xpu
