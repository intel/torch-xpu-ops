/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/xpu/sycl/SleepKernels.h>
#include <c10/xpu/XPUStream.h>
#include <comm/SYCLHelpers.h>

#if !defined(SYCL_COMPILER_VERSION)
#error \
    "SYCL_COMPILER_VERSION is not defined. Ensure SYCLToolkit is found before building torch-xpu-ops."
#endif

namespace at::native::xpu {

namespace syclexp = sycl::ext::oneapi::experimental;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (syclexp::device_has<sycl::aspect::ext_oneapi_clock_sub_group>))
void clock_impl(uint64_t cycles) {
  uint64_t start = syclexp::clock<syclexp::clock_scope::sub_group>();
  while (syclexp::clock<syclexp::clock_scope::sub_group>() - start < cycles) {
  }
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void loop_impl(uint64_t iters) {
  for (volatile uint64_t i = 0; i < iters; i = i + 1) {
  }
}

void clock_kernel(uint64_t cycles) {
  sycl_kernel_submit<clock_impl>(
      1, 1, c10::xpu::getCurrentXPUStream(), 0, cycles);
}

void loop_kernel(uint64_t iters) {
  sycl_kernel_submit<loop_impl>(
      1, 1, c10::xpu::getCurrentXPUStream(), 0, iters);
}

} // namespace at::native::xpu
