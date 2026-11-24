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
#include <ATen/xpu/Sleep.h>
#include <comm/Runtime.h>

namespace at::xpu {

void sleep(uint64_t cycles) {
  auto& device = c10::xpu::get_raw_device(c10::xpu::current_device());
  if (device.has(sycl::aspect::ext_oneapi_clock_sub_group)) {
    native::xpu::clock_kernel(cycles);
    return;
  }

  // Fallback: spin for loop iterations instead of clock cycles. Each loop
  // iteration consumes approximately kCyclesPerLoopIter device clock cycles.
  constexpr uint32_t kCyclesPerLoopIter = 600;
  native::xpu::loop_kernel(cycles / kCyclesPerLoopIter);
}

} // namespace at::xpu
