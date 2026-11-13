/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/xpu/Sleep.h>
#include <c10/xpu/XPUStream.h>
#include <comm/SYCLContext.h>
#include <sycl/ext/oneapi/experimental/clock.hpp>

#if !defined(SYCL_COMPILER_VERSION)
#error \
    "SYCL_COMPILER_VERSION is not defined. Ensure SYCLToolkit is found before building torch-xpu-ops."
#endif

namespace at::xpu {
namespace {

struct SpinKernel {
  uint64_t cycles_;
  SpinKernel(uint64_t cycles) : cycles_(cycles) {}
  void operator()() const {
    namespace syclex = sycl::ext::oneapi::experimental;
    uint64_t start = syclex::clock<syclex::clock_scope::sub_group>();
    while (syclex::clock<syclex::clock_scope::sub_group>() - start < cycles_) {
    }
  }
};

} // namespace

void sleep(uint64_t cycles) {
#if SYCL_COMPILER_VERSION >= 20260000
  // PVC ships an IGC too old to expose ext_oneapi_clock_sub_group, so sleep
  // stays unsupported there. On devices that expose the extension the spin
  // kernel is a regular SYCL functor compiled ahead of time with the rest of
  // the library, so the first call pays only launch cost.
  TORCH_CHECK_NOT_IMPLEMENTED(
      c10::xpu::get_raw_device(c10::xpu::current_device())
          .has(sycl::aspect::ext_oneapi_clock_sub_group),
      "Requires the ext_oneapi_clock_sub_group extension, "
      "which is not supported on this device. ",
      "Please upgrade to a newer driver.");
  auto& queue = getCurrentSYCLQueue();
  queue.single_task(SpinKernel{cycles});
#else
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "sleep is not supported for the current SYCL compiler version. ",
      "Please upgrade to SYCL compiler version 2026.0 or newer.");
#endif
}

} // namespace at::xpu
