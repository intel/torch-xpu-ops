/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifdef USE_C10D_XCCL

#include <comm/Macros.h>
DISABLE_SYCL_DEPRECATED_WARNING_BEGIN
// Official suppression macro provided by Intel SYCL headers for
// host-only compilation (without -fsycl).
#define SYCL_DISABLE_FSYCL_SYCLHPP_WARNING
#include <ATen/xpu/XPUEvent.h>
#undef SYCL_DISABLE_FSYCL_SYCLHPP_WARNING
DISABLE_SYCL_DEPRECATED_WARNING_END
#include <torch/csrc/distributed/c10d/FlightRecorderDetail.hpp>
#include <xccl/ProcessGroupXCCL.hpp>

namespace c10d {

template <>
float getDurationFromEvent<at::xpu::XPUEvent>(
    at::xpu::XPUEvent& xcclStartEvent,
    at::xpu::XPUEvent& xcclEndEvent) {
  TORCH_CHECK(
      xcclEndEvent.query(),
      "getDuration can only be called after work is succeeded.")
  return xcclStartEvent.elapsed_time(xcclEndEvent);
}

template struct FlightRecorder<at::xpu::XPUEvent>;
} // namespace c10d
#endif // USE_C10D_XCCL
