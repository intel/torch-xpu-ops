/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <c10/xpu/XPUStream.h>

namespace at::xpu {

static inline sycl::queue& getCurrentSYCLQueue() {
  return c10::xpu::getCurrentXPUStream().queue();
}

} // namespace at::xpu
