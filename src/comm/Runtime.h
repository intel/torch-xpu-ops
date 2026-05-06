/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <comm/Macros.h>
DISABLE_SYCL_DEPRECATED_WARNING_BEGIN
// Official suppression macro provided by Intel SYCL headers for
// host-only compilation (without -fsycl).
#define SYCL_DISABLE_FSYCL_SYCLHPP_WARNING
#include <c10/xpu/XPUStream.h>
#undef SYCL_DISABLE_FSYCL_SYCLHPP_WARNING
DISABLE_SYCL_DEPRECATED_WARNING_END

namespace at::xpu {

static inline sycl::queue& getCurrentSYCLQueue() {
  return c10::xpu::getCurrentXPUStream().queue();
}

} // namespace at::xpu
