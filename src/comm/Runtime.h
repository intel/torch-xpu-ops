/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once
// Suppress deprecation warnings from oneAPI SYCL headers.
// These are not from our code and would otherwise fail the build under -Werror.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#define SYCL_DISABLE_FSYCL_SYCLHPP_WARNING
#include <c10/xpu/XPUStream.h>
#define SYCL_DISABLE_FSYCL_SYCLHPP_WARNING
#pragma GCC diagnostic pop
namespace at::xpu {

static inline sycl::queue& getCurrentSYCLQueue() {
  return c10::xpu::getCurrentXPUStream().queue();
}

} // namespace at::xpu
