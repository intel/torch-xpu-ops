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
#include <c10/macros/Export.h>
#include <cstdint>

namespace at::native::xpu {

// Spins for `cycles` device clock cycles. Requires the device to support
// ext_oneapi_clock_sub_group.
TORCH_XPU_API void clock_kernel(uint64_t cycles);

// Spins for `iters` loop iterations. Fallback for devices without
// ext_oneapi_clock_sub_group.
TORCH_XPU_API void loop_kernel(uint64_t iters);

} // namespace at::native::xpu
