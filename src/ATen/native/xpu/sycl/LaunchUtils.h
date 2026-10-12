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
#include <algorithm>
#include <bit>

namespace at::native::xpu {
// returns 2**floor(log2(n)); n == 0 clamps to 1 so callers can divide by it
inline int lastPow2(unsigned int n) {
  return std::max<int>(1, std::bit_floor(n));
}
} // namespace at::native::xpu
