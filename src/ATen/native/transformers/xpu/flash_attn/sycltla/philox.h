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

#include <ATen/native/xpu/sycl/Philox4x32.h>
#include <cstdint>

namespace FLASH_NAMESPACE {

inline at::native::xpu::uint4 philox(
    uint64_t seed,
    uint64_t subsequence,
    uint64_t offset) {
  at::native::xpu::uint2 key{
      static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
  at::native::xpu::uint4 counter{
      static_cast<uint32_t>(offset),
      static_cast<uint32_t>(offset >> 32),
      static_cast<uint32_t>(subsequence),
      static_cast<uint32_t>(subsequence >> 32)};
  auto result = at::native::xpu::philox4x32_rounds(counter, key, 7);
  return {result.x, result.y, result.z, result.w};
}

} // namespace FLASH_NAMESPACE
