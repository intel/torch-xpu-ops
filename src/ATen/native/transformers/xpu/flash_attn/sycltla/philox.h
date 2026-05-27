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

#include <cstdint>

namespace FLASH_NAMESPACE {

struct uint2 {
  uint32_t x;
  uint32_t y;
};

struct uint4 {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t w;
};

inline uint2 mulhilo32(uint32_t a, uint32_t b) {
  uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
  return {static_cast<uint32_t>(product), static_cast<uint32_t>(product >> 32)};
}

inline uint4 philox_single_round(const uint4 ctr, const uint2 key) {
  constexpr uint32_t kPhiloxSA = 0xD2511F53;
  constexpr uint32_t kPhiloxSB = 0xCD9E8D57;
  uint2 res0 = mulhilo32(kPhiloxSA, ctr.x);
  uint2 res1 = mulhilo32(kPhiloxSB, ctr.z);
  return {res1.y ^ ctr.y ^ key.x, res1.x, res0.y ^ ctr.w ^ key.y, res0.x};
}

inline uint4 philox(uint64_t seed, uint64_t subsequence, uint64_t offset) {
  constexpr uint32_t kPhilox10A = 0x9E3779B9;
  constexpr uint32_t kPhilox10B = 0xBB67AE85;
  uint2 key;
  key.x = static_cast<uint32_t>(seed);
  key.y = static_cast<uint32_t>(seed >> 32);
  uint4 counter;
  counter.x = static_cast<uint32_t>(offset);
  counter.y = static_cast<uint32_t>(offset >> 32);
  counter.z = static_cast<uint32_t>(subsequence);
  counter.w = static_cast<uint32_t>(subsequence >> 32);
#pragma unroll
  for (int i = 0; i < 6; i++) {
    counter = philox_single_round(counter, key);
    key.x += kPhilox10A;
    key.y += kPhilox10B;
  }
  return philox_single_round(counter, key);
}

} // namespace FLASH_NAMESPACE
