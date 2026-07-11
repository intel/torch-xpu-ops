/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// Stateless Philox-4x32 PRNG implementation for XPU.
//
// Unlike PhiloxRNGEngine (Philox4x32.h), this is a pure function: given
// (seed, offset) it returns 4 pseudo-random uint32 values with no mutable
// state. This makes it suitable for use in stateless random APIs.
//
// Ported from CUDA: aten/src/ATen/cuda/StatelessPhilox4x32.cuh
// See PyTorch PRs #177229 and #177230.

#pragma once

#include <cstdint>

namespace at::native::xpu {

struct philox_uint2 {
  uint32_t x, y;
};

struct philox_uint4 {
  uint32_t x, y, z, w;
};

inline philox_uint4 philox_round(philox_uint4 ctr, philox_uint2 key) {
  constexpr uint32_t kPhiloxSA = 0xD2511F53;
  constexpr uint32_t kPhiloxSB = 0xCD9E8D57;

  uint64_t prod0 =
      static_cast<uint64_t>(kPhiloxSA) * static_cast<uint64_t>(ctr.x);
  uint64_t prod1 =
      static_cast<uint64_t>(kPhiloxSB) * static_cast<uint64_t>(ctr.z);

  uint32_t r0_lo = static_cast<uint32_t>(prod0);
  uint32_t r0_hi = static_cast<uint32_t>(prod0 >> 32);
  uint32_t r1_lo = static_cast<uint32_t>(prod1);
  uint32_t r1_hi = static_cast<uint32_t>(prod1 >> 32);

  return {r1_hi ^ ctr.y ^ key.x, r1_lo, r0_hi ^ ctr.w ^ key.y, r0_lo};
}

// Stateless Philox-4x32-10. Returns 4 pseudo-random uint32 values (128 bits)
// determined entirely by (seed, offset). Each unique offset produces a
// distinct 128-bit output.
template <int N_ROUNDS = 10>
inline philox_uint4 philox_4x32(uint64_t seed, uint64_t offset) {
  philox_uint2 key = {
      static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
  philox_uint4 ctr = {
      static_cast<uint32_t>(offset),
      static_cast<uint32_t>(offset >> 32),
      // restrict subsequence=0
      0,
      0};

  constexpr uint32_t kPhilox10A = 0x9E3779B9;
  constexpr uint32_t kPhilox10B = 0xBB67AE85;

#pragma unroll
  for (int i = 0; i < N_ROUNDS - 1; i++) {
    ctr = philox_round(ctr, key);
    key.x += kPhilox10A;
    key.y += kPhilox10B;
  }
  return philox_round(ctr, key);
}

// Derive a new (seed, offset) key from 4 random uint32 values.
inline void philox_derive_key(
    philox_uint4 r,
    uint64_t* out_seed,
    uint64_t* out_offset) {
  *out_seed = static_cast<uint64_t>(r.x) | (static_cast<uint64_t>(r.y) << 32);
  *out_offset = static_cast<uint64_t>(r.z) | (static_cast<uint64_t>(r.w) << 32);
}

} // namespace at::native::xpu
