/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

/*
 * BSD License
 * 
 * For FBGEMM software
 * 
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 *  * Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 *  * Neither the name Facebook nor the names of its contributors may be used to
 *    endorse or promote products derived from this software without specific
 *    prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <ATen/ATen.h>
#include <sycl/sycl.hpp>

#include <ATen/native/xpu/sycl/fbgemm_utils/utils.h>
#include <ATen/native/xpu/sycl/fbgemm_utils/vec4.h>

using float2 = sycl::float2;
using uint4 = sycl::uint4;
using __half = sycl::half;

namespace fbgemm_utils {

// static constexpr float kQParamEps = 1e-8f;

////////////////////////////////////////////////////////////////////////////////
// Stochastic Rounding RNG State
//
// This is a simple xorshift* RNG with 64 bits of state (vs 384 bits of state
// for curandStatePhilox4_32_10).  It is used for generating uint4 random bits
// for stochastic rounding.
////////////////////////////////////////////////////////////////////////////////

struct StochasticRoundingRNGState {
  uint64_t state = 0;

  constexpr StochasticRoundingRNGState() = default;

  StochasticRoundingRNGState(
      const at::PhiloxXpuState& philox_state,
      const uint64_t salt_value) noexcept {
    init(philox_state, salt_value);
  }

  // From https://github.com/lemire/testingRNG/blob/master/source/splitmix64.h
  constexpr uint64_t splitmix64_stateless(
      uint64_t index) noexcept {
    uint64_t z = (index + UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
  }

  void init(
      const at::PhiloxXpuState& philox_state,
      // The salt value should be different for every *run* and every
      // *thread*.  Passing in threadIdx.x + blockIdx.x * blockDim.x is
      // recommended.
      const uint64_t salt_value) noexcept {
    const auto [s0, s1] = at::xpu::philox::unpack(philox_state);
    state = splitmix64_stateless(s0 ^ s1) ^ splitmix64_stateless(salt_value);

    // Ensure we never have a zero state (insanely low probability, but
    // still...).
    if (state == 0) {
      state = 1;
    }
  }

  // See https://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf and
  // https://en.wikipedia.org/wiki/Xorshift#xorshift*
  constexpr uint4 rand4() noexcept {
    uint4 random_bits = {0, 0, 0, 0};
    uint64_t x = state; /* The state must be seeded with a nonzero value. */
    x ^= x >> 12; // a
    x ^= x << 25; // b
    x ^= x >> 27; // c
    random_bits.x() = (x * UINT64_C(0x2545F4914F6CDD1D)) >> 32;
    x ^= x >> 12; // a
    x ^= x << 25; // b
    x ^= x >> 27; // c
    random_bits.y() = (x * UINT64_C(0x2545F4914F6CDD1D)) >> 32;
    x ^= x >> 12; // a
    x ^= x << 25; // b
    x ^= x >> 27; // c
    random_bits.z() = (x * UINT64_C(0x2545F4914F6CDD1D)) >> 32;
    x ^= x >> 12; // a
    x ^= x << 25; // b
    x ^= x >> 27; // c
    random_bits.w() = (x * UINT64_C(0x2545F4914F6CDD1D)) >> 32;
    // Update internal state
    state = x;
    return random_bits;
  }
};

// ////////////////////////////////////////////////////////////////////////////////
// // Stochastic Rounding Scalar
// ////////////////////////////////////////////////////////////////////////////////

// // Correct for cases where x is not subnormal.
static __half
stochastic_rounding_scalar(float x, uint32_t random_value) {
  uint32_t w_int = sycl::bit_cast<uint32_t>(x);
  unsigned assembles = (w_int & 0xff800000) | (random_value >> 19);
  unsigned subtract = (w_int & 0xff800000);
  float assemble_float = sycl::bit_cast<float>(assembles) - sycl::bit_cast<float>(subtract);
  return sycl::vec<float, 1>(x + assemble_float).convert<__half, sycl::rounding_mode::rtz>()[0];
}


// ////////////////////////////////////////////////////////////////////////////////
// // Stochastic Rounding Vector
// ////////////////////////////////////////////////////////////////////////////////

template <typename dst_t, typename src_t>
inline void stochastic_rounding_vector(
    dst_t* output,
    const Vec4T<src_t>& value,
    StochasticRoundingRNGState& state) {
  value.store(output);
}

template <>
inline void stochastic_rounding_vector(
    at::Half* output,
    const Vec4T<at::Half>& value,
    StochasticRoundingRNGState& state) {

    const auto random_bits = state.rand4();

    sycl::vec<sycl::half, 4> v(
        stochastic_rounding_scalar(value.acc.x(), random_bits.x()),
        stochastic_rounding_scalar(value.acc.y(), random_bits.y()),
        stochastic_rounding_scalar(value.acc.z(), random_bits.z()),
        stochastic_rounding_scalar(value.acc.w(), random_bits.w())
    );
    auto* out_sycl = reinterpret_cast<sycl::half*>(output);
    sycl::multi_ptr<sycl::half, sycl::access::address_space::global_space> mp(out_sycl);
    v.store(0, mp);
}

template <>
inline void stochastic_rounding_vector(
    at::Half* output,
    const Vec4T<float>& value,
    StochasticRoundingRNGState& state) {

    const auto random_bits = state.rand4();

    sycl::vec<sycl::half, 4> v(
          stochastic_rounding_scalar(value.acc.x(), random_bits.x()),
          stochastic_rounding_scalar(value.acc.y(), random_bits.y()),
          stochastic_rounding_scalar(value.acc.z(), random_bits.z()),
          stochastic_rounding_scalar(value.acc.w(), random_bits.w())
      );
      auto* out_sycl = reinterpret_cast<sycl::half*>(output);
      sycl::multi_ptr<sycl::half, sycl::access::address_space::global_space> mp(out_sycl);
      v.store(0, mp);
}

// begin nearest rounding and store implementations
template <typename dst_t, typename src_t>
void nearest_rounding_vector(
    dst_t* output,
    const Vec4T<src_t>& value) {
  value.store(output);
}

} // namespace fbgemm_utils
