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

#include <sycltla/mha_common.h>
#include <sycltla/philox.h>

namespace FLASH_NAMESPACE {

struct Dropout {
  const uint64_t seed, offset;
  const uint16_t p_dropout_in_uint16_t;

  Dropout(
      const uint64_t seed,
      const uint64_t offset,
      const uint16_t p_dropout_in_uint16_t,
      const int bid,
      const int hid,
      const int tid,
      const int nheads)
      : seed(seed),
        offset(
            offset + (bid * nheads + hid) * kSubgroupSize +
            tid % kSubgroupSize),
        p_dropout_in_uint16_t(p_dropout_in_uint16_t) {}

  template <
      bool encode_dropout_in_sign_bit = false,
      typename Engine,
      typename Layout>
  inline void apply_dropout(
      cute::Tensor<Engine, Layout>& tensor,
      int block_row_start,
      int block_col_start) {
    // tensor should have shape (8, MMA_M, MMA_N)
    static_assert(decltype(rank(tensor.layout()))::value == 3);
    static_assert(decltype(size<0>(tensor.layout()))::value == 8);
    using T = typename Engine::value_type;
    auto encode_dropout = [](bool keep, T val) {
      return keep ? val : (encode_dropout_in_sign_bit ? -val : T(0));
    };
#pragma unroll
    for (int m = 0; m < size<1>(tensor); ++m, block_row_start += 1) {
#pragma unroll
      for (int n = 0; n < size<2>(tensor); ++n) {
        uint64_t rowcol =
            (uint64_t(block_row_start) << 32) | uint64_t(block_col_start + n);
        at::native::xpu::uint4 random_uint4 = FLASH_NAMESPACE::philox(seed, rowcol, offset);
        uint16_t(&rnd_16)[8] = reinterpret_cast<uint16_t(&)[8]>(random_uint4);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          tensor(i, m, n) = encode_dropout(
              rnd_16[i] <= p_dropout_in_uint16_t, tensor(i, m, n));
        }
      }
    }
  }
};

} // namespace FLASH_NAMESPACE
