/*
 * Copyright 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions of this file are derived from FBGEMM
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <ATen/ATen.h>
#include <sycl/sycl.hpp>
#include <ATen/xpu/PhiloxXpuState.h>

#include <ATen/native/xpu/sycl/fbgemm_utils/utils.h>
#include <ATen/native/xpu/sycl/fbgemm_utils/vec4.h>
#include <ATen/native/xpu/sycl/fbgemm_utils/stochastic_rounding.h>

namespace fbgemm_utils {

namespace utils {

template <typename T, typename... Ts>
constexpr inline bool is_one_of_v = (std::is_same_v<T, Ts> || ...);

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
constexpr inline T pad4(T value) {
  // Compute the first multiple of 4 that is greater than or equal to the given
  // value
  //
  // First convert value to unsigned type before doing bitwise math, to avoid
  // undefined behavior.  Move x just past the next multiple of 4, then round
  // down to the nearest multiple of 4 by clearing the 2 least significant bits.
  //
  // Example:
  //   pad4(3) = 4
  //   pad4(4) = 4
  //   pad4(5) = 8
  //   pad4(-5) = -4
  using U = std::make_unsigned_t<T>;
  return static_cast<T>((static_cast<U>(value) + U{3}) & ~U{3});
}

} // namespace utils

template <typename dst_t, typename src_t>
Vec4T<dst_t> dequantize_load(
    const src_t* value) {
    
    return Vec4T<dst_t>(value);
}

template <typename emb_t, typename cache_t, typename reg_t>
// TODO: pass in dimension info and calculate qparams for rowwise integer
// quantization
class WeightRow {
 public:
  // Constructor for no stochastic rounding
  WeightRow(emb_t* const row, cache_t* const cache_row, const uint32_t dim)
      : row_(row),
        cache_row_(cache_row),
        dim_(dim),
        stoc_rounding_state_ptr_(nullptr) {}

  // Constructor for stochastic rounding
  WeightRow(
      emb_t* const row,
      cache_t* const cache_row,
      const uint32_t dim,
      const bool stochastic_rounding,
      const at::PhiloxXpuState* stochastic_rounding_philox_args,
      const uint64_t salt_value)
      : row_(row), cache_row_(cache_row), dim_(dim) {
    stoc_rounding_state_ptr_ = nullptr;
    if constexpr (!std::is_same_v<emb_t, float>) {
      if (stochastic_rounding) {
        stoc_rounding_state_.init(*stochastic_rounding_philox_args, salt_value);
        // Store the pointer here to avoid an if-else cond during load/store
        stoc_rounding_state_ptr_ = &stoc_rounding_state_;
      }
    }
  }

//   //////////////////////////////////////////////////////////////////////////////
//   // Load 4 elements from the table row at element offset d into a register
//   // variable (Vec4T<dst_t>)
//   //
//   // If the cache row pointer is valid, then data will be read from the cache
//   // instead of embedding table.
//   //////////////////////////////////////////////////////////////////////////////

  Vec4T<reg_t> load(const int32_t d) const {
    // Load from the cache if resident; else load from the embedding table.
    //
    // Note: This method assumes that reg_t is of higher precision than cache_t
    // and emb_t
    if (cache_row_) {
      return dequantize_load<reg_t, cache_t>(cache_row_ + d);
    } else {
      return dequantize_load<reg_t, emb_t>(row_ + d);
    }
  }

//   //////////////////////////////////////////////////////////////////////////////
//   // Store regster variable of 4 elements (Vec4T<reg_t>) back into the table
//   // into the table row at element offset d
//   //
//   // If the cache row pointer is valid, then data will be written to the cache
//   // instead of embedding table.
//   //////////////////////////////////////////////////////////////////////////////

  template <typename dst_t>
  void quantize_store(
      dst_t* output,
      const Vec4T<reg_t>& value) {
    if (stoc_rounding_state_ptr_) {
      stochastic_rounding_vector<dst_t, reg_t>(
          output, value, *stoc_rounding_state_ptr_);
    } else {
      nearest_rounding_vector<dst_t, reg_t>(output, value);
    }
  }

  void
  store(const Vec4T<reg_t>& v, const int32_t d) {
    // Write back weight (high precision) to cache if resident; else write to
    // embedding table.
    //
    // Note: This method assumes that reg_t is of higher precision than cache_t
    // and emb_t
    if (cache_row_) {
      quantize_store(cache_row_ + d, v);
    } else {
      quantize_store(row_ + d, v);
    }
  }

//   //////////////////////////////////////////////////////////////////////////////
//   // Return a raw pointer to the optimizer state for the row.
//   //
//   // This computes the address at where the optimizer state is stored along the
//   // embedding table row, and returns a reinterpret-casted pointer.  It takes
//   // into account 4-element alignment.
//   //////////////////////////////////////////////////////////////////////////////

  template <typename T>
  T* optimizer_state_ptr() const {
    static_assert(
        std::is_same_v<
            T,
            std::remove_cv_t<
                std::remove_pointer_t<std::remove_reference_t<T>>>>,
        "T must be a pure type (no pointers, references, or cv-qualifiers)");

    auto d_emb_ = dim_;

    // Compute the offset along the row where the optimizer data is stored.
    // Since elements are fetched in groups of 4, the offset should be at the
    // first multiple of 4 that is greater than or equal to D
    const auto d_opt_ = utils::pad4(d_emb_);

    // Return the address at the first position
    //
    // Note: In TBE SSD, we should only ever be using cache_row_, however, the
    // WeightRow class has been overloaded to be used for both UVM and SSD
    // contexts.
    //
    // TODO: Move TBE SSD to use WeightRowAccessor instead in the future.
    if (cache_row_) {
      return reinterpret_cast<T*>(cache_row_ + d_opt_);
    } else {
      return reinterpret_cast<T*>(row_ + d_opt_);
    }
  }

 private:
  // The pointer to the row of weights in the embedding table
  emb_t* const row_;

  // The pointer to the row of weights in the cache
  cache_t* const cache_row_;

  // The number of elements per table row
  const uint32_t dim_;

  // The state for stochastic rounding
  StochasticRoundingRNGState stoc_rounding_state_;
  StochasticRoundingRNGState* stoc_rounding_state_ptr_;

};


template <typename row_t, typename reg_t>
class WeightRowAccessor {
  // The pointer to the row of weights in the table
  const row_t* const row_;

  // The number of elements per table row.
  //
  // This is NOT necessarily equivalent to the row stride D_emb, as there may
  // be quantization parameters and optimizer states packed into the back of
  // the row.
  //
  // dim_ is presumed to be a multiple of 4, since it loads data into Vec4T
  // for max register occupancy.
  const uint32_t dim_;

 public:
  WeightRowAccessor(const row_t* const row, const uint32_t dim)
      : row_(row), dim_(dim) {}

  Vec4T<reg_t> load(const int32_t d) const {
    return dequantize_load<reg_t, row_t>(row_ + d);
  }

};
} // namespace fbgemm_utils
