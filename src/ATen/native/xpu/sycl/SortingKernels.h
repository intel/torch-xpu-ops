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

#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/SortingCommon.h>
#include <ATen/native/xpu/sycl/SortingRadixSort.h>
#include <c10/core/Allocator.h>
#include <comm/SYCLContext.h>

namespace at {
namespace native {
namespace xpu {

// ======================= group sort =======================

template <typename method_t, typename key_t, typename value_t>
struct SegmentedGroupRadixSortPairsFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[sycl::reqd_sub_group_size(method_t::SUBGROUP_SIZE)]] void operator()(
      sycl::nd_item<1> item) const {
    int seg_idx = item.get_group(0);
    int seg_offset = seg_idx * num_elements_;
    auto method = method_t(item, slm_);
    method.load_keys(keys_in_ + seg_offset, num_elements_);
    method.load_values(
        values_in_ == nullptr ? nullptr : values_in_ + seg_offset,
        num_elements_);
    int begin_bit = 0;
    int end_bit = KeyTraits<key_t>::endbit();
    while (true) {
      method.rank_keys(begin_bit, end_bit);
      method.exchange_keys();
      method.exchange_values();
      begin_bit += method_t::RADIX_BITS;
      if (begin_bit >= end_bit)
        break;
    }
    method.store_keys(keys_out_ + seg_offset, num_elements_);
    method.store_values(values_out_ + seg_offset, num_elements_);
  }
  void sycl_ker_config_convention(sycl::handler& cgh) {
    slm_ = sycl_local_acc_t<char>(method_t::LocalMemorySize(), cgh);
  }
  SegmentedGroupRadixSortPairsFunctor(
      const key_t* keys_in,
      key_t* keys_out,
      const value_t* values_in,
      value_t* values_out,
      int num_elements)
      : keys_in_(keys_in),
        keys_out_(keys_out),
        values_in_(values_in),
        values_out_(values_out),
        num_elements_(num_elements) {}

 private:
  const key_t* keys_in_;
  key_t* keys_out_;
  const value_t* values_in_;
  value_t* values_out_;
  int num_elements_;
  sycl_local_acc_t<char> slm_;
};

template <
    typename key_t,
    typename value_t,
    bool IS_DESCENDING,
    int KEYS_PER_ITEM,
    int GROUP_SIZE,
    int SUBGROUP_SIZE>
void segmented_group_radix_sort_pairs_kernel(
    const key_t* keys_in,
    key_t* keys_out,
    const value_t* values_in,
    value_t* values_out,
    int num_segments,
    int num_elements) {
  using method_t = GroupRadixSort<
      key_t,
      GROUP_SIZE,
      SUBGROUP_SIZE,
      KEYS_PER_ITEM,
      IS_DESCENDING,
      value_t>;
  auto caller = SegmentedGroupRadixSortPairsFunctor<method_t, key_t, value_t>(
      keys_in, keys_out, values_in, values_out, num_elements);
  sycl_kernel_submit(
      num_segments * GROUP_SIZE,
      GROUP_SIZE,
      at::xpu::getCurrentSYCLQueue(),
      caller);
}

// ======================= upsweep =======================

template <typename method_t, typename key_t, typename value_t>
struct SegmentedRadixSortPairsUpsweepFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[sycl::reqd_sub_group_size(method_t::SUBGROUP_SIZE)]] void operator()(
      sycl::nd_item<1> item) const {
    int num_tiles = (num_elements_ + method_t::PROCESSING_LENGTH - 1) /
        method_t::PROCESSING_LENGTH;
    int seg_idx = item.get_group(0) / num_tiles;
    int tile_idx = item.get_group(0) % num_tiles;
    auto keys_in_seg = keys_in_ + seg_idx * num_elements_;
    auto counts_seg = counts_ + seg_idx * method_t::RADIX_BUCKETS * num_tiles;
    int tile_offset = tile_idx * method_t::PROCESSING_LENGTH;
    int tile_end = tile_offset + method_t::PROCESSING_LENGTH;
    tile_end = tile_end > num_elements_ ? num_elements_ : tile_end;
    auto method = method_t(
        item,
        keys_in_seg,
        tile_idx,
        begin_bit_,
        end_bit_,
        num_tiles,
        counts_seg,
        slm_);
    method.run(tile_offset, tile_end);
  }
  void sycl_ker_config_convention(sycl::handler& cgh) {
    slm_ = sycl_local_acc_t<char>(method_t::LocalMemorySize(), cgh);
  }
  SegmentedRadixSortPairsUpsweepFunctor(
      const key_t* keys_in,
      int* counts,
      int num_elements,
      int begin_bit,
      int end_bit)
      : keys_in_(keys_in),
        counts_(counts),
        num_elements_(num_elements),
        begin_bit_(begin_bit),
        end_bit_(end_bit) {}

 private:
  const key_t* keys_in_;
  int* counts_;
  int num_elements_;
  int begin_bit_;
  int end_bit_;
  sycl_local_acc_t<char> slm_;
};

template <
    typename key_t,
    typename value_t,
    bool IS_DESCENDING,
    int KEYS_PER_ITEM,
    int GROUP_SIZE,
    int SUBGROUP_SIZE>
void segmented_radix_sort_pairs_upsweep_kernel(
    const key_t* keys_in,
    int* counts,
    int num_segments,
    int num_elements,
    int begin_bit,
    int end_bit) {
  using method_t = RadixSortUpsweep<
      key_t,
      GROUP_SIZE,
      SUBGROUP_SIZE,
      KEYS_PER_ITEM,
      IS_DESCENDING,
      value_t>;
  int num_tiles = (num_elements + method_t::PROCESSING_LENGTH - 1) /
      method_t::PROCESSING_LENGTH;
  auto caller = SegmentedRadixSortPairsUpsweepFunctor<method_t, key_t, value_t>(
      keys_in, counts, num_elements, begin_bit, end_bit);
  sycl_kernel_submit(
      num_segments * num_tiles * GROUP_SIZE,
      GROUP_SIZE,
      at::xpu::getCurrentSYCLQueue(),
      caller);
}

// ======================= scan bins =======================

template <typename method_t>
struct SegmentedRadixSortPairsScanFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[sycl::reqd_sub_group_size(method_t::SUBGROUP_SIZE)]] void operator()(
      sycl::nd_item<1> item) const {
    constexpr int RADIX_BUCKETS = 16;
    int seg_idx = item.get_group(0);
    auto counts_seg = counts_ + seg_idx * RADIX_BUCKETS * num_tiles_;
    auto method = method_t(item, counts_seg, slm_);
    method.run(num_tiles_ * RADIX_BUCKETS);
  }
  void sycl_ker_config_convention(sycl::handler& cgh) {
    slm_ = sycl_local_acc_t<char>(method_t::LocalMemorySize(), cgh);
  }
  SegmentedRadixSortPairsScanFunctor(int* counts, int num_tiles)
      : counts_(counts), num_tiles_(num_tiles) {}

 private:
  int* counts_;
  int num_tiles_;
  sycl_local_acc_t<char> slm_;
};

template <int KEYS_PER_ITEM, int GROUP_SIZE, int SUBGROUP_SIZE>
void segmented_radix_sort_pairs_scan_kernel(
    int* counts,
    int num_tiles,
    int num_segments) {
  using method_t = RadixSortScanBins<GROUP_SIZE, KEYS_PER_ITEM, SUBGROUP_SIZE>;
  auto caller = SegmentedRadixSortPairsScanFunctor<method_t>(counts, num_tiles);
  sycl_kernel_submit(
      num_segments * GROUP_SIZE,
      GROUP_SIZE,
      at::xpu::getCurrentSYCLQueue(),
      caller);
}

// ======================= downsweep =======================

template <typename method_t, typename key_t, typename value_t>
struct SegmentedRadixSortPairsDownsweepFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[sycl::reqd_sub_group_size(method_t::SUBGROUP_SIZE)]] void operator()(
      sycl::nd_item<1> item) const {
    int num_tiles = (num_elements_ + method_t::PROCESSING_LENGTH - 1) /
        method_t::PROCESSING_LENGTH;
    int seg_idx = item.get_group(0) / num_tiles;
    int tile_idx = item.get_group(0) % num_tiles;
    int seg_offset = seg_idx * num_elements_;
    int tile_offset = tile_idx * method_t::PROCESSING_LENGTH;
    auto counts_seg = counts_ + seg_idx * method_t::RADIX_BUCKETS * num_tiles;
    auto method = method_t(item, slm_);
    method.load_keys(keys_in_ + seg_offset, num_elements_, tile_offset);
    method.load_values(
        values_in_ == nullptr ? nullptr : values_in_ + seg_offset,
        num_elements_,
        tile_offset);
    method.load_bin_offsets(counts_seg, tile_idx, num_tiles);
    method.rank_keys(begin_bit_, end_bit_);
    method.exchange_and_store_keys(keys_out_ + seg_offset, num_elements_);
    method.exchange_and_store_values(values_out_ + seg_offset, num_elements_);
  }
  void sycl_ker_config_convention(sycl::handler& cgh) {
    slm_ = sycl_local_acc_t<char>(method_t::LocalMemorySize(), cgh);
  }
  SegmentedRadixSortPairsDownsweepFunctor(
      const key_t* keys_in,
      key_t* keys_out,
      const value_t* values_in,
      value_t* values_out,
      int num_elements,
      int begin_bit,
      int end_bit,
      int* counts)
      : keys_in_(keys_in),
        keys_out_(keys_out),
        values_in_(values_in),
        values_out_(values_out),
        num_elements_(num_elements),
        begin_bit_(begin_bit),
        end_bit_(end_bit),
        counts_(counts) {}

 private:
  const key_t* keys_in_;
  key_t* keys_out_;
  const value_t* values_in_;
  value_t* values_out_;
  int num_elements_;
  int begin_bit_;
  int end_bit_;
  int* counts_;
  sycl_local_acc_t<char> slm_;
};

template <
    typename key_t,
    typename value_t,
    bool IS_DESCENDING,
    int KEYS_PER_ITEM,
    int GROUP_SIZE,
    int SUBGROUP_SIZE>
void segmented_radix_sort_pairs_downsweep_kernel(
    const key_t* keys_in,
    key_t* keys_out,
    const value_t* values_in,
    value_t* values_out,
    int num_segments,
    int num_elements,
    int begin_bit,
    int end_bit,
    int* count) {
  using method_t = GroupRadixSort<
      key_t,
      GROUP_SIZE,
      SUBGROUP_SIZE,
      KEYS_PER_ITEM,
      IS_DESCENDING,
      value_t>;
  int num_tiles = (num_elements + method_t::PROCESSING_LENGTH - 1) /
      method_t::PROCESSING_LENGTH;
  auto caller =
      SegmentedRadixSortPairsDownsweepFunctor<method_t, key_t, value_t>(
          keys_in,
          keys_out,
          values_in,
          values_out,
          num_elements,
          begin_bit,
          end_bit,
          count);
  sycl_kernel_submit(
      num_segments * num_tiles * GROUP_SIZE,
      GROUP_SIZE,
      at::xpu::getCurrentSYCLQueue(),
      caller);
}

// ======================= large sort =======================

template <typename scalar_t>
struct ABBufferCopyFunctor {
  scalar_t operator()(scalar_t x) const {
    return x;
  }
};

template <
    typename key_t,
    typename value_t,
    bool IS_DESCENDING,
    int KEYS_PER_ITEM,
    int GROUP_SIZE,
    int SUBGROUP_SIZE>
void segmented_radix_sort_pairs_kernel(
    const key_t* keys_in,
    key_t* keys_out,
    const value_t* values_in,
    value_t* values_out,
    int num_segments,
    int num_elements) {
  constexpr int TILE_PROCESSING_LENGTH = GROUP_SIZE * KEYS_PER_ITEM;
  int num_tiles =
      (num_elements + TILE_PROCESSING_LENGTH - 1) / TILE_PROCESSING_LENGTH;
  constexpr int RADIX_BITS = 4;
  constexpr int RADIX_BUCKETS = 16;
  int begin_bit = 0;
  int end_bit = KeyTraits<key_t>::endbit();
  int* counts;
  key_t* keys_temp;
  value_t* values_temp;

  at::DataPtr counts_data = c10::GetAllocator(kXPU)->allocate(
      num_segments * RADIX_BUCKETS * num_tiles * sizeof(int));
  at::DataPtr keys_temp_data = c10::GetAllocator(kXPU)->allocate(
      num_segments * num_elements * sizeof(key_t));
  at::DataPtr values_temp_data = c10::GetAllocator(kXPU)->allocate(
      num_segments * num_elements * sizeof(value_t));

  counts = (int*)counts_data.get();
  keys_temp = (key_t*)keys_temp_data.get();
  values_temp = (value_t*)values_temp_data.get();

  key_t* keys_in_ = const_cast<key_t*>(keys_in);
  key_t* keys_out_ = keys_temp;
  value_t* values_in_ = const_cast<value_t*>(values_in);
  value_t* values_out_ = values_temp;

  while (true) {
    segmented_radix_sort_pairs_upsweep_kernel<
        key_t,
        value_t,
        IS_DESCENDING,
        KEYS_PER_ITEM,
        GROUP_SIZE,
        SUBGROUP_SIZE>(
        keys_in_, counts, num_segments, num_elements, begin_bit, end_bit);

    segmented_radix_sort_pairs_scan_kernel<
        KEYS_PER_ITEM,
        GROUP_SIZE,
        SUBGROUP_SIZE>(counts, num_tiles, num_segments);

    segmented_radix_sort_pairs_downsweep_kernel<
        key_t,
        value_t,
        IS_DESCENDING,
        KEYS_PER_ITEM,
        GROUP_SIZE,
        SUBGROUP_SIZE>(
        keys_in_,
        keys_out_,
        values_in_,
        values_out_,
        num_segments,
        num_elements,
        begin_bit,
        end_bit,
        counts);

    if (begin_bit == 0) {
      keys_in_ = keys_temp;
      keys_out_ = keys_out;
      values_in_ = values_temp;
      values_out_ = values_out;
    } else {
      std::swap(keys_in_, keys_out_);
      std::swap(values_in_, values_out_);
    }
    begin_bit += RADIX_BITS;
    if (begin_bit >= end_bit)
      break;
  }

  // Among basic types, the bit size of bool is not an even multiple of 4. AB
  // buffer switching is required.
  if constexpr (std::is_same<key_t, bool>::value) {
    auto input_calc = TrivialOffsetCalculator<2>();
    at::detail::Array<char*, 2> data;
    if (keys_out) {
      data[0] = (char*)keys_out;
      data[1] = (char*)keys_temp;
      auto fn = ABBufferCopyFunctor<key_t>();
      auto vec_size = memory::can_vectorize_up_to<decltype(fn)>(data);
      launch_vectorized_kernel(
          num_segments * num_elements, fn, data, input_calc, vec_size);
    }
    if (values_out) {
      data[0] = (char*)values_out;
      data[1] = (char*)values_temp;
      auto fn = ABBufferCopyFunctor<value_t>();
      auto vec_size = memory::can_vectorize_up_to<decltype(fn)>(data);
      launch_vectorized_kernel(
          num_segments * num_elements, fn, data, input_calc, vec_size);
    }
  }
}

// ======================= group radix select =======================

template <int n, typename index_t>
inline index_t make_alignment_n(index_t size) {
  return (size + n - 1) / n * n;
}

template <typename method_t, typename key_t, typename value_t>
struct SegmentedGroupRadixSelectPairsFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  enum {
    MAX_KV_BYTES = std::max(sizeof(key_t), sizeof(value_t)),
  };

  [[sycl::reqd_sub_group_size(method_t::SUBGROUP_SIZE)]] void operator()(
      sycl::nd_item<1> item) const {
    int seg_idx = item.get_group(0);
    int seg_offset = seg_idx * nelements_;
    auto method = method_t(item, slm_);

    auto keys_in_seg = keys_in_ + seg_offset;
    auto values_in_seg =
        values_in_ == nullptr ? nullptr : values_in_ + seg_offset;

    key_t* keys_temp = reinterpret_cast<key_t*>(
        slm_.template get_multi_ptr<sycl::access::decorated::no>().get() +
        make_alignment_n<MAX_KV_BYTES>(method_t::LocalMemorySize()));
    value_t* values_temp = reinterpret_cast<value_t*>(
        reinterpret_cast<char*>(keys_temp) +
        make_alignment_n<MAX_KV_BYTES>(k_ * sizeof(key_t)));

    method.load_keys(keys_in_seg, nelements_);
    method.load_values(values_in_seg, nelements_);

    int num_start = method_t::PROCESSING_LENGTH;
    while (num_start < nelements_) {
      method.topk(KeyTraits<key_t>::endbit(), 0, k_, keys_temp, values_temp);
      sycl::group_barrier(item.get_group());
      method.topk_append_keys(
          keys_in_seg, keys_temp, nelements_, num_start, k_);
      method.topk_append_values(
          values_in_seg, values_temp, nelements_, num_start, k_);
      num_start += method_t::PROCESSING_LENGTH - k_;
      sycl::group_barrier(item.get_group());
    }

    method.topk(
        KeyTraits<key_t>::endbit(),
        0,
        k_,
        keys_out_ + seg_idx * k_,
        values_out_ + seg_idx * k_);
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    slm_ = sycl_local_acc_t<char>(
        make_alignment_n<MAX_KV_BYTES>(method_t::LocalMemorySize()) +
            make_alignment_n<MAX_KV_BYTES>(k_ * sizeof(key_t)) +
            k_ * sizeof(value_t),
        cgh);
  }

  SegmentedGroupRadixSelectPairsFunctor(
      const key_t* keys_in,
      key_t* keys_out,
      const value_t* values_in,
      value_t* values_out,
      int nelements,
      int k)
      : keys_in_(keys_in),
        keys_out_(keys_out),
        values_in_(values_in),
        values_out_(values_out),
        nelements_(nelements),
        k_(k) {}

 private:
  const key_t* keys_in_;
  key_t* keys_out_;
  const value_t* values_in_;
  value_t* values_out_;
  int nelements_;
  int k_;
  sycl_local_acc_t<char> slm_;
};

template <
    typename key_t,
    typename value_t,
    bool IS_DESCENDING,
    int KEYS_PER_ITEM,
    int GROUP_SIZE,
    int SUBGROUP_SIZE>
inline void group_radix_select_pairs_kernel(
    const key_t* keys_in,
    key_t* keys_out,
    const value_t* values_in,
    value_t* values_out,
    int num_segments,
    int num_elements,
    int k) {
  using method_t = GroupRadixSort<
      key_t,
      GROUP_SIZE,
      SUBGROUP_SIZE,
      KEYS_PER_ITEM,
      IS_DESCENDING,
      value_t>;
  TORCH_CHECK(k <= method_t::PROCESSING_LENGTH);
  auto caller = SegmentedGroupRadixSelectPairsFunctor<method_t, key_t, value_t>(
      keys_in, keys_out, values_in, values_out, num_elements, k);
  sycl_kernel_submit(
      num_segments * GROUP_SIZE,
      GROUP_SIZE,
      at::xpu::getCurrentSYCLQueue(),
      caller);
}

// ======================= interface =======================

// NOTE: Subgroup size of 32 provides better performance currently.
// TODO: Additional selection logic is needed to adapt to different platforms.
template <
    typename key_t,
    typename value_t,
    bool IS_DESCENDING,
    int SUBGROUP_SIZE = 32>
void segmented_sort_pairs_(
    const key_t* keys_in,
    key_t* keys_out,
    const value_t* values_in,
    value_t* values_out,
    int num_segments,
    int num_elements) {
  constexpr int scaling_coef = sizeof(key_t) * sizeof(value_t) >= 64
      ? 2
      : 1; // Attempt to reduce register pressure for performance.
  if (num_elements > 4096 / scaling_coef) {
    // Considering register pressure, we use a problem size of 4096 to delineate
    // the boundary between single tile sort and group sort.
    segmented_radix_sort_pairs_kernel<
        key_t,
        value_t,
        IS_DESCENDING,
        4 / scaling_coef,
        512,
        SUBGROUP_SIZE>(
        keys_in, keys_out, values_in, values_out, num_segments, num_elements);
  } else if (num_elements > 2048 / scaling_coef) {
    segmented_group_radix_sort_pairs_kernel<
        key_t,
        value_t,
        IS_DESCENDING,
        4 / scaling_coef,
        1024,
        SUBGROUP_SIZE>(
        keys_in, keys_out, values_in, values_out, num_segments, num_elements);
  } else if (num_elements > 1024 / scaling_coef) {
    segmented_group_radix_sort_pairs_kernel<
        key_t,
        value_t,
        IS_DESCENDING,
        4 / scaling_coef,
        512,
        SUBGROUP_SIZE>(
        keys_in, keys_out, values_in, values_out, num_segments, num_elements);
  } else if (num_elements > 512 / scaling_coef) {
    segmented_group_radix_sort_pairs_kernel<
        key_t,
        value_t,
        IS_DESCENDING,
        4 / scaling_coef,
        256,
        SUBGROUP_SIZE>(
        keys_in, keys_out, values_in, values_out, num_segments, num_elements);
  } else if (num_elements > 256 / scaling_coef) {
    segmented_group_radix_sort_pairs_kernel<
        key_t,
        value_t,
        IS_DESCENDING,
        4 / scaling_coef,
        128,
        SUBGROUP_SIZE>(
        keys_in, keys_out, values_in, values_out, num_segments, num_elements);
  } else {
    segmented_group_radix_sort_pairs_kernel<
        key_t,
        value_t,
        IS_DESCENDING,
        4 / scaling_coef,
        64,
        SUBGROUP_SIZE>(
        keys_in, keys_out, values_in, values_out, num_segments, num_elements);
  }
}

template <typename key_t, typename value_t>
void segmented_sort_pairs(
    const key_t* keys_in,
    key_t* keys_out,
    const value_t* values_in,
    value_t* values_out,
    int num_segments,
    int num_elements,
    bool descending) {
  if (descending)
    segmented_sort_pairs_<key_t, value_t, true>(
        keys_in, keys_out, values_in, values_out, num_segments, num_elements);
  else
    segmented_sort_pairs_<key_t, value_t, false>(
        keys_in, keys_out, values_in, values_out, num_segments, num_elements);
}

template <typename key_t, typename value_t>
void sort_pairs(
    const key_t* keys_in,
    key_t* keys_out,
    const value_t* values_in,
    value_t* values_out,
    int num_elements,
    bool descending) {
  segmented_sort_pairs<key_t, value_t>(
      keys_in, keys_out, values_in, values_out, 1, num_elements, descending);
}

inline uint64_t radix_select_last_power2(uint64_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

template <
    typename key_t,
    typename value_t,
    bool IS_DESCENDING,
    int SUBGROUP_SIZE = 32>
void segmented_group_select_pairs_(
    const key_t* keys_in,
    key_t* keys_out,
    const value_t* values_in,
    value_t* values_out,
    int num_segments,
    int num_elements,
    int k) {
#define RUN_RADIX_SELECT(PADDED_N)   \
  {                                  \
    group_radix_select_pairs_kernel< \
        key_t,                       \
        value_t,                     \
        IS_DESCENDING,               \
        4,                           \
        PADDED_N / 4,                \
        SUBGROUP_SIZE>(              \
        keys_in,                     \
        keys_out,                    \
        values_in,                   \
        values_out,                  \
        num_segments,                \
        num_elements,                \
        k);                          \
  }
  constexpr int max_group_size = 1024; // simd32-specific
  if (num_elements <= max_group_size * 4) {
    switch (radix_select_last_power2(num_elements)) {
      case 4096:
        RUN_RADIX_SELECT(4096); // gsz 1024
        break;
      case 2048:
        RUN_RADIX_SELECT(2048); // gsz 512
        break;
      case 1024:
        RUN_RADIX_SELECT(1024); // gsz 256
        break;
      case 512:
        RUN_RADIX_SELECT(512); // gsz 128
        break;
      default:
        RUN_RADIX_SELECT(256); // gsz 64
        break;
    }
  } else {
    switch (max_group_size) {
      case 1024:
        RUN_RADIX_SELECT(4096);
        break;
      case 512:
        RUN_RADIX_SELECT(2048);
        break;
      default:
        RUN_RADIX_SELECT(1024);
        break;
    }
  }
#undef RUN_RADIX_SELECT
}

template <typename key_t, typename value_t>
void segmented_group_select_pairs(
    const key_t* keys_in,
    key_t* keys_out,
    const value_t* values_in,
    value_t* values_out,
    int num_segments,
    int num_elements,
    int k,
    bool largest) {
  if (largest)
    segmented_group_select_pairs_<key_t, value_t, true>(
        keys_in,
        keys_out,
        values_in,
        values_out,
        num_segments,
        num_elements,
        k);
  else
    segmented_group_select_pairs_<key_t, value_t, false>(
        keys_in,
        keys_out,
        values_in,
        values_out,
        num_segments,
        num_elements,
        k);
}

} // namespace xpu
} // namespace native
} // namespace at
