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

#include <ATen/native/xpu/sycl/SortingCommon.h>

namespace at {
namespace native {
namespace xpu {

template <
    typename KeyT,
    int GROUP_THREADS_,
    int SUBGROUP_SIZE_,
    int KEYS_PER_THREAD_,
    bool IS_DESCENDING_ = false,
    typename ValueT = NullType,
    typename DigitT = uint16_t, // Covering GROUP_THREADS * KEYS_PER_THREAD.
    typename CounterT = uint32_t, // Packed scan datatype
    // We are going to bundle multiple counters with 'DigitT' type to perform
    // packed prefix sum.
    int RADIX_BITS_ = 4>
class GroupRadixSort {
 public:
  using KeyTraitsT = typename KeyTraits<KeyT>::Type;

  enum {
    GROUP_THREADS = GROUP_THREADS_,
    SUBGROUP_SIZE = SUBGROUP_SIZE_,
    KEYS_PER_THREAD = KEYS_PER_THREAD_,
    IS_DESCENDING = IS_DESCENDING_,
    RADIX_BITS = RADIX_BITS_,

    PROCESSING_LENGTH = GROUP_THREADS * KEYS_PER_THREAD,
    RADIX_BUCKETS = 1 << RADIX_BITS,
    KEYS_ONLY = std::is_same<ValueT, NullType>::value,
    PACKING_RATIO = sizeof(CounterT) / sizeof(DigitT),
    COUNTER_LANES = RADIX_BUCKETS / PACKING_RATIO,
    LOG_COUNTER_LANES = Log2<COUNTER_LANES>::VALUE,
    DIGIT_BITS = sizeof(DigitT) << 3,
    DIGIT_MASK = (1 << DIGIT_BITS) - 1,
    IS_INT_TYPE = std::is_integral<ValueT>::value,
  };

  static_assert(sizeof(CounterT) >= sizeof(DigitT), "");
  static_assert(sizeof(CounterT) % sizeof(DigitT) == 0, "");
  static_assert(
      ((1l << (sizeof(DigitT) << 3)) - 1) >= (GROUP_THREADS * KEYS_PER_THREAD),
      " ");

 private:
  union RankT {
    CounterT counters[COUNTER_LANES][GROUP_THREADS];
    CounterT counters_flat[COUNTER_LANES * GROUP_THREADS];
    DigitT buckets[COUNTER_LANES][GROUP_THREADS][PACKING_RATIO];
  };

  union LocalStorage {
    RankT rank_storage;
    struct {
      KeyTraitsT exchange_ukeys[PROCESSING_LENGTH];
      int relative_bin_offsets[RADIX_BUCKETS];
    };
    ValueT exchange_values[PROCESSING_LENGTH];
  };

  sycl::nd_item<1>& item_;
  LocalStorage& local_storage_;
  int lid_;
  int bin_offset_;

  int ranks_[KEYS_PER_THREAD];
  KeyTraitsT ukeys_[KEYS_PER_THREAD];
  ValueT values_[KEYS_PER_THREAD];
  int relative_bin_offsets_[KEYS_PER_THREAD];
  int begin_bit_;
  int pass_bits_;
  bool enable_bin_offsets_ = false;

 public:
  static int LocalMemorySize() {
    return sizeof(LocalStorage);
  }

  inline void load_bin_offsets(int* counts, int group_id, int num_groups) {
    int bin_idx = lid_;
    if (lid_ < RADIX_BUCKETS) {
      if (IS_DESCENDING)
        bin_idx = RADIX_BUCKETS - bin_idx - 1;
      bin_offset_ = counts[group_id + bin_idx * num_groups];
    }
    enable_bin_offsets_ = true;
    item_.barrier(sycl_local_fence);
  }

  inline GroupRadixSort(sycl::nd_item<1>& item, sycl_local_acc_t<char> buffer)
      : item_(item),
        local_storage_(reinterpret_cast<LocalStorage&>(
            *(buffer.template get_multi_ptr<sycl::access::decorated::no>()
                  .get()))),
        lid_(item.get_local_id(0)) {}

  inline void load_keys(
      const KeyT* keys_group_in,
      int num_elements,
      int group_offset = 0) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      int offset = group_offset + lid_ * KEYS_PER_THREAD + ITEM;
      if (offset < num_elements) {
        ukeys_[ITEM] =
            KeyTraits<KeyT>::convert(c10::load(&keys_group_in[offset]));
      } else {
        KeyTraitsT padding_key;
        if (IS_DESCENDING) {
          padding_key = 0;
        } else {
          constexpr uint64_t KEY_TRAITS_TYPE_MASK = 1ll
              << ((sizeof(KeyTraitsT) << 3) - 1);
          padding_key = static_cast<KeyTraitsT>(KEY_TRAITS_TYPE_MASK);
          padding_key = padding_key ^ (padding_key - 1);
        }
        ukeys_[ITEM] = padding_key;
      }
    }
  }

  inline void load_values(
      const ValueT* values_group_in,
      int num_elements,
      int group_offset = 0) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      int offset = group_offset + lid_ * KEYS_PER_THREAD + ITEM;
      if (offset < num_elements) {
        if constexpr (IS_INT_TYPE) {
          values_[ITEM] =
              values_group_in == nullptr ? offset : values_group_in[offset];
        } else {
          values_[ITEM] = values_group_in[offset];
        }
      }
    }
  }

  inline void store_keys(KeyT* keys_group_out, int num_elements) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      local_storage_.exchange_ukeys[lid_ * KEYS_PER_THREAD + ITEM] =
          ukeys_[ITEM];
    }
    item_.barrier(sycl_local_fence);
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      int offset = lid_ + ITEM * GROUP_THREADS;
      if (offset < num_elements) {
        keys_group_out[offset] =
            KeyTraits<KeyT>::deconvert(local_storage_.exchange_ukeys[offset]);
      }
    }
    item_.barrier(sycl_local_fence);
  }

  inline void store_keys(KeyT* out, int offset_select, int num_selected) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      if (ranks_[ITEM] < offset_select) {
        auto key = KeyTraits<KeyT>::deconvert(ukeys_[ITEM]);
        out[num_selected + ranks_[ITEM]] = key;
      }
    }
  }

  inline void store_values(ValueT* values_group_out, int num_elements) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      local_storage_.exchange_values[lid_ * KEYS_PER_THREAD + ITEM] =
          values_[ITEM];
    }
    item_.barrier(sycl_local_fence);
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      int offset = lid_ + ITEM * GROUP_THREADS;
      if (offset < num_elements) {
        values_group_out[offset] = local_storage_.exchange_values[offset];
      }
    }
    item_.barrier(sycl_local_fence);
  }

  inline void store_values(ValueT* out, int offset_select, int num_selected) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      if (ranks_[ITEM] < offset_select) {
        out[num_selected + ranks_[ITEM]] = values_[ITEM];
      }
    }
  }

  inline void exchange_and_store_keys(KeyT* keys_out, int num_elements) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      local_storage_.exchange_ukeys[ranks_[ITEM]] = ukeys_[ITEM];
    }
    item_.barrier(sycl_local_fence);
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      int offset = lid_ + ITEM * GROUP_THREADS;
      auto ukey = local_storage_.exchange_ukeys[offset];
      relative_bin_offsets_[ITEM] =
          local_storage_.relative_bin_offsets[extract_digit(ukey)];
      offset += relative_bin_offsets_[ITEM];
      if (offset < num_elements) {
        keys_out[offset] = KeyTraits<KeyT>::deconvert(ukey);
      }
    }
    item_.barrier(sycl_local_fence);
  }

  inline void exchange_and_store_values(ValueT* values_out, int num_elements) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      local_storage_.exchange_values[ranks_[ITEM]] = values_[ITEM];
    }
    item_.barrier(sycl_local_fence);
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      int offset = lid_ + ITEM * GROUP_THREADS;
      auto value = local_storage_.exchange_values[offset];
      offset += relative_bin_offsets_[ITEM];
      if (offset < num_elements) {
        values_out[offset] = value;
      }
    }
    item_.barrier(sycl_local_fence);
  }

  inline void exchange_keys() {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      local_storage_.exchange_ukeys[ranks_[ITEM]] = ukeys_[ITEM];
    }
    item_.barrier(sycl_local_fence);
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      int offset = lid_ * KEYS_PER_THREAD + ITEM;
      ukeys_[ITEM] = local_storage_.exchange_ukeys[offset];
    }
    item_.barrier(sycl_local_fence);
  }

  inline void exchange_keys(
      int lower_offset,
      int upper_offset,
      uint32_t* mask) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      if (ranks_[ITEM] >= lower_offset && ranks_[ITEM] < upper_offset) {
        local_storage_.exchange_ukeys[ranks_[ITEM] - lower_offset] =
            ukeys_[ITEM];
      }
    }
    item_.barrier(sycl_local_fence);
    *mask = 0u;
    int new_length = upper_offset - lower_offset;
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      int offset = lid_ * KEYS_PER_THREAD + ITEM;
      if (offset < new_length) {
        *mask |= (1u << ITEM);
        ukeys_[ITEM] = local_storage_.exchange_ukeys[offset];
      }
    }
    item_.barrier(sycl_local_fence);
  }

  inline void exchange_values() {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      local_storage_.exchange_values[ranks_[ITEM]] = values_[ITEM];
    }
    item_.barrier(sycl_local_fence);
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      int offset = lid_ * KEYS_PER_THREAD + ITEM;
      values_[ITEM] = local_storage_.exchange_values[offset];
    }
    item_.barrier(sycl_local_fence);
  }

  inline void exchange_values(int lower_offset, int upper_offset) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      if (ranks_[ITEM] >= lower_offset && ranks_[ITEM] < upper_offset) {
        local_storage_.exchange_values[ranks_[ITEM] - lower_offset] =
            values_[ITEM];
      }
    }
    item_.barrier(sycl_local_fence);
    int new_length = upper_offset - lower_offset;
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      int offset = lid_ * KEYS_PER_THREAD + ITEM;
      if (offset < new_length) {
        values_[ITEM] = local_storage_.exchange_values[offset];
      }
    }
    item_.barrier(sycl_local_fence);
  }

  inline DigitT extract_digit(KeyTraitsT key) {
    return ((key >> begin_bit_) & ((1 << pass_bits_) - 1));
  }

  inline void rank_keys(int begin_bit, int end_bit) {
    begin_bit_ = begin_bit;
    pass_bits_ = end_bit - begin_bit_;
    pass_bits_ = RADIX_BITS < pass_bits_ ? RADIX_BITS : pass_bits_;
    DigitT* digit_counters[KEYS_PER_THREAD];

    // reset buckets
#pragma unroll
    for (int ITEM = 0; ITEM < COUNTER_LANES; ++ITEM) {
      local_storage_.rank_storage.counters[ITEM][lid_] = 0;
    }
    item_.barrier(sycl_local_fence);

#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      auto digit = extract_digit(ukeys_[ITEM]);
      auto sub_counter = digit >> LOG_COUNTER_LANES;
      auto counter_lane = digit & (COUNTER_LANES - 1);
      if (IS_DESCENDING) {
        sub_counter = PACKING_RATIO - 1 - sub_counter;
        counter_lane = COUNTER_LANES - 1 - counter_lane;
      }
      digit_counters[ITEM] =
          &local_storage_.rank_storage.buckets[counter_lane][lid_][sub_counter];
      ranks_[ITEM] = *digit_counters[ITEM];
      *digit_counters[ITEM] = ranks_[ITEM] + 1;
    }
    item_.barrier(sycl_local_fence);

    CounterT exclusive = group_exclusive_cumsum<
        CounterT,
        COUNTER_LANES,
        GROUP_THREADS,
        SUBGROUP_SIZE>(local_storage_.rank_storage.counters_flat, item_);

    CounterT c = 0;
#pragma unroll
    for (int STEP = 1; STEP < PACKING_RATIO; ++STEP) {
      exclusive = exclusive << DIGIT_BITS;
      c += exclusive;
    }

#pragma unroll
    for (int INDEX = 0; INDEX < COUNTER_LANES; ++INDEX) {
      local_storage_.rank_storage.counters[INDEX][lid_] += c;
    }
    item_.barrier(sycl_local_fence);

    // inc rank
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      ranks_[ITEM] += *digit_counters[ITEM];
    }
    item_.barrier(sycl_local_fence);

    if (enable_bin_offsets_) {
      int digit = lid_;
      if (lid_ < RADIX_BUCKETS) {
        if (IS_DESCENDING)
          digit = RADIX_BUCKETS - digit - 1;
        auto sub_counter = digit >> LOG_COUNTER_LANES;
        auto counter_lane = digit & (COUNTER_LANES - 1);
        int digit_offset =
            local_storage_.rank_storage.buckets[counter_lane][0][sub_counter];
        local_storage_.relative_bin_offsets[lid_] = bin_offset_ - digit_offset;
      }
      item_.barrier(sycl_local_fence);
    }
  }

  inline void find_select_offset(
      int carry,
      int num_to_select,
      int* out_offset_select,
      int* out_offset_active) {
    *out_offset_select = 0;
    *out_offset_active = 0;
#pragma unroll
    for (int DIGIT = 1; DIGIT < RADIX_BUCKETS; ++DIGIT) {
      auto sub_counter = DIGIT >> LOG_COUNTER_LANES;
      auto counter_lane = DIGIT & (COUNTER_LANES - 1);
      auto count = (int)(local_storage_.rank_storage
                             .buckets[counter_lane][0][sub_counter]);
      if (count > num_to_select) {
        *out_offset_active = count;
        break;
      }
      *out_offset_select = count;
    }
    if (*out_offset_active == 0)
      *out_offset_active = carry;
  }

  inline void rank_keys(
      int begin_bit,
      int pass_bits,
      uint32_t active_mask,
      int num_to_select,
      int* out_offset_select,
      int* out_offset_active) {
    begin_bit_ = begin_bit;
    pass_bits_ = pass_bits;
    DigitT* digit_counters[KEYS_PER_THREAD];

    // reset buckets
#pragma unroll
    for (int ITEM = 0; ITEM < COUNTER_LANES; ++ITEM) {
      local_storage_.rank_storage.counters[ITEM][lid_] = 0;
    }
    item_.barrier(sycl_local_fence);

#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      ranks_[ITEM] = PROCESSING_LENGTH;
      if (active_mask >> ITEM & 1) {
        auto digit = extract_digit(ukeys_[ITEM]);
        auto sub_counter = digit >> LOG_COUNTER_LANES;
        auto counter_lane = digit & (COUNTER_LANES - 1);
        if (IS_DESCENDING) {
          sub_counter = PACKING_RATIO - 1 - sub_counter;
          counter_lane = COUNTER_LANES - 1 - counter_lane;
        }
        digit_counters[ITEM] = &local_storage_.rank_storage
                                    .buckets[counter_lane][lid_][sub_counter];
        ranks_[ITEM] = *digit_counters[ITEM];
        *digit_counters[ITEM] = ranks_[ITEM] + 1;
      }
    }
    item_.barrier(sycl_local_fence);

    CounterT exclusive = group_exclusive_cumsum<
        CounterT,
        COUNTER_LANES,
        GROUP_THREADS,
        SUBGROUP_SIZE>(local_storage_.rank_storage.counters_flat, item_);

    int carry = 0;
#pragma unroll
    for (int STEP = 0; STEP < PACKING_RATIO; ++STEP) {
      DigitT cc = (exclusive >> (STEP * DIGIT_BITS)) & DIGIT_MASK;
      carry += cc;
    }

    CounterT c = 0;
#pragma unroll
    for (int STEP = 1; STEP < PACKING_RATIO; ++STEP) {
      exclusive = exclusive << DIGIT_BITS;
      c += exclusive;
    }

#pragma unroll
    for (int INDEX = 0; INDEX < COUNTER_LANES; ++INDEX) {
      local_storage_.rank_storage.counters[INDEX][lid_] += c;
    }
    item_.barrier(sycl_local_fence);

    // inc rank
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      ranks_[ITEM] += *digit_counters[ITEM];
    }
    item_.barrier(sycl_local_fence);

    find_select_offset(
        carry, num_to_select, out_offset_select, out_offset_active);

    item_.barrier(sycl_local_fence);
  }

  inline void topk(
      int begin_bit,
      int end_bit,
      int k,
      KeyT* out_keys,
      ValueT* out_values) {
    uint32_t active_mask = 0xffffffff;
    int num_selected = 0;
    while (true) {
      int pass_bits = begin_bit - end_bit;
      pass_bits = pass_bits < RADIX_BITS ? pass_bits : RADIX_BITS;
      begin_bit -= pass_bits;
      int offset_select, offset_active;
      rank_keys(
          begin_bit,
          pass_bits,
          active_mask,
          k - num_selected,
          &offset_select,
          &offset_active);
      if (begin_bit == end_bit)
        offset_select = k - num_selected;
      if (offset_select > 0) {
        store_keys(out_keys, offset_select, num_selected);
        if (!KEYS_ONLY)
          store_values(out_values, offset_select, num_selected);
      }
      num_selected += offset_select;
      if (num_selected == k)
        break;
      exchange_keys(offset_select, offset_active, &active_mask);
      if (!KEYS_ONLY)
        exchange_values(offset_select, offset_active);
    }
  }

  inline void topk_append_keys(
      const KeyT* keys_in,
      const KeyT* keys_temp,
      int num_elements,
      int num_start,
      int k) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      int offset = lid_ * KEYS_PER_THREAD + ITEM;
      if (offset < k) {
        ukeys_[ITEM] = KeyTraits<KeyT>::convert(c10::load(&keys_temp[offset]));
      } else {
        offset += num_start - k;
        if (offset < num_elements) {
          ukeys_[ITEM] = KeyTraits<KeyT>::convert(c10::load(&keys_in[offset]));
        } else {
          KeyTraitsT padding_key;
          if (IS_DESCENDING) {
            padding_key = 0;
          } else {
            constexpr uint64_t KEY_TRAITS_TYPE_MASK = 1ll
                << ((sizeof(KeyTraitsT) << 3) - 1);
            padding_key = static_cast<KeyTraitsT>(KEY_TRAITS_TYPE_MASK);
            padding_key = padding_key ^ (padding_key - 1);
          }
          ukeys_[ITEM] = padding_key;
        }
      }
    }
  }

  inline void topk_append_values(
      const ValueT* values_in,
      const ValueT* values_temp,
      int num_elements,
      int num_start,
      int k) {
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      int offset = lid_ * KEYS_PER_THREAD + ITEM;
      if (offset < k) {
        values_[ITEM] = values_temp[offset];
      } else {
        offset += num_start - k;
        if (offset < num_elements) {
          if constexpr (IS_INT_TYPE) {
            values_[ITEM] = values_in == nullptr ? offset : values_in[offset];
          } else {
            values_[ITEM] = values_in[offset];
          }
        }
      }
    }
  }
};

template <
    typename KeyT,
    int GROUP_THREADS_,
    int SUBGROUP_SIZE_,
    int KEYS_PER_THREAD_,
    bool IS_DESCENDING_ = false,
    typename ValueT = NullType,
    typename DigitT = unsigned char,
    typename CounterT = uint32_t, // Packed scan datatype
    // We are going to bundle multiple counters with 'DigitT' type to perform
    // packed prefix sum.
    int RADIX_BITS = 4>
class RadixSortUpsweep {
 public:
  using KeyTraitsT = typename KeyTraits<KeyT>::Type;
  enum {
    GROUP_THREADS = GROUP_THREADS_,
    SUBGROUP_SIZE = SUBGROUP_SIZE_,
    KEYS_PER_THREAD = KEYS_PER_THREAD_,
    IS_DESCENDING = IS_DESCENDING_,

    PROCESSING_LENGTH = GROUP_THREADS * KEYS_PER_THREAD,
    RADIX_BUCKETS = 1 << RADIX_BITS,
    KEYS_ONLY = std::is_same<ValueT, NullType>::value,
    PACKING_RATIO = sizeof(CounterT) / sizeof(DigitT),
    LOG_PACKING_RATIO = Log2<PACKING_RATIO>::VALUE,
    COUNTER_LANES = RADIX_BUCKETS / PACKING_RATIO,

    SUBGROUPS = (GROUP_THREADS + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE,
    LANES_PER_SUBGROUP =
        std::max<int>(1, (COUNTER_LANES + SUBGROUPS - 1) / SUBGROUPS),
  };

  static_assert(sizeof(CounterT) >= sizeof(DigitT), "");
  static_assert(sizeof(CounterT) % sizeof(DigitT) == 0, "");

 private:
  union LocalStorage {
    CounterT counters[COUNTER_LANES][GROUP_THREADS];
    DigitT buckets[COUNTER_LANES][GROUP_THREADS][PACKING_RATIO];
    int group_counters[SUBGROUP_SIZE][RADIX_BUCKETS];
  };

  sycl::nd_item<1>& item_;
  const KeyT* keys_in_;
  int lid_;
  int gid_;
  int begin_bit_;
  int end_bit_;
  int num_groups_;
  int* count_out_;
  int subgroup_id_;
  int subgroup_tid_;

  LocalStorage& local_storage_;
  int local_counts_[LANES_PER_SUBGROUP][PACKING_RATIO];

 public:
  static int LocalMemorySize() {
    return sizeof(LocalStorage);
  }

  inline RadixSortUpsweep(
      sycl::nd_item<1>& item,
      const KeyT* keys_in,
      int gid,
      int begin_bit,
      int end_bit,
      int num_groups,
      int* count_out,
      sycl_local_acc_t<char> local_ptr)
      : item_(item),
        keys_in_(keys_in),
        lid_(item.get_local_id(0)),
        gid_(gid),
        begin_bit_(begin_bit),
        end_bit_(end_bit),
        num_groups_(num_groups),
        count_out_(count_out),
        local_storage_(reinterpret_cast<LocalStorage&>(
            *(local_ptr.template get_multi_ptr<sycl::access::decorated::no>()
                  .get()))) {
    subgroup_id_ = lid_ / SUBGROUP_SIZE;
    subgroup_tid_ = lid_ % SUBGROUP_SIZE;
  }

  inline DigitT extract_digit(KeyTraitsT key) {
    auto pass_bits = end_bit_ - begin_bit_;
    pass_bits = RADIX_BITS < pass_bits ? RADIX_BITS : pass_bits;
    return ((key >> begin_bit_) & ((1 << pass_bits) - 1));
  }

  inline void process_full_tile(int group_offset) {
    KeyTraitsT keys[KEYS_PER_THREAD];
    auto group_ptr = keys_in_ + group_offset;
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      keys[ITEM] = KeyTraits<KeyT>::convert(
          c10::load(&group_ptr[lid_ + ITEM * GROUP_THREADS]));
    }
    item_.barrier(sycl_local_fence);
#pragma unroll
    for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
      auto digit = extract_digit(keys[ITEM]);
      auto sub_counter = digit & (PACKING_RATIO - 1);
      auto row_offset = digit >> LOG_PACKING_RATIO;
      local_storage_.buckets[row_offset][lid_][sub_counter]++;
    }
  }

  inline void process_partial_tile(int group_offset, int group_end) {
    for (int offset = group_offset + lid_; offset < group_end;
         offset += GROUP_THREADS) {
      KeyTraitsT key = KeyTraits<KeyT>::convert(c10::load(&keys_in_[offset]));
      auto digit = extract_digit(key);
      auto sub_counter = digit & (PACKING_RATIO - 1);
      auto row_offset = digit >> LOG_PACKING_RATIO;
      local_storage_.buckets[row_offset][lid_][sub_counter]++;
    }
  }

  inline void reset_digit_counters() {
#pragma unroll
    for (int LANE = 0; LANE < COUNTER_LANES; ++LANE)
      local_storage_.counters[LANE][lid_] = 0;
  }

  inline void reset_unpacked_counters() {
#pragma unroll
    for (int LANE = 0; LANE < LANES_PER_SUBGROUP; ++LANE) {
#pragma unroll
      for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO;
           ++UNPACKED_COUNTER) {
        local_counts_[LANE][UNPACKED_COUNTER] = 0;
      }
    }
  }

  inline void unpack_digit_counts() {
#pragma unroll
    for (int LANE = 0; LANE < LANES_PER_SUBGROUP; ++LANE) {
      int counter_lane = (LANE * SUBGROUPS) + subgroup_id_;
      if (counter_lane < COUNTER_LANES) {
#pragma unroll
        for (int PACKED_COUNTER = 0; PACKED_COUNTER < GROUP_THREADS;
             PACKED_COUNTER += SUBGROUP_SIZE) {
#pragma unroll
          for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO;
               ++UNPACKED_COUNTER) {
            int counter =
                local_storage_
                    .buckets[counter_lane][subgroup_tid_ + PACKED_COUNTER]
                            [UNPACKED_COUNTER];
            local_counts_[LANE][UNPACKED_COUNTER] += counter;
          }
        }
      }
    }
  }

  inline void extract_counts() {
#pragma unroll
    for (int LANE = 0; LANE < LANES_PER_SUBGROUP; ++LANE) {
      int counter_lane = (LANE * SUBGROUPS) + subgroup_id_;
      if (counter_lane < COUNTER_LANES) {
        int digit_row = counter_lane << LOG_PACKING_RATIO;
#pragma unroll
        for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO;
             ++UNPACKED_COUNTER) {
          int bin_idx = digit_row + UNPACKED_COUNTER;
          local_storage_.group_counters[subgroup_tid_][bin_idx] =
              local_counts_[LANE][UNPACKED_COUNTER];
        }
      }
    }

    item_.barrier(sycl_local_fence);

    if ((RADIX_BUCKETS % GROUP_THREADS != 0) && (lid_ < RADIX_BUCKETS)) {
      int bin_idx = lid_;
      int bin_count = 0;
#pragma unroll
      for (int i = 0; i < SUBGROUP_SIZE; ++i)
        bin_count += local_storage_.group_counters[i][bin_idx];
      if (IS_DESCENDING)
        bin_idx = RADIX_BUCKETS - bin_idx - 1;
      count_out_[(num_groups_ * bin_idx) + gid_] = bin_count;
    }
  }

  inline void run(int group_offset, int group_end) {
    reset_digit_counters();
    reset_unpacked_counters();

    // Unroll batches of full tiles
    int UNROLL_COUNT = 255 / 4; // the largest value for counter
    int UNROLLED_ELEMENTS = UNROLL_COUNT * PROCESSING_LENGTH;
    while (group_offset + UNROLLED_ELEMENTS <= group_end) {
      for (int i = 0; i < UNROLL_COUNT; ++i) {
        process_full_tile(group_offset);
        group_offset += PROCESSING_LENGTH;
      }
      item_.barrier(sycl_local_fence);
      unpack_digit_counts();
      item_.barrier(sycl_local_fence);
      reset_digit_counters();
    }

    while (group_offset + PROCESSING_LENGTH <= group_end) {
      process_full_tile(group_offset);
      group_offset += PROCESSING_LENGTH;
    }

    process_partial_tile(group_offset, group_end);
    item_.barrier(sycl_local_fence);
    unpack_digit_counts();
    item_.barrier(sycl_local_fence);
    extract_counts();
  }
};

template <int GROUP_THREADS, int THREAD_WORK_SIZE, int SUBGROUP_SIZE_>
class RadixSortScanBins {
 public:
  enum {
    SUBGROUP_SIZE = SUBGROUP_SIZE_,
    PROCESSING_LENGTH = GROUP_THREADS * THREAD_WORK_SIZE,
    NUM_SUBGROUPS = GROUP_THREADS / SUBGROUP_SIZE,
  };

 private:
  sycl::nd_item<1>& item_;
  int* count_;
  int* slm_;
  int lid_;

 public:
  static int LocalMemorySize() {
    return NUM_SUBGROUPS * sizeof(int);
  }

  inline RadixSortScanBins(
      sycl::nd_item<1>& item,
      int* count,
      sycl_local_acc_t<char> slm)
      : item_(item),
        count_(count),
        slm_(reinterpret_cast<int*>(
            slm.template get_multi_ptr<sycl::access::decorated::no>().get())),
        lid_(item.get_local_id(0)) {}

  template <bool is_partial>
  inline void consume_tile(
      int group_offset,
      int& running_prefix,
      int tile_bound = 0) {
    // Load
    int partial_output[THREAD_WORK_SIZE];
    auto d_local = count_ + group_offset;
#pragma unroll
    for (int ITEM = 0; ITEM < THREAD_WORK_SIZE; ++ITEM) {
      int offset = lid_ * THREAD_WORK_SIZE + ITEM;
      if constexpr (is_partial) {
        if (offset < tile_bound) {
          partial_output[ITEM] = d_local[offset];
        } else {
          partial_output[ITEM] = *d_local;
        }
      } else {
        partial_output[ITEM] = d_local[offset];
      }
    }
    item_.barrier(sycl_local_fence);
    // Thread reduce
    int thread_partial = partial_output[0];
#pragma unroll
    for (int ITEM = 1; ITEM < THREAD_WORK_SIZE; ++ITEM) {
      thread_partial = thread_partial + partial_output[ITEM];
    }
    // Subgroup scan
    int subgroup_tid = lid_ % SUBGROUP_SIZE;
    int subgroup_id = lid_ / SUBGROUP_SIZE;
    const int SUBGROUP_SCAN_STEPS = Log2<SUBGROUP_SIZE>::VALUE;
    int subgroup_inclusive_sum, subgroup_exclusive_sum;
    auto subgroup = item_.get_sub_group();
    subgroup_cumsum<int, SUBGROUP_SCAN_STEPS>(
        subgroup,
        subgroup_tid,
        thread_partial,
        subgroup_inclusive_sum,
        subgroup_exclusive_sum);
    if (subgroup_tid == (SUBGROUP_SIZE - 1))
      slm_[subgroup_id] = subgroup_inclusive_sum;
    item_.barrier(sycl_local_fence);
    // Group scan
    int group_all_sum = 0, subgroup_prefix_sum;
#pragma unroll
    for (int i = 0; i < NUM_SUBGROUPS; ++i) {
      if (subgroup_id == i)
        subgroup_prefix_sum = group_all_sum;
      group_all_sum += slm_[i];
    }
    subgroup_exclusive_sum += subgroup_prefix_sum;
    subgroup_exclusive_sum += running_prefix;
    running_prefix += group_all_sum;
    // Write back
    int inclusive = partial_output[0];
    inclusive = subgroup_exclusive_sum + inclusive;
    partial_output[0] = subgroup_exclusive_sum;
    int exclusive = inclusive;
#pragma unroll
    for (int ITEM = 1; ITEM < THREAD_WORK_SIZE; ++ITEM) {
      inclusive = exclusive + partial_output[ITEM];
      partial_output[ITEM] = exclusive;
      exclusive = inclusive;
    }
#pragma unroll
    for (int ITEM = 0; ITEM < THREAD_WORK_SIZE; ITEM++) {
      int offset = lid_ * THREAD_WORK_SIZE + ITEM;
      if constexpr (is_partial) {
        if (offset < tile_bound) {
          d_local[offset] = partial_output[ITEM];
        }
      } else {
        d_local[offset] = partial_output[ITEM];
      }
    }
  }

  inline void run(int num_counts) {
    int group_offset = 0;
    int running_prefix = 0;
    while (group_offset + PROCESSING_LENGTH <= num_counts) {
      consume_tile<false>(group_offset, running_prefix);
      group_offset += PROCESSING_LENGTH;
    }
    if (group_offset < num_counts) {
      consume_tile<true>(
          group_offset, running_prefix, num_counts - group_offset);
    }
  }
};

} // namespace xpu
} // namespace native
} // namespace at
