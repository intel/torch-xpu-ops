#pragma once

#include <ATen/detail/FunctionTraits.h>
#include <aten/sycl/SortingRadixSort.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <comm/SYCLContext.h>

namespace at {
namespace native {
namespace xpu {

// ======================= group sort =======================

template <typename method_t, typename key_t, typename value_t>
struct SegmentedGroupRadixSortPairsFunctor {
  [[intel::reqd_sub_group_size(method_t::SUBGROUP_SIZE)]] void operator()(
      sycl::nd_item<1> item) const {
    int seg_idx = item.get_group(0);
    int seg_offset = seg_idx * num_elements_;
    int lid = item.get_local_id(0);
    auto method = method_t(item, slm_);
    method.load_keys(keys_in_ + seg_offset, num_elements_);
    method.load_values(
        values_in_ == nullptr ? nullptr : values_in_ + seg_offset,
        num_elements_);
    int begin_bit = 0;
    int end_bit = 8 * sizeof(key_t);
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
  SegmentedGroupRadixSortPairsFunctor(
      const key_t* keys_in,
      key_t* keys_out,
      const value_t* values_in,
      value_t* values_out,
      int num_elements,
      sycl_local_acc_t<char> slm)
      : keys_in_(keys_in),
        keys_out_(keys_out),
        values_in_(values_in),
        values_out_(values_out),
        num_elements_(num_elements),
        slm_(slm) {}

 private:
  const key_t* keys_in_;
  key_t* keys_out_;
  const value_t* values_in_;
  value_t* values_out_;
  int num_elements_;
  sycl_local_acc_t<char> slm_;
};

template <typename method_t, typename key_t, typename value_t>
struct SegmentedGroupRadixSortPairsFunctorCreator {
  auto operator()(::sycl::handler& cgh) const {
    sycl_local_acc_t<char> shared(method_t::LocalMemorySize(), cgh);
    return SegmentedGroupRadixSortPairsFunctor<method_t, key_t, value_t>(
        keys_in_, keys_out_, values_in_, values_out_, num_elements_, shared);
  }
  SegmentedGroupRadixSortPairsFunctorCreator(
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
  auto creator =
      SegmentedGroupRadixSortPairsFunctorCreator<method_t, key_t, value_t>(
          keys_in, keys_out, values_in, values_out, num_elements);
  sycl_kernel_submit<typename function_traits<decltype(creator)>::result_type>(
      num_segments * GROUP_SIZE,
      GROUP_SIZE,
      at::xpu::getCurrentSYCLQueue(),
      creator);
}

// ======================= upsweep =======================

template <typename method_t, typename key_t, typename value_t>
struct SegmentedRadixSortPairsUpsweepFunctor {
  [[intel::reqd_sub_group_size(method_t::SUBGROUP_SIZE)]] void operator()(
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
  SegmentedRadixSortPairsUpsweepFunctor(
      const key_t* keys_in,
      int* counts,
      int num_elements,
      int begin_bit,
      int end_bit,
      sycl_local_acc_t<char> slm)
      : keys_in_(keys_in),
        counts_(counts),
        num_elements_(num_elements),
        begin_bit_(begin_bit),
        end_bit_(end_bit),
        slm_(slm) {}

 private:
  const key_t* keys_in_;
  int* counts_;
  int num_elements_;
  int begin_bit_;
  int end_bit_;
  sycl_local_acc_t<char> slm_;
};

template <typename method_t, typename key_t, typename value_t>
struct SegmentedRadixSortPairsUpsweepFunctorCreator {
  auto operator()(::sycl::handler& cgh) const {
    sycl_local_acc_t<char> shared(method_t::LocalMemorySize(), cgh);
    return SegmentedRadixSortPairsUpsweepFunctor<method_t, key_t, value_t>(
        keys_in_, counts_, num_elements_, begin_bit_, end_bit_, shared);
  }
  SegmentedRadixSortPairsUpsweepFunctorCreator(
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
  auto creator =
      SegmentedRadixSortPairsUpsweepFunctorCreator<method_t, key_t, value_t>(
          keys_in, counts, num_elements, begin_bit, end_bit);
  sycl_kernel_submit<typename function_traits<decltype(creator)>::result_type>(
      num_segments * num_tiles * GROUP_SIZE,
      GROUP_SIZE,
      at::xpu::getCurrentSYCLQueue(),
      creator);
}

// ======================= scan bins =======================

template <typename method_t>
struct SegmentedRadixSortPairsScanFunctor {
  [[intel::reqd_sub_group_size(method_t::SUBGROUP_SIZE)]] void operator()(
      sycl::nd_item<1> item) const {
    constexpr int RADIX_BUCKETS = 16;
    int seg_idx = item.get_group(0);
    auto counts_seg = counts_ + seg_idx * RADIX_BUCKETS * num_tiles_;
    auto method = method_t(item, counts_seg, slm_);
    method.run(num_tiles_ * RADIX_BUCKETS);
  }
  SegmentedRadixSortPairsScanFunctor(
      int* counts,
      int num_tiles,
      sycl_local_acc_t<char> slm)
      : counts_(counts), num_tiles_(num_tiles), slm_(slm) {}

 private:
  int* counts_;
  int num_tiles_;
  sycl_local_acc_t<char> slm_;
};

template <typename method_t>
struct SegmentedRadixSortPairsScanFunctorCreator {
  auto operator()(::sycl::handler& cgh) const {
    sycl_local_acc_t<char> shared(method_t::LocalMemorySize(), cgh);
    return SegmentedRadixSortPairsScanFunctor<method_t>(
        counts_, num_tiles_, shared);
  }
  SegmentedRadixSortPairsScanFunctorCreator(int* counts, int num_tiles)
      : counts_(counts), num_tiles_(num_tiles) {}

 private:
  int* counts_;
  int num_tiles_;
};

template <int KEYS_PER_ITEM, int GROUP_SIZE, int SUBGROUP_SIZE>
void segmented_radix_sort_pairs_scan_kernel(
    int* counts,
    int num_tiles,
    int num_segments) {
  using method_t = RadixSortScanBins<GROUP_SIZE, KEYS_PER_ITEM, SUBGROUP_SIZE>;
  auto creator =
      SegmentedRadixSortPairsScanFunctorCreator<method_t>(counts, num_tiles);
  sycl_kernel_submit<typename function_traits<decltype(creator)>::result_type>(
      num_segments * GROUP_SIZE,
      GROUP_SIZE,
      at::xpu::getCurrentSYCLQueue(),
      creator);
}

// ======================= downsweep =======================

template <typename method_t, typename key_t, typename value_t>
struct SegmentedRadixSortPairsDownsweepFunctor {
  [[intel::reqd_sub_group_size(method_t::SUBGROUP_SIZE)]] void operator()(
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
  SegmentedRadixSortPairsDownsweepFunctor(
      const key_t* keys_in,
      key_t* keys_out,
      const value_t* values_in,
      value_t* values_out,
      int num_elements,
      int begin_bit,
      int end_bit,
      int* counts,
      sycl_local_acc_t<char> slm)
      : keys_in_(keys_in),
        keys_out_(keys_out),
        values_in_(values_in),
        values_out_(values_out),
        num_elements_(num_elements),
        begin_bit_(begin_bit),
        end_bit_(end_bit),
        counts_(counts),
        slm_(slm) {}

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

template <typename method_t, typename key_t, typename value_t>
struct SegmentedRadixSortPairsDownsweepFunctorCreator {
  auto operator()(::sycl::handler& cgh) const {
    sycl_local_acc_t<char> shared(method_t::LocalMemorySize(), cgh);
    return SegmentedRadixSortPairsDownsweepFunctor<method_t, key_t, value_t>(
        keys_in_,
        keys_out_,
        values_in_,
        values_out_,
        num_elements_,
        begin_bit_,
        end_bit_,
        counts_,
        shared);
  }
  SegmentedRadixSortPairsDownsweepFunctorCreator(
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
  auto creator =
      SegmentedRadixSortPairsDownsweepFunctorCreator<method_t, key_t, value_t>(
          keys_in,
          keys_out,
          values_in,
          values_out,
          num_elements,
          begin_bit,
          end_bit,
          count);
  sycl_kernel_submit<typename function_traits<decltype(creator)>::result_type>(
      num_segments * num_tiles * GROUP_SIZE,
      GROUP_SIZE,
      at::xpu::getCurrentSYCLQueue(),
      creator);
}

// ======================= large sort =======================

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
  constexpr int end_bit = sizeof(key_t) * 8;
  constexpr int RADIX_BITS = 4;
  constexpr int RADIX_BUCKETS = 16;
  int begin_bit = 0;
  int* counts;
  key_t* keys_temp;
  value_t* values_temp;

  at::DataPtr counts_data = c10::xpu::XPUCachingAllocator::get()->allocate(
      num_segments * RADIX_BUCKETS * num_tiles * sizeof(int));
  at::DataPtr keys_temp_data = c10::xpu::XPUCachingAllocator::get()->allocate(
      num_segments * num_elements * sizeof(key_t));
  at::DataPtr values_temp_data = c10::xpu::XPUCachingAllocator::get()->allocate(
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
}

// ======================= interface =======================

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
  if (num_elements > 4096) {
    segmented_radix_sort_pairs_kernel<
        key_t,
        value_t,
        IS_DESCENDING,
        4,
        512,
        SUBGROUP_SIZE>(
        keys_in, keys_out, values_in, values_out, num_segments, num_elements);
  } else if (num_elements > 2048) {
    segmented_group_radix_sort_pairs_kernel<
        key_t,
        value_t,
        IS_DESCENDING,
        4,
        1024,
        SUBGROUP_SIZE>(
        keys_in, keys_out, values_in, values_out, num_segments, num_elements);
  } else if (num_elements > 1024) {
    segmented_group_radix_sort_pairs_kernel<
        key_t,
        value_t,
        IS_DESCENDING,
        4,
        512,
        SUBGROUP_SIZE>(
        keys_in, keys_out, values_in, values_out, num_segments, num_elements);
  } else if (num_elements > 512) {
    segmented_group_radix_sort_pairs_kernel<
        key_t,
        value_t,
        IS_DESCENDING,
        4,
        256,
        SUBGROUP_SIZE>(
        keys_in, keys_out, values_in, values_out, num_segments, num_elements);
  } else if (num_elements > 256) {
    segmented_group_radix_sort_pairs_kernel<
        key_t,
        value_t,
        IS_DESCENDING,
        4,
        128,
        SUBGROUP_SIZE>(
        keys_in, keys_out, values_in, values_out, num_segments, num_elements);
  } else {
    segmented_group_radix_sort_pairs_kernel<
        key_t,
        value_t,
        IS_DESCENDING,
        4,
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

} // namespace xpu
} // namespace native
} // namespace at
