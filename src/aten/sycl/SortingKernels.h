#pragma once

#include <ATen/detail/FunctionTraits.h>
#include <aten/sycl/SortingRadixSort.h>
#include <comm/SYCLContext.h>

namespace at {
namespace native {
namespace xpu {

template <typename method_t, typename key_t, typename value_t>
struct SegmentedGroupRadixSortPairsFunctor {
  [[intel::reqd_sub_group_size(method_t::SUBGROUP_SIZE)]] void operator()(
      sycl::nd_item<1> item) const {
    int seg_idx = item.get_group(0);
    int seg_offset = seg_idx * num_elements_;
    int lid = item.get_local_id(0);
    auto method = method_t(item, slm_);
    method.load_keys(keys_in_ + seg_offset, num_elements_);
    method.load_values(values_in_ + seg_offset, num_elements_);
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
    std::cout << "Using tile sort\n";
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
