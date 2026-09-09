/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch.h>
#include <ATen/native/xpu/sycl/Philox4x32.h>
#include <ATen/native/xpu/sycl/SortingKernels.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <comm/SYCLContext.h>
#include <comm/xpu_aten.h>

#include <ATen/ops/arange.h>

#include <ATen/native/xpu/sycl/RandpermKernel.h>

namespace at::native::xpu {

template <int N>
struct alignas(N) OpaqueType {
  char data[N];
};

template <typename T, typename scalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void handle_duplicate_keys_kernel(
    const T* keys,
    scalar_t* data,
    const T mask,
    const int n,
    const PhiloxXpuState philox_args) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  auto tid = item.get_global_id(0);
  // find the beginning of islands
  if (tid >= n - 1)
    return; // out of range
  if ((keys[tid] & mask) != (keys[tid + 1] & mask))
    return; // not in an island
  if (tid != 0 && (keys[tid] & mask) == (keys[tid - 1] & mask))
    return; // not the beginning of an island

  // find the size of islands
  int island_size = 0;
  do {
    island_size++;
  } while ((tid + island_size < n) &&
           (keys[tid + island_size] & mask) == (keys[tid] & mask));

  // do random permutation inside each island.
  data += tid;
  auto seeds = at::xpu::philox::unpack(philox_args);
  randStatePhilox4_32_10_t state;
  rand_init(std::get<0>(seeds), tid, std::get<1>(seeds), &state);

  for (int i = island_size - 1; i > 0; i--) {
    unsigned int r = rand(&state) % (i + 1);
    if (i != r) {
      scalar_t tmp = data[i];
      data[i] = data[r];
      data[r] = tmp;
    }
  }
}

// See note [Algorithm of randperm]
template <typename T, typename scalar_t>
void randperm_handle_duplicate_keys(
    T* keys,
    scalar_t* data,
    int bits,
    int64_t n,
    std::optional<at::Generator>& gen_) {
  auto gen = get_generator_or_default<at::XPUGeneratorImpl>(
      gen_, at::xpu::detail::getDefaultXPUGenerator());

  int64_t counter_offset = n;

  PhiloxXpuState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_xpu_state(counter_offset);
  }

  T mask = static_cast<T>((1UL << bits) - 1);

  auto local_range =
      syclMaxWorkGroupSize<handle_duplicate_keys_kernel<T, scalar_t>>() / 2;
  auto num_wg = (n + local_range - 1) / local_range;
  auto global_range = num_wg * local_range;

  sycl_kernel_submit<handle_duplicate_keys_kernel<T, scalar_t>>(
      global_range,
      local_range,
      at::xpu::getCurrentSYCLQueue(),
      0,
      keys,
      data,
      mask,
      n,
      rng_engine_inputs);
}

Tensor randperm_kernel(
    Tensor& result,
    int64_t n,
    std::optional<Generator> generator) {
  auto range = at::arange(n, result.options());

  Tensor shuffled;
  void* shuffled_data;
  if (result.is_contiguous()) {
    shuffled_data = result.data_ptr();
  } else {
    shuffled = at::empty(n, result.options());
    shuffled_data = shuffled.data_ptr();
  }

  auto opt = TensorOptions().device(result.device());

  // See note [Algorithm of randperm]
  const double log_threshold_12 = std::log(0.9) * 12;
  double nd = static_cast<double>(n);
  int bits = std::min(
      64,
      static_cast<int>(
          std::ceil(std::log2(nd - (6 * nd * nd + 1) / log_threshold_12))));

  if (n == 0) {
    return result;
  } else if (bits <= 32) {
    // For asserting device type match of the generator and result,
    // we deligate that to the 'random_' function below.
    using key_type = int;
    auto keys = at::empty(result.sizes(), opt.dtype(kInt))
                    .random_(
                        std::numeric_limits<key_type>::min(),
                        std::numeric_limits<key_type>::max(),
                        generator);
    auto keys_tmp = at::empty_like(keys);
    auto keys_out = keys_tmp.mutable_data_ptr<key_type>();
    AT_DISPATCH_ALL_TYPES_AND(kHalf, result.scalar_type(), "randperm_xpu", [&] {
      using dtype = OpaqueType<sizeof(scalar_t)>;
      auto shuffled_data_ = reinterpret_cast<dtype*>(shuffled_data);
      auto* range_data = reinterpret_cast<const dtype*>(range.const_data_ptr());
      sort_pairs<key_type, dtype>(
          keys.const_data_ptr<key_type>(),
          keys_out,
          range_data,
          shuffled_data_,
          n,
          false);

      randperm_handle_duplicate_keys(
          keys_out, shuffled_data_, bits, n, generator);
    });
  } else {
    using key_type = int64_t;
    auto keys = at::empty(result.sizes(), opt.dtype(kLong))
                    .random_(
                        std::numeric_limits<key_type>::min(),
                        std::numeric_limits<key_type>::max(),
                        generator);
    auto keys_tmp = at::empty_like(keys);
    auto keys_out = keys_tmp.mutable_data_ptr<key_type>();
    AT_DISPATCH_ALL_TYPES_AND(kHalf, result.scalar_type(), "randperm_xpu", [&] {
      using dtype = OpaqueType<sizeof(scalar_t)>;
      auto shuffled_data_ = reinterpret_cast<dtype*>(shuffled_data);
      auto* range_data = reinterpret_cast<const dtype*>(range.const_data_ptr());

      sort_pairs<key_type, dtype>(
          keys.const_data_ptr<key_type>(),
          keys_out,
          range_data,
          shuffled_data_,
          n,
          false);

      randperm_handle_duplicate_keys(
          keys_out, shuffled_data_, bits, n, generator);
    });
  }

  if (!result.is_contiguous()) {
    result.copy_(shuffled);
  }

  return result;
}
} // namespace at::native::xpu
