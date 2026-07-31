/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// SYCL kernels for _philox_key_split and _philox_key_fold_in.
// Ported from CUDA: aten/src/ATen/native/cuda/PhiloxKeySplit.cu
// See PyTorch PR #177229.

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/Philox4x32.h>
#include <ATen/native/xpu/sycl/PhiloxKeySplitKernels.h>
#include <comm/SYCLContext.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_philox_key_fold_in_native.h>
#include <ATen/ops/_philox_key_split_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

namespace at::native::xpu {

inline void philox_derive_key(
    uint4 r,
    uint64_t* out_seed,
    uint64_t* out_offset) {
  *out_seed = static_cast<uint64_t>(r.x) | (static_cast<uint64_t>(r.y) << 32);
  *out_offset = static_cast<uint64_t>(r.z) | (static_cast<uint64_t>(r.w) << 32);
}

struct PhiloxKeySplitFunctor {
  void operator()(sycl::nd_item<1> item) const {
    const int64_t global_id = static_cast<int64_t>(item.get_global_id(0));
    const int64_t global_range = static_cast<int64_t>(item.get_global_range(0));

    for (int64_t idx = global_id; idx < total_; idx += global_range) {
      const int64_t split_idx = idx / num_keys_;
      const int64_t key_idx = idx % num_keys_;

      const uint64_t seed = input_[key_idx * 2];
      const uint64_t offset = input_[key_idx * 2 + 1];
      const uint64_t split_offset = offset + static_cast<uint64_t>(split_idx);

      const uint2 key = {
          static_cast<uint32_t>(seed),
          static_cast<uint32_t>(seed >> 32),
      };

      const uint4 counter = {
          static_cast<uint32_t>(split_offset),
          static_cast<uint32_t>(split_offset >> 32),
          0,
          0,
      };

      const auto r = philox4x32_10(counter, key);

      const int64_t out = idx * 2;
      philox_derive_key(r, &output_[out], &output_[out + 1]);
    }
  }

  PhiloxKeySplitFunctor(
      const uint64_t* input,
      uint64_t* output,
      int64_t num_keys,
      int64_t total)
      : input_(input), output_(output), num_keys_(num_keys), total_(total) {}

 private:
  const uint64_t* input_;
  uint64_t* output_;
  int64_t num_keys_;
  int64_t total_;
};

struct PhiloxKeyFoldInFunctor {
  void operator()(sycl::nd_item<1> item) const {
    for (int64_t idx = (item.get_global_id(0)); idx < num_keys_;
         idx += item.get_global_range(0)) {
      uint64_t seed = input_[idx * 2];
      uint64_t offset = input_[idx * 2 + 1];

      uint2 key = {
          static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
      uint4 ctr = {
          static_cast<uint32_t>(offset + static_cast<uint64_t>(data_)),
          static_cast<uint32_t>((offset + static_cast<uint64_t>(data_)) >> 32),
          // restrict subsequence=0
          0,
          0};

      auto r = philox4x32_10(counter, key);
      philox_derive_key(r, &output_[idx * 2], &output_[idx * 2 + 1]);
    }
  }

  PhiloxKeyFoldInFunctor(
      const uint64_t* input,
      uint64_t* output,
      int64_t num_keys,
      int64_t data)
      : input_(input), output_(output), num_keys_(num_keys), data_(data) {}

 private:
  const uint64_t* input_;
  uint64_t* output_;
  int64_t num_keys_;
  int64_t data_;
};

Tensor _philox_key_split_xpu(const Tensor& key, int64_t num_splits) {
  TORCH_CHECK(
      key.dim() >= 1 && key.size(-1) == 2,
      "_philox_key_split: key must have shape (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(
      key.scalar_type() == kUInt64,
      "_philox_key_split: key must have dtype uint64, got ",
      key.scalar_type());
  TORCH_CHECK(
      num_splits > 0,
      "_philox_key_split: num_splits must be positive, got ",
      num_splits);

  // Output shape: (num_splits, *key.shape)
  auto output_sizes = key.sizes().vec();
  output_sizes.insert(output_sizes.begin(), num_splits);
  Tensor output = at::empty(output_sizes, key.options());
  int64_t num_keys = key.numel() / 2;
  if (num_keys == 0) {
    return output;
  }

  const int64_t total_elements = num_keys * num_splits;
  constexpr int64_t work_group_size =
      256; // TODO: wg_size 256 on performance of XPU remains to be investigated
  const int64_t work_group_num =
      (total_elements + work_group_size - 1) / work_group_size;
  auto key_contig = key.contiguous();
  auto functor = PhiloxKeySplitFunctor(
      key_contig.data_ptr<uint64_t>(),
      output.data_ptr<uint64_t>(),
      num_keys,
      total_elements);

  sycl_kernel_submit(
      sycl::range<1>(work_group_num * work_group_size),
      sycl::range<1>(work_group_size),
      at::xpu::getCurrentSYCLQueue(),
      functor);

  return output;
}

Tensor _philox_key_fold_in_xpu(const Tensor& key, int64_t data) {
  TORCH_CHECK(
      key.dim() >= 1 && key.size(-1) == 2,
      "_philox_key_fold_in: key must have shape (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(
      key.scalar_type() == kUInt64,
      "_philox_key_fold_in: key must have dtype uint64, got ",
      key.scalar_type());

  Tensor output = at::empty_like(key);
  int64_t num_keys = key.numel() / 2;
  if (num_keys == 0) {
    return output;
  }

  constexpr int64_t work_group_size =
      256; // TODO: wg_size 256 on performance of XPU remains to be investigated
  const int64_t work_group_num =
      (num_keys + work_group_size - 1) / work_group_size;
  auto key_contig = key.contiguous();
  auto functor = PhiloxKeyFoldInFunctor(
      key_contig.data_ptr<uint64_t>(),
      output.data_ptr<uint64_t>(),
      num_keys,
      data);

  sycl_kernel_submit(
      sycl::range<1>(work_group_num * work_group_size),
      sycl::range<1>(work_group_size),
      at::xpu::getCurrentSYCLQueue(),
      functor);

  return output;
}

} // namespace at::native::xpu
