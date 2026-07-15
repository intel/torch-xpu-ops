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

#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/PhiloxKeySplitKernels.h>
#include <ATen/native/xpu/sycl/StatelessPhilox4x32.h>
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

struct PhiloxKeySplitFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t tid = static_cast<int64_t>(item.get_global_id(0));
    if (tid >= total_)
      return;
    int64_t split_idx = tid / num_keys_;
    int64_t key_idx = tid % num_keys_;

    uint64_t seed = input_[key_idx * 2];
    uint64_t offset = input_[key_idx * 2 + 1];

    auto r = philox_4x32(seed, offset + static_cast<uint64_t>(split_idx));
    int64_t out = (split_idx * num_keys_ + key_idx) * 2;
    philox_derive_key(r, &output_[out], &output_[out + 1]);
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
    int64_t idx = static_cast<int64_t>(item.get_global_id(0));
    if (idx >= num_keys_)
      return;

    uint64_t seed = input_[idx * 2];
    uint64_t offset = input_[idx * 2 + 1];

    auto r = philox_4x32(seed, offset + static_cast<uint64_t>(data_));
    philox_derive_key(r, &output_[idx * 2], &output_[idx * 2 + 1]);
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

  const int64_t total_work_items = num_keys * num_splits;
  constexpr int64_t work_group_size = 256;
  const int64_t work_group_num =
      (total_work_items + work_group_size - 1) / work_group_size;
  auto key_contig = key.contiguous();
  auto functor = PhiloxKeySplitFunctor(
      key_contig.data_ptr<uint64_t>(),
      output.data_ptr<uint64_t>(),
      num_keys,
      total_work_items);

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

  constexpr int64_t work_group_size = 256;
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
