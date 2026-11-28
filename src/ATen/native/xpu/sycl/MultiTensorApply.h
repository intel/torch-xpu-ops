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
#include <ATen/ceil_div.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <comm/xpu_aten.h>
#include <vector>

#include <ATen/core/CachingHostAllocator.h>
#include <ATen/native/xpu/sycl/GroupReduceUtils.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <comm/SYCLContext.h>

#include <tuple>
#include <utility>

namespace at::native::xpu {

// Instruction level Parallelism, namely vec size
static constexpr int64_t kILP = 4;
static constexpr int64_t kElementPerThread = 128;
template <typename T>
bool is_aligned(T* p) {
  return ((uint64_t)p) % (kILP * sizeof(T)) == 0;
}

template <typename T, int64_t kILP = kILP>
void load_store(T* dst, T* src, int dst_offset, int src_offset) {
  using LT = at::native::memory::aligned_vector<T, kILP>;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

template <typename scalar_vals_t, int n>
struct TLMetaForAddressScalar {
  void* addresses[n];
  uint32_t numel_to_tensor;
  scalar_vals_t scalar_vals;
};

template <int n>
struct TLMetaForAddress {
  void* addresses[n];
  uint32_t numel_to_tensor;
};

template <int n>
struct TLFusedMetaForAddress {
  void* addresses[n];
  uint32_t numel_to_tensor;
  void* state_steps_addresses;
};

struct TLMetaForWG {
  uint32_t wg_to_tensor;
  uint32_t wg_to_chunk;
};

// Packs the per-tensor address table and the per-work-group table into one
// allocation per memory space. In both buffers, the work-group table starts at
// offset addr_bytes_.
template <typename address_meta_t>
struct MultiTensorApplyMetadata {
  static_assert(
      sizeof(address_meta_t) % alignof(TLMetaForWG) == 0,
      "the work-group table must stay aligned behind the address table");

  MultiTensorApplyMetadata(size_t n_tensors, uint64_t total_wg)
      : addr_bytes_(sizeof(address_meta_t) * n_tensors),
        wg_bytes_(sizeof(TLMetaForWG) * total_wg),
        device_(c10::GetAllocator(at::kXPU)->allocate(addr_bytes_ + wg_bytes_)),
        host_(
            at::getHostAllocator(at::kXPU)->allocate(addr_bytes_ + wg_bytes_)) {
  }

  address_meta_t* address_meta() {
    return static_cast<address_meta_t*>(device_.get());
  }

  TLMetaForWG* work_group_meta() {
    return reinterpret_cast<TLMetaForWG*>(
        static_cast<char*>(device_.get()) + addr_bytes_);
  }

  address_meta_t* host_address_meta() {
    return static_cast<address_meta_t*>(host_.get());
  }

  TLMetaForWG* host_work_group_meta() {
    return reinterpret_cast<TLMetaForWG*>(
        static_cast<char*>(host_.get()) + addr_bytes_);
  }

  uint64_t work_group_count() const {
    return wg_bytes_ / sizeof(TLMetaForWG);
  }

  void build_work_group_table(
      const std::vector<at::Tensor>& tensors,
      int64_t chunk_size) {
    auto* wg_meta = host_work_group_meta();
    uint64_t pos = 0;
    for (size_t t = 0; t < tensors.size(); ++t) {
      const auto chunk_for_wg = at::ceil_div(tensors[t].numel(), chunk_size);
      for (size_t chunk_id = 0; chunk_id < chunk_for_wg; ++chunk_id, ++pos) {
        wg_meta[pos].wg_to_tensor = t;
        wg_meta[pos].wg_to_chunk = chunk_id;
      }
    }
    TORCH_CHECK(
        pos == work_group_count(),
        "Work group index dose not equal to the allocated memory size, segment fault might occur");
  }

  // Copies the packed metadata to the device and keeps the pinned host block
  // alive until the copy completes.
  void copy_to_device() {
    getCurrentSYCLQueue().memcpy(
        device_.get(), host_.get(), addr_bytes_ + wg_bytes_);
    at::getHostAllocator(at::kXPU)->record_event(
        host_address_meta(),
        host_.get_context(),
        at::xpu::getCurrentXPUStream());
  }

 private:
  size_t addr_bytes_;
  size_t wg_bytes_;
  c10::DataPtr device_;
  c10::DataPtr host_;
};

inline uint64_t count_work_groups(
    const std::vector<at::Tensor>& tensors,
    int64_t chunk_size) {
  uint64_t total_wg = 0;
  for (size_t t = 0; t < tensors.size(); ++t) {
    total_wg += at::ceil_div(tensors[t].numel(), chunk_size);
  }
  return total_wg;
}

static int64_t multi_tensor_apply_kernel_get_wg_size(int simd) {
  return get_group_reduce_group_size(simd);
}

static int64_t multi_tensor_apply_kernel_get_chunk_size(int simd) {
  int64_t max_wg_size = multi_tensor_apply_kernel_get_wg_size(simd);
  return max_wg_size * kElementPerThread;
}

static inline int64_t multi_tensor_apply_fused_kernel_get_wg_size() {
  return at::xpu::getDeviceMaxSubGroupSize() *
      at::xpu::getDeviceEUCountPerXeCore();
}

static inline int64_t multi_tensor_apply_fused_kernel_get_chunk_size() {
  int64_t max_wg_size = multi_tensor_apply_fused_kernel_get_wg_size();
  return max_wg_size * kILP;
}

// functions marked with `nd_range_kernel` cannot be variadic.
// Functors that reduce over a fixed sub-group width specialize this trait.
template <typename T, typename = void>
struct CallableSubGroupSize : std::integral_constant<int, -1> {};

template <typename T, typename Y, typename U>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void multi_tensor_apply_kernel(
    int64_t kChunkSize,
    T tlAddressMeta,
    Y tlWGMeta,
    U callable) {
  static_assert(
      CallableSubGroupSize<U>::value == -1,
      "callables with a fixed sub-group width must use the pinned kernel");
  auto item = syclext::this_work_item::get_nd_item<1>();
  callable(kChunkSize, tlAddressMeta, tlWGMeta, item);
}

template <int SG_SIZE, typename T, typename Y, typename U>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::sub_group_size<SG_SIZE>))
void multi_tensor_apply_kernel_with_sub_group_size(
    int64_t kChunkSize,
    T tlAddressMeta,
    Y tlWGMeta,
    U callable) {
  static_assert(
      SG_SIZE == CallableSubGroupSize<U>::value,
      "launched sub-group width must match the callable's sub-group width");
  auto item = syclext::this_work_item::get_nd_item<1>();
  callable(kChunkSize, tlAddressMeta, tlWGMeta, item);
}

// SG_SIZE == -1 keeps the default kernel for callables that are not templated
// on the sub-group width.
template <int SG_SIZE, typename T, typename Y, typename U>
constexpr auto multi_tensor_apply_kernel_for() {
  if constexpr (SG_SIZE == -1) {
    return multi_tensor_apply_kernel<T, Y, U>;
  } else {
    return multi_tensor_apply_kernel_with_sub_group_size<SG_SIZE, T, Y, U>;
  }
}

// For the variadic kernel template, fold the extra args into
// a single non-variadic, device-copyable functor
template <typename U, typename... ArgTypes>
struct MultiTensorApplyCallableWrapper {
  U callable;
  std::tuple<ArgTypes...> args;

  template <typename T, typename Y, typename Item>
  void operator()(int64_t kChunkSize, T tlAddressMeta, Y tlWGMeta, Item item)
      const {
    std::apply(
        [&](const ArgTypes&... unpacked) {
          callable(kChunkSize, tlAddressMeta, tlWGMeta, item, unpacked...);
        },
        args);
  }
};

template <typename U, typename... ArgTypes>
struct CallableSubGroupSize<
    MultiTensorApplyCallableWrapper<U, ArgTypes...>,
    void> : CallableSubGroupSize<U> {};

template <
    bool fused_kernel,
    typename T,
    typename Y,
    typename U,
    typename... ArgTypes>
void launch_multi_tensor_apply_kernel(
    T tlAddressMeta,
    Y tlWGMeta,
    U callable,
    int num_wg,
    ArgTypes... args) {
  constexpr int SG_SIZE = CallableSubGroupSize<U>::value;
  static_assert(
      !(fused_kernel && SG_SIZE != -1),
      "fused optimizer kernels do not reduce over a fixed sub-group width");
  auto& q = getCurrentSYCLQueue();
  int64_t simd = SG_SIZE == -1 ? at::xpu::getDeviceMaxSubGroupSize() : SG_SIZE;
  int64_t max_wg_size = multi_tensor_apply_kernel_get_wg_size(simd);
  int64_t kChunkSize = multi_tensor_apply_kernel_get_chunk_size(simd);

  if constexpr (fused_kernel) {
    max_wg_size = multi_tensor_apply_fused_kernel_get_wg_size();
    kChunkSize = multi_tensor_apply_fused_kernel_get_chunk_size();
  }

  // See MultiTensorApplyCallableWrapper above: fold extra args into a
  // non-variadic, device-copyable callable
  using WrappedCallable = MultiTensorApplyCallableWrapper<U, ArgTypes...>;
  WrappedCallable wrapped{callable, std::tuple<ArgTypes...>(args...)};

  constexpr auto kfn =
      multi_tensor_apply_kernel_for<SG_SIZE, T, Y, WrappedCallable>();

  sycl_kernel_submit<kfn>(
      sycl::range<1>(num_wg * max_wg_size),
      sycl::range<1>(max_wg_size),
      q,
      0,
      kChunkSize,
      tlAddressMeta,
      tlWGMeta,
      wrapped);
}

template <int depth, typename scalar_t, typename T, typename... ArgTypes>
void multi_tensor_apply(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    at::ArrayRef<Scalar> scalars,
    T callable,
    ArgTypes... args) {
  using scalar_vals_t = typename T::opmath_t;
  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match he depth");
  size_t n_tensors = tensor_lists[0].size();

  constexpr int SG_SIZE = CallableSubGroupSize<T>::value;
  int64_t simd = SG_SIZE == -1 ? at::xpu::getDeviceMaxSubGroupSize() : SG_SIZE;
  int64_t kChunkSize = multi_tensor_apply_kernel_get_chunk_size(simd);

  MultiTensorApplyMetadata<TLMetaForAddressScalar<scalar_vals_t, depth>>
      metadata(n_tensors, count_work_groups(tensor_lists[0], kChunkSize));
  auto* tlAddress = metadata.host_address_meta();

  // this loop record all the tensor address and numel info.
  for (size_t t = 0; t < n_tensors; ++t) {
    auto numel = tensor_lists[0][t].numel();
    tlAddress[t].numel_to_tensor = numel;
    tlAddress[t].scalar_vals = scalars[t].to<scalar_t>();
    for (int d = 0; d < depth; ++d) {
      tlAddress[t].addresses[d] = tensor_lists[d][t].mutable_data_ptr();
    }
  }
  metadata.build_work_group_table(tensor_lists[0], kChunkSize);
  metadata.copy_to_device();

  launch_multi_tensor_apply_kernel<false>(
      metadata.address_meta(),
      metadata.work_group_meta(),
      callable,
      metadata.work_group_count(),
      args...);
}

template <int depth, typename T, typename... ArgTypes>
void multi_tensor_apply(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    T callable,
    ArgTypes... args) {
  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match he depth");
  size_t n_tensors = tensor_lists[0].size();

  constexpr int SG_SIZE = CallableSubGroupSize<T>::value;
  int64_t simd = SG_SIZE == -1 ? at::xpu::getDeviceMaxSubGroupSize() : SG_SIZE;
  int64_t kChunkSize = multi_tensor_apply_kernel_get_chunk_size(simd);

  MultiTensorApplyMetadata<TLMetaForAddress<depth>> metadata(
      n_tensors, count_work_groups(tensor_lists[0], kChunkSize));
  auto* tlAddress = metadata.host_address_meta();

  // this loop record all the tensor address and numel info.
  for (size_t t = 0; t < n_tensors; ++t) {
    auto numel = tensor_lists[0][t].numel();
    tlAddress[t].numel_to_tensor = numel;
    for (int d = 0; d < depth; ++d) {
      tlAddress[t].addresses[d] = tensor_lists[d][t].mutable_data_ptr();
    }
  }
  metadata.build_work_group_table(tensor_lists[0], kChunkSize);
  metadata.copy_to_device();

  launch_multi_tensor_apply_kernel<false>(
      metadata.address_meta(),
      metadata.work_group_meta(),
      callable,
      metadata.work_group_count(),
      args...);
}

template <int depth, typename T, typename... ArgTypes>
void multi_tensor_apply_for_fused_optimizer(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    at::TensorList state_steps,
    T callable,
    ArgTypes... args) {
  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match the depth");
  const auto n_tensors = tensor_lists[0].size();

  int64_t kChunkSize = multi_tensor_apply_fused_kernel_get_chunk_size();

  MultiTensorApplyMetadata<TLFusedMetaForAddress<depth>> metadata(
      n_tensors, count_work_groups(tensor_lists[0], kChunkSize));
  auto* tlAddress = metadata.host_address_meta();

  // this loop record all the tensor address and numel info.
  for (size_t t = 0; t < n_tensors; ++t) {
    auto numel = tensor_lists[0][t].numel();
    tlAddress[t].numel_to_tensor = numel;
    tlAddress[t].state_steps_addresses = state_steps[t].mutable_data_ptr();
    for (int d = 0; d < depth; ++d) {
      tlAddress[t].addresses[d] = tensor_lists[d][t].mutable_data_ptr();
    }
  }
  metadata.build_work_group_table(tensor_lists[0], kChunkSize);
  metadata.copy_to_device();

  launch_multi_tensor_apply_kernel<true>(
      metadata.address_meta(),
      metadata.work_group_meta(),
      callable,
      metadata.work_group_count(),
      args...);
}

} // namespace at::native::xpu

// Opt MultiTensorApplyCallableWrapper into SYCL device-copyability. std::tuple
// is not implicitly device-copyable, so declare the wrapper copyable whenever
// its callable and bound argument types are.
template <typename U, typename... ArgTypes>
struct sycl::is_device_copyable<
    ::at::native::xpu::MultiTensorApplyCallableWrapper<U, ArgTypes...>>
    : std::bool_constant<(
          sycl::is_device_copyable_v<U> && ... &&
          sycl::is_device_copyable_v<ArgTypes>)> {};
