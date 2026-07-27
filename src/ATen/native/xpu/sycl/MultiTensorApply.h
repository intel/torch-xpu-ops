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
#include <c10/core/ScalarType.h>
#include <comm/xpu_aten.h>
#include <vector>

#include <ATen/core/CachingHostAllocator.h>
#include <ATen/native/xpu/sycl/GroupReduceUtils.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <comm/SYCLContext.h>

#ifdef _WIN32
#include <tuple>
#include <utility>
#endif

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

static int64_t multi_tensor_apply_kernel_get_wg_size(int simd) {
  return get_group_reduce_group_size(simd);
}

static int64_t multi_tensor_apply_kernel_get_chunk_size(int simd) {
  int64_t max_wg_size = multi_tensor_apply_kernel_get_wg_size(simd);
  return max_wg_size * kElementPerThread;
}

static inline int64_t multi_tensor_apply_fused_kernel_get_wg_size() {
  return syclMaxWorkItemsPerSubSlice();
}

static inline int64_t multi_tensor_apply_fused_kernel_get_chunk_size() {
  int64_t max_wg_size = multi_tensor_apply_fused_kernel_get_wg_size();
  return max_wg_size * kILP;
}

#ifdef _WIN32
// Windows-only workaround for an icx SYCL integration-header bug with MSVC as
// host compiler: a variadic free-function kernel makes icx emit a KernelInfo
// specialization whose non-type template argument (the kernel pointer) MSVC
// cannot convert (error C2440). Fold the extra args into a single non-variadic,
// device-copyable functor so the emitted template-id is non-variadic.
// TODO: remove once the icx integration-header generator is fixed
// [CMPLRLLVM-77173]
template <typename T, typename Y, typename U>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void multi_tensor_apply_kernel(
    int64_t kChunkSize,
    T tlAddressMeta,
    Y tlWGMeta,
    U callable) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  callable(kChunkSize, tlAddressMeta, tlWGMeta, item);
}

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
#else
template <typename T, typename Y, typename U, typename... ArgTypes>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void multi_tensor_apply_kernel(
    int64_t kChunkSize,
    T tlAddressMeta,
    Y tlWGMeta,
    U callable,
    ArgTypes... args) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  callable(kChunkSize, tlAddressMeta, tlWGMeta, item, args...);
}
#endif // _WIN32

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
  auto& q = getCurrentSYCLQueue();
  int64_t simd = syclMaxSubGroupSize();
  int64_t max_wg_size = multi_tensor_apply_kernel_get_wg_size(simd);
  int64_t kChunkSize = multi_tensor_apply_kernel_get_chunk_size(simd);

  if constexpr (fused_kernel) {
    max_wg_size = multi_tensor_apply_fused_kernel_get_wg_size();
    kChunkSize = multi_tensor_apply_fused_kernel_get_chunk_size();
  }

#ifdef _WIN32
  // See MultiTensorApplyCallableWrapper above: fold extra args into a
  // non-variadic, device-copyable callable so the kernel template-id emitted in
  // the SYCL integration header is non-variadic (works around MSVC C2440).
  using WrappedCallable = MultiTensorApplyCallableWrapper<U, ArgTypes...>;
  WrappedCallable wrapped{callable, std::tuple<ArgTypes...>(args...)};

  constexpr auto kfn = multi_tensor_apply_kernel<T, Y, WrappedCallable>;

  sycl_kernel_submit<kfn>(
      sycl::range<1>(num_wg * max_wg_size),
      sycl::range<1>(max_wg_size),
      q,
      0,
      kChunkSize,
      tlAddressMeta,
      tlWGMeta,
      wrapped);
#else
  constexpr auto kfn = multi_tensor_apply_kernel<T, Y, U, ArgTypes...>;

  sycl_kernel_submit<kfn>(
      sycl::range<1>(num_wg * max_wg_size),
      sycl::range<1>(max_wg_size),
      q,
      0,
      kChunkSize,
      tlAddressMeta,
      tlWGMeta,
      callable,
      args...);
#endif // _WIN32
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

  auto& q = getCurrentSYCLQueue();
  int64_t simd = syclMaxSubGroupSize();
  int64_t kChunkSize = multi_tensor_apply_kernel_get_chunk_size(simd);

  auto addressStorage = at::empty(
      {(int)(sizeof(TLMetaForAddressScalar<scalar_vals_t, depth>) * n_tensors)},
      tensor_lists[0][0].options().dtype(at::kByte));
  auto metaAddressInput =
      static_cast<TLMetaForAddressScalar<scalar_vals_t, depth>*>(
          addressStorage.mutable_data_ptr());
  TLMetaForAddressScalar<scalar_vals_t, depth>* tlAddress = nullptr;

  auto tlAddress_dptr = at::getHostAllocator(at::kXPU)->allocate(
      sizeof(TLMetaForAddressScalar<scalar_vals_t, depth>) * n_tensors);
  tlAddress =
      (TLMetaForAddressScalar<scalar_vals_t, depth>*)tlAddress_dptr.get();
  uint64_t totalWG = 0;

  // this loop record all the tensor address and numel info.
  for (size_t t = 0; t < n_tensors; ++t) {
    auto numel = tensor_lists[0][t].numel();
    tlAddress[t].numel_to_tensor = numel;
    tlAddress[t].scalar_vals = scalars[t].to<scalar_t>();
    totalWG += (numel + kChunkSize - 1) / kChunkSize;
    for (int d = 0; d < depth; ++d) {
      tlAddress[t].addresses[d] = tensor_lists[d][t].mutable_data_ptr();
    }
  }

  q.memcpy(
      (void*)metaAddressInput,
      (void*)tlAddress,
      sizeof(TLMetaForAddressScalar<scalar_vals_t, depth>) * n_tensors);
  at::getHostAllocator(at::kXPU)->record_event(
      (void*)tlAddress,
      tlAddress_dptr.get_context(),
      at::xpu::getCurrentXPUStream());

  auto wgMetaStorage = at::empty(
      {(int)(sizeof(TLMetaForWG) * totalWG)},
      tensor_lists[0][0].options().dtype(at::kByte));
  auto metaWGInput =
      static_cast<TLMetaForWG*>(wgMetaStorage.mutable_data_ptr());
  TLMetaForWG* tlWGMeta = nullptr;

  auto tlWGMeta_dptr =
      at::getHostAllocator(at::kXPU)->allocate(sizeof(TLMetaForWG) * totalWG);
  tlWGMeta = (TLMetaForWG*)tlWGMeta_dptr.get();
  uint64_t posWG = 0;
  // this loop record the correspond tensor and chunk info for each work group.
  for (size_t t = 0; t < n_tensors; ++t) {
    auto numel = tensor_lists[0][t].numel();
    auto chunkForWG = (numel + kChunkSize - 1) / kChunkSize;
    for (size_t chunkId = 0; chunkId < chunkForWG; ++chunkId, ++posWG) {
      tlWGMeta[posWG].wg_to_tensor = t;
      tlWGMeta[posWG].wg_to_chunk = chunkId;
    }
  }
  TORCH_CHECK(
      posWG == totalWG,
      "Work group index dose not equal to the allocated memory size, segment fault might occur");

  q.memcpy((void*)metaWGInput, (void*)tlWGMeta, sizeof(TLMetaForWG) * totalWG);
  at::getHostAllocator(at::kXPU)->record_event(
      (void*)tlWGMeta,
      tlWGMeta_dptr.get_context(),
      at::xpu::getCurrentXPUStream());

  launch_multi_tensor_apply_kernel<false>(
      metaAddressInput, metaWGInput, callable, totalWG, args...);
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

  auto& q = getCurrentSYCLQueue();
  int64_t simd = syclMaxSubGroupSize();
  int64_t kChunkSize = multi_tensor_apply_kernel_get_chunk_size(simd);

  auto addressStorage = at::empty(
      {(int)(sizeof(TLMetaForAddress<depth>) * n_tensors)},
      tensor_lists[0][0].options().dtype(at::kByte));
  auto metaAddressInput =
      static_cast<TLMetaForAddress<depth>*>(addressStorage.mutable_data_ptr());
  TLMetaForAddress<depth>* tlAddress = nullptr;

  auto tlAddress_dptr = at::getHostAllocator(at::kXPU)->allocate(
      sizeof(TLMetaForAddress<depth>) * n_tensors);
  tlAddress = (TLMetaForAddress<depth>*)tlAddress_dptr.get();
  uint64_t totalWG = 0;

  // this loop record all the tensor address and numel info.
  for (size_t t = 0; t < n_tensors; ++t) {
    auto numel = tensor_lists[0][t].numel();
    tlAddress[t].numel_to_tensor = numel;
    totalWG += (numel + kChunkSize - 1) / kChunkSize;
    for (int d = 0; d < depth; ++d) {
      tlAddress[t].addresses[d] = tensor_lists[d][t].mutable_data_ptr();
    }
  }

  q.memcpy(
      (void*)metaAddressInput,
      (void*)tlAddress,
      sizeof(TLMetaForAddress<depth>) * n_tensors);
  at::getHostAllocator(at::kXPU)->record_event(
      (void*)tlAddress,
      tlAddress_dptr.get_context(),
      at::xpu::getCurrentXPUStream());

  auto wgMetaStorage = at::empty(
      {(int)(sizeof(TLMetaForWG) * totalWG)},
      tensor_lists[0][0].options().dtype(at::kByte));
  auto metaWGInput =
      static_cast<TLMetaForWG*>(wgMetaStorage.mutable_data_ptr());
  TLMetaForWG* tlWGMeta = nullptr;

  auto tlWGMeta_dptr =
      at::getHostAllocator(at::kXPU)->allocate(sizeof(TLMetaForWG) * totalWG);
  tlWGMeta = (TLMetaForWG*)tlWGMeta_dptr.get();
  uint64_t posWG = 0;
  // this loop record the correspond tensor and chunk info for each work group.
  for (size_t t = 0; t < n_tensors; ++t) {
    auto numel = tensor_lists[0][t].numel();
    auto chunkForWG = (numel + kChunkSize - 1) / kChunkSize;
    for (size_t chunkId = 0; chunkId < chunkForWG; ++chunkId, ++posWG) {
      tlWGMeta[posWG].wg_to_tensor = t;
      tlWGMeta[posWG].wg_to_chunk = chunkId;
    }
  }
  TORCH_CHECK(
      posWG == totalWG,
      "Work group index dose not equal to the allocated memory size, segment fault might occur");

  q.memcpy((void*)metaWGInput, (void*)tlWGMeta, sizeof(TLMetaForWG) * totalWG);
  at::getHostAllocator(at::kXPU)->record_event(
      (void*)tlWGMeta,
      tlWGMeta_dptr.get_context(),
      at::xpu::getCurrentXPUStream());

  launch_multi_tensor_apply_kernel<false>(
      metaAddressInput, metaWGInput, callable, totalWG, args...);
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

  auto& q = getCurrentSYCLQueue();
  int64_t kChunkSize = multi_tensor_apply_fused_kernel_get_chunk_size();

  auto addressStorage = at::empty(
      {(int)(sizeof(TLFusedMetaForAddress<depth>) * n_tensors)},
      tensor_lists[0][0].options().dtype(at::kByte));
  auto metaFusedAddressInput = static_cast<TLFusedMetaForAddress<depth>*>(
      addressStorage.mutable_data_ptr());
  TLFusedMetaForAddress<depth>* tlAddress = nullptr;

  auto tlAddress_dptr = at::getHostAllocator(at::kXPU)->allocate(
      sizeof(TLFusedMetaForAddress<depth>) * n_tensors);
  tlAddress = (TLFusedMetaForAddress<depth>*)tlAddress_dptr.get();
  uint64_t totalWG = 0;

  // this loop record all the tensor address and numel info.
  for (size_t t = 0; t < n_tensors; ++t) {
    auto numel = tensor_lists[0][t].numel();
    tlAddress[t].numel_to_tensor = numel;
    tlAddress[t].state_steps_addresses = state_steps[t].mutable_data_ptr();
    totalWG += (numel + kChunkSize - 1) / kChunkSize;
    for (int d = 0; d < depth; ++d) {
      tlAddress[t].addresses[d] = tensor_lists[d][t].mutable_data_ptr();
    }
  }

  q.memcpy(
      (void*)metaFusedAddressInput,
      (void*)tlAddress,
      sizeof(TLFusedMetaForAddress<depth>) * n_tensors);
  at::getHostAllocator(at::kXPU)->record_event(
      (void*)tlAddress,
      tlAddress_dptr.get_context(),
      at::xpu::getCurrentXPUStream());

  auto wgMetaStorage = at::empty(
      {(int)(sizeof(TLMetaForWG) * totalWG)},
      tensor_lists[0][0].options().dtype(at::kByte));
  auto metaWGInput =
      static_cast<TLMetaForWG*>(wgMetaStorage.mutable_data_ptr());
  TLMetaForWG* tlWGMeta = nullptr;

  auto tlWGMeta_dptr =
      at::getHostAllocator(at::kXPU)->allocate(sizeof(TLMetaForWG) * totalWG);
  tlWGMeta = (TLMetaForWG*)tlWGMeta_dptr.get();
  uint64_t posWG = 0;
  // this loop record the correspond tensor and chunk info for each work group.
  for (size_t t = 0; t < n_tensors; ++t) {
    auto numel = tensor_lists[0][t].numel();
    auto chunkForWG = (numel + kChunkSize - 1) / kChunkSize;
    for (size_t chunkId = 0; chunkId < chunkForWG; ++chunkId, ++posWG) {
      tlWGMeta[posWG].wg_to_tensor = t;
      tlWGMeta[posWG].wg_to_chunk = chunkId;
    }
  }
  TORCH_CHECK(
      posWG == totalWG,
      "Work group index dose not equal to the allocated memory size, segment fault might occur");

  q.memcpy((void*)metaWGInput, (void*)tlWGMeta, sizeof(TLMetaForWG) * totalWG);
  at::getHostAllocator(at::kXPU)->record_event(
      (void*)tlWGMeta,
      tlWGMeta_dptr.get_context(),
      at::xpu::getCurrentXPUStream());

  launch_multi_tensor_apply_kernel<true>(
      metaFusedAddressInput, metaWGInput, callable, totalWG, args...);
}

} // namespace at::native::xpu

#ifdef _WIN32
// Opt MultiTensorApplyCallableWrapper into SYCL device-copyability. std::tuple
// is not implicitly device-copyable, so declare the wrapper copyable whenever
// its callable and bound argument types are.
// TODO: remove once the icx integration-header generator is fixed
// [CMPLRLLVM-77173]
template <typename U, typename... ArgTypes>
struct sycl::is_device_copyable<
    ::at::native::xpu::MultiTensorApplyCallableWrapper<U, ArgTypes...>>
    : std::bool_constant<(
          sycl::is_device_copyable_v<U> && ... &&
          sycl::is_device_copyable_v<ArgTypes>)> {};
#endif // _WIN32
