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
#include <ATen/core/Tensor.h>
#include <ATen/detail/XPUHooksInterface.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <ATen/xpu/CachingHostAllocator.h>
#include <c10/core/ScalarType.h>
#include <comm/SYCLContext.h>

using namespace at;

namespace xpu::sycl {

// Synchronous memory copy via the current SYCL queue. At least one of `dst` or
// `src` must reside on the device. If both are on device, they must be on the
// same device or P2P access must be enabled.
inline void memcpyAndSync(void* dst, const void* src, size_t n_bytes) {
  if (n_bytes == 0) {
    return;
  }

  getCurrentXPUStream().queue().memcpy(dst, src, n_bytes).wait();
}

// Asynchronous memory copy via the current SYCL queue. Same placement rules
// as memcpyAndSync. Caller must ensure both `dst` and `src` remain valid until
// the copy completes (e.g. via event recording or pinned memory).
inline void memcpyAsync(void* dst, const void* src, size_t n_bytes) {
  if (n_bytes == 0) {
    return;
  }

  getCurrentXPUStream().queue().memcpy(dst, src, n_bytes);
}

// Asynchronous host-to-device copy for known-pinned `src`. Skips the
// `isPinnedPtr` check, suitable for hot paths where `src` is guaranteed pinned.
// `hctx` is the allocator context associated with `src`, used for event
// recording.
inline void memcpyPinnedHostToDeviceAsync(
    void* dst,
    const void* src,
    size_t n_bytes,
    const void* hctx) {
  if (n_bytes == 0) {
    return;
  }

  memcpyAsync(dst, src, n_bytes);
  at::getHostAllocator(at::kXPU)->record_event(
      const_cast<void*>(src), const_cast<void*>(hctx), getCurrentXPUStream());
}

// Asynchronous host-to-device memory copy. For pinned `src`, record an event
// directly. For non-pinned `src`, stage through pinned memory so the source
// remains valid until the copy completes. `hctx` is the allocator context
// associated with `src`, used for event recording.
inline void memcpyHostToDeviceAsync(
    void* dst,
    const void* src,
    size_t n_bytes,
    const void* hctx) {
  if (n_bytes == 0) {
    return;
  }

  if (at::detail::getXPUHooks().isPinnedPtr(src)) {
    memcpyPinnedHostToDeviceAsync(dst, src, n_bytes, hctx);
    return;
  }

  // Non-pinned host memory may be freed before the async copy completes.
  // Stage through pinned memory to keep the data alive.
  auto stage_mem_dptr = at::getHostAllocator(at::kXPU)->allocate(n_bytes);
  void* stage_mem = stage_mem_dptr.get();
  TORCH_CHECK(stage_mem, "Fail to allocate host memory from XPU HostAllocator");
  std::memcpy(stage_mem, src, n_bytes);
  memcpyAsync(dst, stage_mem, n_bytes);
  at::getHostAllocator(at::kXPU)->record_event(
      stage_mem, stage_mem_dptr.get_context(), getCurrentXPUStream());
}

// Asynchronous device-to-host memory copy. Record an event if `dst` is pinned
// to ensure the memory won't be reused until the copy completes. `hctx` is the
// allocator context associated with `dst`, used for event recording.
inline void memcpyDeviceToHostAsync(
    void* dst,
    const void* src,
    size_t n_bytes,
    const void* hctx) {
  if (n_bytes == 0) {
    return;
  }

  memcpyAsync(dst, src, n_bytes);
  if (at::detail::getXPUHooks().isPinnedPtr(dst)) {
    at::getHostAllocator(at::kXPU)->record_event(
        dst, const_cast<void*>(hctx), getCurrentXPUStream());
  }
}

// Synchronous memset on device memory. Caller must ensure `dst` is on the
// current device.
inline void memsetAndSync(void* dst, int value, size_t n_bytes) {
  if (n_bytes == 0) {
    return;
  }

  getCurrentXPUStream().queue().memset(dst, value, n_bytes).wait();
}

// Asynchronous memset on device memory. Caller must ensure `dst` is on the
// current device.
inline void memsetAsync(void* dst, int value, size_t n_bytes) {
  if (n_bytes == 0) {
    return;
  }

  getCurrentXPUStream().queue().memset(dst, value, n_bytes);
}

} // namespace xpu::sycl
