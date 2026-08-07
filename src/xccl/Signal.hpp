#pragma once

#include <atomic>

#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <comm/SYCLContext.h>

namespace c10d::symmetric_memory {

using at::native::memory::get_alignment;

// =============================================================================
// Signal primitives using store/load + atomic_fence
// (sycl::atomic_ref is not supported, use explicit fence instead)
// =============================================================================
//
// Note on memory scope:
// We intentionally use memory_scope::system because signal pads are exchanged
// across ranks/devices (including peer/device-visible IPC mappings). These
// flags are polled and updated by kernels running on different devices, so a
// device/work-group scope is too narrow for this protocol.

// Store value with release fence (for put_signal)
// Order: store first, then release fence to flush the store
inline void store_release(uint32_t* addr, uint32_t val) {
  *addr = val;
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
}

// Load value with acquire fence (for get_signal/wait_signal)
// Order: acquire fence first, then load to see the latest value
inline uint32_t load_acquire(uint32_t* addr) {
  sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
  uint32_t val = *addr;
  return val;
}

// =============================================================================
// Put signal: wait until addr == 0, then set to 1 (release semantics)
// =============================================================================

inline bool try_put_signal_device(uint32_t* addr, size_t timeout_ms) {
  // Wait until the slot is free (value == 0)
  while (load_acquire(addr) != 0) {
    // Spin wait (no timeout check as IGC issue)
    continue;
  }
  // Set signal to 1 with release semantics
  store_release(addr, 1);
  return true;
}

// =============================================================================
// Wait signal: wait until addr == 1, then set to 0 (acquire semantics)
// =============================================================================
inline bool try_wait_signal_device(uint32_t* addr, size_t timeout_ms) {
  // Wait until signal is set (value == 1)
  while (load_acquire(addr) != 1) {
    // Spin wait (no timeout check as IGC issue)
    continue;
  }
  // Clear signal to 0 with release semantics
  store_release(addr, 0);
  return true;
}

void barrier_impl_xpu(
    uint32_t** signal_pads,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    at::xpu::XPUStream& stream);

void put_signal_impl_xpu(
    uint32_t** signal_pads,
    int dst_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    at::xpu::XPUStream& stream);

void wait_signal_impl_xpu(
    uint32_t** signal_pads,
    int src_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    at::xpu::XPUStream& stream);
} // namespace c10d::symmetric_memory
