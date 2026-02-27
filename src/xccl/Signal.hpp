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

inline size_t global_timer_ns() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             now.time_since_epoch())
      .count();
}

constexpr size_t ns_per_ms = 1e6;

// =============================================================================
// Put signal: wait until addr == 0, then set to 1 (release semantics)
// =============================================================================

// Device-compatible version using iteration count
template <std::memory_order Sem>
bool try_put_signal_device(uint32_t* addr, size_t max_iterations = 1000) {
  size_t iterations = 0;
  // Wait until the slot is free (value == 0)
  while (load_acquire(addr) != 0) {
    if (max_iterations != 0 && iterations++ > max_iterations) {
      return false;
    }
  }
  // Set signal to 1 with release semantics
  store_release(addr, 1);
  return true;
}

// Host version using timeout
template <std::memory_order Sem>
bool try_put_signal(uint32_t* addr, size_t timeout_ms) {
  size_t deadline = global_timer_ns() + timeout_ms * ns_per_ms;
  // Wait until the slot is free (value == 0)
  while (load_acquire(addr) != 0) {
    if (timeout_ms != 0 && global_timer_ns() > deadline) {
      return false;
    }
  }
  // Set signal to 1 with release semantics
  store_release(addr, 1);
  return true;
}

// Blocking version
template <std::memory_order Sem>
void put_signal(uint32_t* addr) {
  // Wait until the slot is free (value == 0)
  while (load_acquire(addr) != 0)
    ;
  // Set signal to 1 with release semantics
  store_release(addr, 1);
}

// =============================================================================
// Wait signal: wait until addr == 1, then set to 0 (acquire semantics)
// =============================================================================

// Device-compatible version using iteration count
template <std::memory_order Sem>
bool try_wait_signal_device(uint32_t* addr, size_t max_iterations = 1000) {
  size_t iterations = 0;
  // Wait until signal is set (value == 1)
  while (load_acquire(addr) != 1) {
    // Spin wait (no timeout check to avoid early exit)
    continue;
  }
  // Clear signal to 0 with release semantics
  store_release(addr, 0);
  return true;
}

// Host version using timeout
template <std::memory_order Sem>
bool try_wait_signal(uint32_t* addr, size_t timeout_ms) {
  size_t deadline = global_timer_ns() + timeout_ms * ns_per_ms;
  // Wait until signal is set (value == 1)
  while (load_acquire(addr) != 1) {
    // Spin wait (no timeout check to avoid early exit)
    continue;
  }
  // Clear signal to 0 with release semantics
  store_release(addr, 0);
  return true;
}

// Blocking version
template <std::memory_order Sem>
void wait_signal(uint32_t* addr) {
  // Wait until signal is set (value == 1)
  while (load_acquire(addr) != 1)
    ;
  // Clear signal to 0 with release semantics
  store_release(addr, 0);
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
