#pragma once
#include <sycl/sycl.hpp>

namespace c10d::symmetric_memory {

// XPU-specific atomic operations using SYCL
template <sycl::memory_order MemOrder>
inline uint32_t xpu_atomic_cas(uint32_t* addr, uint32_t compare, uint32_t val) {
  auto atomic_ref =
      sycl::atomic_ref<uint32_t, MemOrder, sycl::memory_scope::system>(*addr);
  atomic_ref.compare_exchange_strong(compare, val);
  return compare;
}

template <sycl::memory_order MemOrder>
inline bool xpu_try_put_signal(uint32_t* addr, size_t timeout_ms) {
  // Simple implementation - in practice, timeout should be handled differently
  while (xpu_atomic_cas<MemOrder>(addr, 0, 1) != 0) {
    // Busy wait - could implement proper timeout handling
  }
  return true;
}

template <sycl::memory_order MemOrder>
inline bool xpu_try_wait_signal(uint32_t* addr, size_t timeout_ms) {
  // Simple implementation - in practice, timeout should be handled differently
  while (xpu_atomic_cas<MemOrder>(addr, 1, 0) != 1) {
    // Busy wait - could implement proper timeout handling
  }
  return true;
}

template <sycl::memory_order MemOrder>
inline void xpu_put_signal(uint32_t* addr) {
  auto atomic_ref =
      sycl::atomic_ref<uint32_t, MemOrder, sycl::memory_scope::system>(*addr);
  atomic_ref.store(1);
}

template <sycl::memory_order MemOrder>
inline void xpu_wait_signal(uint32_t* addr) {
  auto atomic_ref =
      sycl::atomic_ref<uint32_t, MemOrder, sycl::memory_scope::system>(*addr);
  while (atomic_ref.load() != 1) {
    // Busy wait
  }
  atomic_ref.store(0); // Reset for next use
}

} // namespace c10d::symmetric_memory
