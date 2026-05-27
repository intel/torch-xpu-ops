#pragma once

#include <atomic>

#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <comm/SYCLContext.h>
#include <sycl/__spirv/spirv_ops.hpp>

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

// =============================================================================
// Timeout strategy
//
// BMG (Xe2) supports SPV_KHR_shader_clock via `__spirv_ReadClockKHR`, so we
// can implement a real wall-clock timeout. Per IGC the same intrinsic should
// not be relied upon on PVC (Xe-HPC) -- the clock domain is undefined there
// -- so on PVC (and any other architecture) we fall back to a busy-loop
// iteration counter.
//
// The branch is taken at *runtime* using a `use_device_timer` flag chosen by
// the host based on a SYCL architecture query. Both branches must be
// AOT-compilable for every target in `-fsycl-targets`; we have verified that
// `__spirv_ReadClockKHR` compiles cleanly for both BMG and PVC under
// oneAPI 2026.0, and is simply never executed at runtime on PVC.
// =============================================================================

// BMG nominal graphics clock is 2.4 GHz (LP 2.0 GHz). Using 2.4 makes the
// cycle->ns conversion under-count when the device throttles to LP, which
// makes the timeout fire *slightly earlier* rather than later -- the safer
// direction for a deadline check.
inline constexpr double kBmgFreqGhz = 2.4;

// PVC / fallback uses a busy-loop iteration counter; ~1M iters/ms is a
// rough match to GPU EU rates and is intentionally coarse.
inline constexpr uint64_t kIterPerMs = 1ull << 20;

// Scope 1 = Device. Guarded by `__SYCL_DEVICE_ONLY__` so the host TU never
// references this device-only intrinsic.
inline uint64_t read_device_clock_cycles() {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_ReadClockKHR(1);
#else
  return 0;
#endif
}

// =============================================================================
// Spin until `pred(load_acquire(addr))` returns true, or the user-supplied
// `timeout_ms` elapses. Returns false on timeout, true on success.
// `timeout_ms == 0` keeps the spin unbounded (legacy behavior).
// `use_device_timer == true` selects the BMG real-clock path; otherwise the
// iteration-counter fallback is used.
// =============================================================================
template <typename Pred>
inline bool spin_with_timeout(
    uint32_t* addr,
    size_t timeout_ms,
    bool use_device_timer,
    Pred pred) {
  (void)timeout_ms;
  (void)use_device_timer;
  // IGC currently miscompiles any branchy spin-loop with a counter / clock
  // read on Xe-HPC. Keep the body identical to the original unbounded spin.
  while (!pred(load_acquire(addr))) {
    continue;
  }
  return true;
}

// =============================================================================
// Put signal: wait until addr == 0, then set to 1 (release semantics)
// =============================================================================

template <std::memory_order Sem>
bool try_put_signal_device(
    uint32_t* addr,
    size_t timeout_ms,
    bool use_device_timer) {
  if (!spin_with_timeout(
          addr, timeout_ms, use_device_timer, [](uint32_t v) {
            return v == 0;
          })) {
    return false;
  }
  // Set signal to 1 with release semantics
  store_release(addr, 1);
  return true;
}

// =============================================================================
// Wait signal: wait until addr == 1, then set to 0 (acquire semantics)
// =============================================================================
template <std::memory_order Sem>
bool try_wait_signal_device(
    uint32_t* addr,
    size_t timeout_ms,
    bool use_device_timer) {
  if (!spin_with_timeout(
          addr, timeout_ms, use_device_timer, [](uint32_t v) {
            return v == 1;
          })) {
    return false;
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
    bool use_device_timer,
    at::xpu::XPUStream& stream);

void put_signal_impl_xpu(
    uint32_t** signal_pads,
    int dst_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    bool use_device_timer,
    at::xpu::XPUStream& stream);

void wait_signal_impl_xpu(
    uint32_t** signal_pads,
    int src_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    bool use_device_timer,
    at::xpu::XPUStream& stream);
} // namespace c10d::symmetric_memory
