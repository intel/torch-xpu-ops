#pragma once

#include <atomic>

#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <comm/SYCLContext.h>

namespace c10d::symmetric_memory {

using at::native::memory::get_alignment;

// =============================================================================
// Signal primitives using LSC load/store + atomic_fence
// (sycl::atomic_ref is not supported, use explicit fence instead)
// =============================================================================
//
// Protocol mirrors CUDASymmetricMemory-inl.cuh: each slot toggles 0 -> 1 (put)
// then 1 -> 0 (wait), so pads return to zero and the barrier stays reusable.
// CUDA expresses this with a single system-scope CAS. Without atomic_ref we
// spin on a load and follow with a store, which is equivalent here because
// every slot has exactly one writer and one reader.
//
// Note on memory scope:
// We intentionally use memory_scope::system because signal pads are exchanged
// across ranks/devices (including peer/device-visible IPC mappings). These
// flags are polled and updated by kernels running on different devices, so a
// device/work-group scope is too narrow for this protocol.
//
// Note on cache control:
// A default store leaves the flag in L1 write-back where the peer GPU never
// sees it, and a default load in a spin loop keeps hitting the stale L1 copy.
// Both accesses therefore use LSC messages with L1 bypassed (`.uc`).

// L1-uncached / L3-cached load, so a spin loop observes remote updates.
inline uint32_t ld_flag_sys(const uint32_t* addr) {
#ifdef __SYCL_DEVICE_ONLY__
  uint32_t val;
  asm volatile("lsc_load.ugm.uc.ca (M1, 32) %0:d32 flat[%1]:a64"
               : "=rw"(val)
               : "rw"(addr));
  return val;
#else
  return *static_cast<const volatile uint32_t*>(addr);
#endif
}

// L1-uncached / L3-write-back store, so the flag reaches the peer.
inline void st_flag_sys(uint32_t* addr, uint32_t val) {
#ifdef __SYCL_DEVICE_ONLY__
  asm volatile("lsc_store.ugm.uc.wb (M1, 32) flat[%0]:a64 %1:d32"
               :
               : "rw"(addr), "rw"(val)
               : "memory");
#else
  *static_cast<volatile uint32_t*>(addr) = val;
#endif
}

// Store value with release semantics (for put_signal)
// Order: fence first, so preceding writes are visible once the flag is.
// The fence is acq_rel rather than release so it also carries the acquire for
// the preceding spin: two back-to-back system fences after a spin loop crash
// the device linker on oneAPI 2026.0 ("Callee is not a pointer type").
// atomic_fence only establishes ordering; the LSC fence additionally evicts
// pending UGM writes to the system coherence point.
inline void store_release(uint32_t* addr, uint32_t val) {
  sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
#ifdef __SYCL_DEVICE_ONLY__
  asm volatile("lsc_fence.ugm.evict.sysrel" ::: "memory");
#endif
  st_flag_sys(addr, val);
}

// Load value with acquire semantics (for get_signal/wait_signal)
// Order: load first, then acquire fence to order subsequent reads.
inline uint32_t load_acquire(uint32_t* addr) {
  uint32_t val = ld_flag_sys(addr);
  sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
  return val;
}

// Spin without fencing. A system-scope fence per iteration would flush caches
// on every poll; callers pair this with the acq_rel fence in store_release.
inline void spin_until_eq(uint32_t* addr, uint32_t val) {
  while (ld_flag_sys(addr) != val) {
    // Spin wait (no timeout check: XPU has no device-side global timer yet,
    // an IGC request is pending)
    continue;
  }
}

// =============================================================================
// Put signal: wait until addr == 0, then set to 1 (release semantics)
// =============================================================================

inline bool try_put_signal_device(uint32_t* addr, size_t timeout_ms) {
  // Wait until the slot is free (value == 0)
  spin_until_eq(addr, 0);
  // Set signal to 1 with release semantics
  store_release(addr, 1);
  return true;
}

// =============================================================================
// Wait signal: wait until addr == 1, then set to 0 (acquire semantics)
// =============================================================================
inline bool try_wait_signal_device(uint32_t* addr, size_t timeout_ms) {
  // Wait until signal is set (value == 1)
  spin_until_eq(addr, 1);
  // Clear signal to 0; the acq_rel fence inside also provides the acquire that
  // orders subsequent reads of the data the peer published.
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
