#pragma once

#include <atomic>

#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <comm/SYCLContext.h>

#if defined(__SYCL_DEVICE_ONLY__)
#define DPCPP_CONSTANT __attribute__((opencl_constant))
#else
#define DPCPP_CONSTANT
#endif

#define DPCPP_KER_STRING(var, str) static const DPCPP_CONSTANT char var[] = str;
#define DPCPP_KER_PRINTF sycl::ext::oneapi::experimental::printf

#define DPCPP_K_PRINT(fmt_str, ...)           \
  {                                           \
    DPCPP_KER_STRING(fmt_var, fmt_str);       \
    DPCPP_KER_PRINTF(fmt_var, ##__VA_ARGS__); \
  }

namespace c10d::symmetric_memory {

using at::native::memory::get_alignment;

template <std::memory_order Sem>
uint32_t cas(uint32_t* addr, uint32_t compare, uint32_t val) {
  sycl::atomic_ref<
      uint32_t,
      sycl::memory_order::acq_rel,
      sycl::memory_scope::system>
      ref(*addr);
  ref.compare_exchange_strong(compare, val);
  return compare;
}

// template <std::memory_order Sem>
// uint32_t cas(uint32_t* addr, uint32_t compare, uint32_t val) {
//   uint32_t old_value = *addr;
//   if (old_value == compare) {
//     *addr = val;
//   }
//   return old_value;
// }

inline size_t global_timer_ns() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             now.time_since_epoch())
      .count();
}

constexpr size_t ns_per_ms = 1e6;

// Device-compatible version using a simple counter approach
template <std::memory_order Sem>
bool try_put_signal_device(uint32_t* addr, size_t max_iterations = 1000) {
  size_t iterations = 0;
  uint32_t initial_value = *addr;

  DPCPP_KER_PRINTF(
      "[DEBUG] try_put_signal: addr=%p, initial_value=%u, max_iter=%zu\n",
      addr,
      initial_value,
      max_iterations);

  while (cas<Sem>(addr, 0, 1) != 0) {
    if (max_iterations != 0 && iterations++) {
      // if (max_iterations != 0 && iterations++ > max_iterations) {
      // DPCPP_KER_PRINTF(
      //     "[DEBUG] put_signal TIMEOUT: addr=%p, initial_value=%u,
      //     iterations=%zu\n", addr, initial_value, iterations);
      // return false;
    }
    if (iterations > 0 && iterations % 5000000 == 0) {
      uint32_t current_value = *addr;
      DPCPP_KER_PRINTF(
          "[DEBUG] put_signal progress: addr=%p, current_value=%u, iterations=%zu\n",
          addr,
          current_value,
          iterations);
    }
  }
  DPCPP_KER_PRINTF(
      "[DEBUG] put_signal SUCCESS: addr=%p, iterations=%zu\n",
      addr,
      iterations);
  return true;
}

template <std::memory_order Sem>
bool try_wait_signal_device(uint32_t* addr, size_t max_iterations = 1000) {
  size_t iterations = 0;
  uint32_t initial_value = *addr;

  DPCPP_KER_PRINTF(
      "[DEBUG] try_wait_signal: addr=%p, initial_value=%u, max_iter=%zu\n",
      addr,
      initial_value,
      max_iterations);

  while (cas<Sem>(addr, 1, 0) != 1) {
    if (max_iterations != 0 && iterations++) {
      // if (max_iterations != 0 && iterations++ > max_iterations) {
      // DPCPP_KER_PRINTF(
      //     "[DEBUG] wait_signal TIMEOUT: addr=%p, initial_value=%u,
      //     iterations=%zu\n", addr, initial_value, iterations);
      // return false;
    }

    // 添加简单的退避机制：每1000次迭代后暂停
    if (iterations > 0 && iterations % 1000 == 0) {
      for (volatile int i = 0; i < 10; ++i) {
      }
    }

    // 减少打印频率，每5百万次迭代打印一次进度
    if (iterations > 0 && iterations % 5000000 == 0) {
      uint32_t current_value = *addr;
      DPCPP_KER_PRINTF(
          "[DEBUG] wait_signal progress: addr=%p, current_value=%u, iterations=%zu\n",
          addr,
          current_value,
          iterations);
    }
  }
  DPCPP_KER_PRINTF(
      "[DEBUG] wait_signal SUCCESS: addr=%p, iterations=%zu\n",
      addr,
      iterations);
  return true;
}

template <std::memory_order Sem>
bool try_put_signal(uint32_t* addr, size_t timeout_ms) {
  size_t deadline = global_timer_ns() + timeout_ms * ns_per_ms;
  while (cas<Sem>(addr, 0, 1) != 0) {
    if (timeout_ms != 0 && global_timer_ns() > deadline) {
      return false;
    }
  }
  return true;
}

template <std::memory_order Sem>
bool try_wait_signal(uint32_t* addr, size_t timeout_ms) {
  size_t deadline = global_timer_ns() + timeout_ms * ns_per_ms;
  while (cas<Sem>(addr, 1, 0) != 1) {
    if (timeout_ms != 0 && global_timer_ns() > deadline) {
      return false;
    }
  }
  return true;
}

template <std::memory_order Sem>
void put_signal(uint32_t* addr) {
  while (cas<Sem>(addr, 0, 1) != 0)
    ;
}

template <std::memory_order Sem>
void wait_signal(uint32_t* addr) {
  while (cas<Sem>(addr, 1, 0) != 1)
    ;
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
