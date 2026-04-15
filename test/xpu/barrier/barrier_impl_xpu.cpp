#include "barrier_impl_xpu.hpp"

#include <algorithm>

namespace test_xpu_barrier {

namespace {

inline void store_release(uint32_t* addr, uint32_t val) {
  *addr = val;
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
}

inline uint32_t load_acquire(uint32_t* addr) {
  sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
  return *addr;
}

inline bool try_put_signal_device(uint32_t* addr, size_t max_iterations) {
  size_t iterations = 0;
  while (load_acquire(addr) != 0) {
    if (max_iterations != 0 && iterations++ > max_iterations) {
      return false;
    }
  }
  store_release(addr, 1);
  return true;
}

inline bool try_wait_signal_device(uint32_t* addr, size_t max_iterations) {
  size_t iterations = 0;
  while (load_acquire(addr) != 1) {
    if (max_iterations != 0 && iterations++ > max_iterations) {
      return false;
    }
  }
  store_release(addr, 0);
  return true;
}

struct BarrierKernel {
  void operator()(sycl::nd_item<1> item) const {
    const size_t thread_id = item.get_local_id(0);

    if (thread_id < static_cast<size_t>(world_size)) {
      const int target_rank = static_cast<int>(thread_id);
      if (target_rank == rank) {
        return;
      }

      (void)try_put_signal_device(
          signal_pads[target_rank] + world_size * channel + rank,
          timeout_ms == 0 ? 0 : timeout_ms * 1000);
      (void)try_wait_signal_device(
          signal_pads[rank] + world_size * channel + target_rank,
          timeout_ms == 0 ? 0 : timeout_ms * 1000);
    }
  }

  uint32_t** signal_pads;
  int channel;
  int rank;
  int world_size;
  size_t timeout_ms;
};

struct BarrierAllRanksKernel {
  void operator()(sycl::nd_item<1> item) const {
    const size_t thread_id = item.get_local_id(0);
    if (thread_id >= static_cast<size_t>(world_size)) {
      return;
    }

    const int rank = static_cast<int>(thread_id);

    for (int target_rank = 0; target_rank < world_size; ++target_rank) {
      if (target_rank == rank) {
        continue;
      }

      (void)try_put_signal_device(
          signal_pads[target_rank] + world_size * channel + rank,
          timeout_ms == 0 ? 0 : timeout_ms * 1000);
      (void)try_wait_signal_device(
          signal_pads[rank] + world_size * channel + target_rank,
          timeout_ms == 0 ? 0 : timeout_ms * 1000);
    }
  }

  uint32_t** signal_pads;
  int channel;
  int world_size;
  size_t timeout_ms;
};

size_t get_num_threads_per_block(int world_size, const sycl::device& device) {
  const size_t max_work_group =
      device.get_info<sycl::info::device::max_work_group_size>();
  return std::min(max_work_group, std::max<size_t>(32, world_size));
}

} // namespace

sycl::event barrier_impl_xpu(
    uint32_t** signal_pads,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    sycl::queue& queue) {
  const size_t local_range = get_num_threads_per_block(world_size, queue.get_device());
  if (local_range == 0) {
    return sycl::event{};
  }

  BarrierKernel kernel{signal_pads, channel, rank, world_size, timeout_ms};
  return queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(local_range), sycl::range<1>(local_range)),
      kernel);
}

sycl::event barrier_impl_xpu_all_ranks(
    uint32_t** signal_pads,
    int channel,
    int world_size,
    size_t timeout_ms,
    sycl::queue& queue) {
  const size_t local_range = get_num_threads_per_block(world_size, queue.get_device());
  if (local_range == 0) {
    return sycl::event{};
  }

  BarrierAllRanksKernel kernel{signal_pads, channel, world_size, timeout_ms};
  return queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(local_range), sycl::range<1>(local_range)),
      kernel);
}

} // namespace test_xpu_barrier
