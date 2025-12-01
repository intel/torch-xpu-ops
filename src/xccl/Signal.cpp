#include <ATen/xpu/XPUContext.h>
#include <comm/SYCLContext.h>
#include <xccl/Signal.hpp>
#include <chrono>

namespace c10d::symmetric_memory {

struct barrierKernel {
  void operator()(sycl::nd_item<1> item) const {
    auto thread_id = item.get_local_id(0);

    if (thread_id < world_size) {
      auto target_rank = thread_id;
      if (target_rank == rank) {
        return;
      }
      auto put_success = try_put_signal_device<std::memory_order_release>(
          signal_pads[target_rank] + world_size * channel + rank, 10000000);
      if (!put_success) {
        assert(0);
      }

      auto wait_success = try_wait_signal_device<std::memory_order_acquire>(
          signal_pads[rank] + world_size * channel + target_rank, 10000000);
      if (!wait_success) {
        assert(0);
      }
    }
  }

  barrierKernel(
      uint32_t** signal_pads,
      int channel,
      int rank,
      int world_size,
      size_t timeout_ms)
      : signal_pads(signal_pads),
        channel(channel),
        rank(rank),
        world_size(world_size),
        timeout_ms(timeout_ms) {}

 private:
  uint32_t** signal_pads;
  int channel;
  int rank;
  int world_size;
  size_t timeout_ms;
};

void barrier_impl_xpu(
    uint32_t** signal_pads,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    at::xpu::XPUStream& stream) {
  int64_t maxNumThreadsPerBlock = syclMaxWorkGroupSize<barrierKernel>();
  const size_t numThreadsPerBlock =
      std::min<size_t>(maxNumThreadsPerBlock, std::max(32, world_size));

  if (!(numThreadsPerBlock > 0)) {
    return;
  }
  int64_t numBlocks = 1;
  auto global_range = numBlocks * numThreadsPerBlock;
  auto local_range = numThreadsPerBlock;

  using Kernel = barrierKernel;
  auto kfn = Kernel(signal_pads, channel, rank, world_size, timeout_ms);

  sycl_kernel_submit(global_range, local_range, stream.queue(), kfn);
}

struct putSignalKernel {
  void operator()(sycl::nd_item<1> item) const {
    auto thread_id = item.get_local_id(0);

    if (thread_id == 0) {
      uint32_t* target_addr =
          signal_pads[dst_rank] + world_size * channel + rank;

      auto put_success = try_put_signal_device<std::memory_order_release>(
          target_addr, 10000000);
      if (!put_success) {
        assert(0);
      }
    }
  }

  putSignalKernel(
      uint32_t** signal_pads,
      int dst_rank,
      int channel,
      int rank,
      int world_size,
      size_t timeout_ms)
      : signal_pads(signal_pads),
        dst_rank(dst_rank),
        channel(channel),
        rank(rank),
        world_size(world_size),
        timeout_ms(timeout_ms) {}

 private:
  uint32_t** signal_pads;
  int dst_rank;
  int channel;
  int rank;
  int world_size;
  size_t timeout_ms;
};

void put_signal_impl_xpu(
    uint32_t** signal_pads,
    int dst_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    at::xpu::XPUStream& stream) {
  int64_t maxNumThreadsPerBlock = syclMaxWorkGroupSize<putSignalKernel>();
  const size_t numThreadsPerBlock = std::min<size_t>(maxNumThreadsPerBlock, 32);

  if (!(numThreadsPerBlock > 0)) {
    return;
  }

  int64_t numBlocks = 1;
  auto global_range = numBlocks * numThreadsPerBlock;
  auto local_range = numThreadsPerBlock;

  using Kernel = putSignalKernel;
  auto kfn =
      Kernel(signal_pads, dst_rank, channel, rank, world_size, timeout_ms);

  sycl_kernel_submit(global_range, local_range, stream.queue(), kfn);
}

struct waitSignalKernel {
  void operator()(sycl::nd_item<1> item) const {
    auto thread_id = item.get_local_id(0);

    if (thread_id == 0) {
      uint32_t* target_addr =
          signal_pads[rank] + world_size * channel + src_rank;

      auto wait_success = try_wait_signal_device<std::memory_order_acquire>(
          target_addr, 10000000);
      if (!wait_success) {
        assert(0);
      }

      sycl::atomic_fence(sycl::memory_order_seq_cst, sycl::memory_scope_system);
    }
  }

  waitSignalKernel(
      uint32_t** signal_pads,
      int src_rank,
      int channel,
      int rank,
      int world_size,
      size_t timeout_ms)
      : signal_pads(signal_pads),
        src_rank(src_rank),
        channel(channel),
        rank(rank),
        world_size(world_size),
        timeout_ms(timeout_ms) {}

 private:
  uint32_t** signal_pads;
  int src_rank;
  int channel;
  int rank;
  int world_size;
  size_t timeout_ms;
};

void wait_signal_impl_xpu(
    uint32_t** signal_pads,
    int src_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    at::xpu::XPUStream& stream) {
  int64_t maxNumThreadsPerBlock = syclMaxWorkGroupSize<waitSignalKernel>();
  const size_t numThreadsPerBlock = std::min<size_t>(maxNumThreadsPerBlock, 32);

  if (!(numThreadsPerBlock > 0)) {
    return;
  }

  int64_t numBlocks = 1;
  auto global_range = numBlocks * numThreadsPerBlock;
  auto local_range = numThreadsPerBlock;

  using Kernel = waitSignalKernel;
  auto kfn =
      Kernel(signal_pads, src_rank, channel, rank, world_size, timeout_ms);

  sycl_kernel_submit(global_range, local_range, stream.queue(), kfn);
}

} // namespace c10d::symmetric_memory
