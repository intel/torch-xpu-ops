#include <ATen/xpu/XPUContext.h>
#include <comm/SYCLContext.h>
#include <xccl/Signal.hpp>
#include <chrono>

namespace c10d::symmetric_memory {

struct barrierKernel {
  void operator()(sycl::nd_item<1> item) const {
    auto thread_id = item.get_local_id(0);

    // DPCPP_KER_PRINTF(
    //     "[DEBUG] Barrier kernel start: thread_id=%zu, rank=%d, world_size=%d,
    //     channel=%d\n", thread_id, rank, world_size, channel);

    if (thread_id < world_size) {
      auto target_rank = thread_id;
      if (target_rank == rank) {
        DPCPP_KER_PRINTF("[DEBUG] Barrier skipping self rank %d\n", rank);
        return;
      }

      uint32_t* put_addr =
          signal_pads[target_rank] + world_size * channel + rank;
      DPCPP_KER_PRINTF(
          "[DEBUG] Barrier putting signal from rank %d to rank %zu at addr=%p\n",
          rank,
          target_rank,
          put_addr);

      auto put_success =
          try_put_signal_device<std::memory_order_release>(put_addr, 10000000);
      if (!put_success) {
        DPCPP_KER_PRINTF(
            "[DEBUG] Barrier FAILED to put signal from rank %d to rank %zu\n",
            rank,
            target_rank);
        assert(0);
      }

      DPCPP_KER_PRINTF(
          "[DEBUG] Barrier successfully put signal from rank %d to rank %zu\n",
          rank,
          target_rank);

      uint32_t* wait_addr =
          signal_pads[rank] + world_size * channel + target_rank;
      DPCPP_KER_PRINTF(
          "[DEBUG] Barrier waiting for signal at rank %d from rank %zu at addr=%p\n",
          rank,
          target_rank,
          wait_addr);

      auto wait_success = try_wait_signal_device<std::memory_order_acquire>(
          wait_addr, 10000000);
      if (!wait_success) {
        DPCPP_KER_PRINTF(
            "[DEBUG] Barrier TIMEOUT waiting at rank %d for rank %zu\n",
            rank,
            target_rank);
        assert(0);
      }

      DPCPP_KER_PRINTF(
          "[DEBUG] Barrier successfully received signal at rank %d from rank %zu\n",
          rank,
          target_rank);
    }

    // DPCPP_KER_PRINTF(
    //     "[DEBUG] Barrier kernel complete for thread %zu\n", thread_id);
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

    // DPCPP_KER_PRINTF(
    //     "[DEBUG] PutSignal kernel start: thread_id=%zu, rank=%d->%d,
    //     channel=%d\n", thread_id, rank, dst_rank, channel);

    if (thread_id == 0) {
      uint32_t* put_addr = signal_pads[dst_rank] + world_size * channel + rank;
      DPCPP_KER_PRINTF(
          "[DEBUG] PutSignal putting signal from rank %d to rank %d at addr=%p\n",
          rank,
          dst_rank,
          put_addr);

      auto put_success =
          try_put_signal_device<std::memory_order_release>(put_addr, 10000000);
      if (!put_success) {
        DPCPP_KER_PRINTF(
            "[DEBUG] PutSignal FAILED from rank %d to rank %d\n",
            rank,
            dst_rank);
        assert(0);
      }

      DPCPP_KER_PRINTF(
          "[DEBUG] PutSignal SUCCESS from rank %d to rank %d\n",
          rank,
          dst_rank);
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

    // DPCPP_KER_PRINTF(
    //     "[DEBUG] WaitSignal kernel start: thread_id=%zu, rank=%d<-%d,
    //     channel=%d\n", thread_id, rank, src_rank, channel);

    if (thread_id == 0) {
      uint32_t* wait_addr = signal_pads[rank] + world_size * channel + src_rank;
      DPCPP_KER_PRINTF(
          "[DEBUG] WaitSignal waiting at rank %d for signal from rank %d at addr=%p\n",
          rank,
          src_rank,
          wait_addr);

      auto wait_success = try_wait_signal_device<std::memory_order_acquire>(
          wait_addr, 10000000);
      if (!wait_success) {
        DPCPP_KER_PRINTF(
            "[DEBUG] WaitSignal TIMEOUT at rank %d waiting for rank %d\n",
            rank,
            src_rank);
        assert(0);
      }

      DPCPP_KER_PRINTF(
          "[DEBUG] WaitSignal SUCCESS at rank %d received from rank %d\n",
          rank,
          src_rank);

      // sycl::atomic_fence(sycl::memory_order_seq_cst,
      // sycl::memory_scope_system);
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
