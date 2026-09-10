#include <ATen/xpu/XPUContext.h>
#include <comm/SYCLContext.h>
#include <xccl/Signal.hpp>
#include <chrono>

namespace c10d::symmetric_memory {

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void barrier_kernel(
    uint32_t** signal_pads,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  auto thread_id = item.get_local_id(0);

  if (thread_id < world_size) {
    auto target_rank = thread_id;
    if (target_rank == rank) {
      return;
    }
    auto put_success = try_put_signal_device(
        signal_pads[target_rank] + world_size * channel + rank, timeout_ms);
    if (!put_success) {
      SYCL_KERNEL_ASSERT(false);
    }

    auto wait_success = try_wait_signal_device(
        signal_pads[rank] + world_size * channel + target_rank, timeout_ms);
    if (!wait_success) {
      SYCL_KERNEL_ASSERT(false);
    }
  }
}

void barrier_impl_xpu(
    uint32_t** signal_pads,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    at::xpu::XPUStream& stream) {
  int64_t maxNumThreadsPerBlock = syclMaxWorkGroupSize<barrier_kernel>();
  const size_t numThreadsPerBlock =
      std::min<size_t>(maxNumThreadsPerBlock, std::max(32, world_size));

  if (!(numThreadsPerBlock > 0)) {
    return;
  }
  int64_t numBlocks = 1;
  auto global_range = numBlocks * numThreadsPerBlock;
  auto local_range = numThreadsPerBlock;

  sycl_kernel_submit<barrier_kernel>(
      global_range,
      local_range,
      stream.queue(),
      0,
      signal_pads,
      channel,
      rank,
      world_size,
      timeout_ms);
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void put_signal_kernel(
    uint32_t** signal_pads,
    int dst_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  auto thread_id = item.get_local_id(0);

  if (thread_id == 0) {
    auto put_success = try_put_signal_device(
        signal_pads[dst_rank] + world_size * channel + rank, timeout_ms);
    if (!put_success) {
      SYCL_KERNEL_ASSERT(false);
    }
  }
}

void put_signal_impl_xpu(
    uint32_t** signal_pads,
    int dst_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    at::xpu::XPUStream& stream) {
  int64_t maxNumThreadsPerBlock = syclMaxWorkGroupSize<put_signal_kernel>();
  const size_t numThreadsPerBlock = std::min<size_t>(maxNumThreadsPerBlock, 32);

  if (!(numThreadsPerBlock > 0)) {
    return;
  }

  int64_t numBlocks = 1;
  auto global_range = numBlocks * numThreadsPerBlock;
  auto local_range = numThreadsPerBlock;

  sycl_kernel_submit<put_signal_kernel>(
      global_range,
      local_range,
      stream.queue(),
      0,
      signal_pads,
      dst_rank,
      channel,
      rank,
      world_size,
      timeout_ms);
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void wait_signal_kernel(
    uint32_t** signal_pads,
    int src_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  auto thread_id = item.get_local_id(0);

  if (thread_id == 0) {
    auto wait_success = try_wait_signal_device(
        signal_pads[rank] + world_size * channel + src_rank, timeout_ms);
    if (!wait_success) {
      SYCL_KERNEL_ASSERT(false);
    }
  }
}

void wait_signal_impl_xpu(
    uint32_t** signal_pads,
    int src_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    at::xpu::XPUStream& stream) {
  int64_t maxNumThreadsPerBlock = syclMaxWorkGroupSize<wait_signal_kernel>();
  const size_t numThreadsPerBlock = std::min<size_t>(maxNumThreadsPerBlock, 32);

  if (!(numThreadsPerBlock > 0)) {
    return;
  }

  int64_t numBlocks = 1;
  auto global_range = numBlocks * numThreadsPerBlock;
  auto local_range = numThreadsPerBlock;

  sycl_kernel_submit<wait_signal_kernel>(
      global_range,
      local_range,
      stream.queue(),
      0,
      signal_pads,
      src_rank,
      channel,
      rank,
      world_size,
      timeout_ms);
}

} // namespace c10d::symmetric_memory
