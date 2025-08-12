#include <xccl/XPUSymmetricMemory.hpp>
#include <xccl/XPUSymmetricMemoryUtils.hpp>

#include <ATen/ceil_div.h>
#include <c10/util/Exception.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

#include <sys/socket.h>
#include <unistd.h>
#include <shared_mutex>

namespace c10d {
namespace symmetric_memory {

// A set of exchange methods with prefix "XPUSymmetricMemory"
static XPUStoreExchange storeExchange = XPUStoreExchange("XPUSymmetricMemory");

XPUAllocationRef::XPUAllocationRef(
    void* ptr,
    HandleType handle,
    size_t block_size,
    int device_idx)
    : ptr(ptr),
      handle(handle),
      block_size(block_size),
      device_idx(device_idx) {}

XPUAllocationRef::~XPUAllocationRef() {
  if (is_finalizing()) {
    return;
  }
  // Set device context
  c10::xpu::set_device(device_idx);

  // Synchronize before releasing memory - using raw SYCL queue access
  try {
    auto& sycl_device = c10::xpu::get_raw_device(device_idx);
    auto& sycl_context = c10::xpu::get_device_context();

    // Create a queue for synchronization
    sycl::queue queue(sycl_context, sycl_device);
    queue.wait();

    // Get Level Zero context from SYCL context
    sycl::context ctx = queue.get_context();
    ze_context_handle_t context =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

    if (ptr != nullptr) {
      zeMemFree(context, ptr);
    }
  } catch (...) {
    // Ignore errors during destruction to avoid exceptions in destructors
  }
}

XPUSymmetricMemory::XPUSymmetricMemory(
    std::vector<c10::intrusive_ptr<XPUAllocationRef>> alloc_refs,
    std::vector<void*> buffers,
    std::vector<void*> signal_pads,
    size_t buffer_size,
    int local_device_idx,
    int rank,
    int world_size)
    : alloc_refs_(std::move(alloc_refs)),
      buffers_(std::move(buffers)),
      signal_pads_(std::move(signal_pads)),
      buffer_size_(buffer_size),
      local_device_idx_(local_device_idx),
      rank_(rank),
      world_size_(world_size) {
  const size_t arr_size = sizeof(void*) * world_size_;
  buffers_dev_ = reinterpret_cast<void**>(
      c10::xpu::XPUCachingAllocator::raw_alloc(arr_size));
  signal_pads_dev_ = reinterpret_cast<void**>(
      c10::xpu::XPUCachingAllocator::raw_alloc(arr_size));

  // Create SYCL queue for memory operations
  auto& sycl_device = c10::xpu::get_raw_device(local_device_idx);
  auto& sycl_context = c10::xpu::get_device_context();
  sycl::queue queue(sycl_context, sycl_device);

  queue.memcpy(buffers_dev_, buffers_.data(), arr_size).wait();
  queue.memcpy(signal_pads_dev_, signal_pads_.data(), arr_size).wait();
}

std::vector<void*> XPUSymmetricMemory::get_buffer_ptrs() {
  return buffers_;
}

std::vector<void*> XPUSymmetricMemory::get_signal_pad_ptrs() {
  return signal_pads_;
}

void** XPUSymmetricMemory::get_buffer_ptrs_dev() {
  return buffers_dev_;
}

void** XPUSymmetricMemory::get_signal_pad_ptrs_dev() {
  return signal_pads_dev_;
}

size_t XPUSymmetricMemory::get_buffer_size() {
  return buffer_size_;
}

size_t XPUSymmetricMemory::get_signal_pad_size() {
  return symmetric_memory::signal_pad_size;
}

bool XPUSymmetricMemory::has_multicast_support() {
  // XPU currently doesn't support multicast memory in the same way as CUDA
  return false;
}

void* XPUSymmetricMemory::get_multicast_ptr() {
  return nullptr;
}

at::Tensor XPUSymmetricMemory::get_buffer(
    int rank,
    c10::IntArrayRef sizes,
    c10::ScalarType dtype,
    int64_t storage_offset) {
  const size_t numel = std::accumulate(
      sizes.begin(),
      sizes.end(),
      static_cast<size_t>(1),
      std::multiplies<size_t>());
  const auto element_size = c10::elementSize(dtype);
  const auto req_size = (numel + storage_offset) * element_size;
  TORCH_CHECK(
      req_size <= buffer_size_,
      "XPUSymmetricMemory::get_buffer: the requested size (",
      req_size,
      " bytes) exceeds the allocated size (",
      buffer_size_,
      " bytes)");
  auto data_ptr = reinterpret_cast<uint8_t*>(buffers_[rank]) +
      storage_offset * element_size;
  auto device = c10::Device(c10::DeviceType::XPU, local_device_idx_);
  auto options = at::TensorOptions().dtype(dtype).device(device);
  return at::for_blob(data_ptr, sizes)
      .options(options)
      .target_device(device)
      .make_tensor();
}

at::Tensor XPUSymmetricMemory::get_signal_pad(
    int rank,
    c10::IntArrayRef sizes,
    std::optional<c10::ScalarType> dtype,
    int64_t storage_offset) {
  // If the dtype is unspecified, default it to UInt32, as it
  // is the most common type for signaling purposes.
  if (!dtype.has_value()) {
    dtype = c10::ScalarType::UInt32;
  }

  // If the shape is unspecified, treat the signal pad as a 1d tensor.
  const auto element_size = c10::elementSize(*dtype);
  std::vector<int64_t> shape;
  if (!sizes.empty()) {
    shape = sizes.vec();
  } else {
    shape.push_back(symmetric_memory::signal_pad_size / element_size);
  }

  const size_t numel = std::accumulate(
      shape.begin(),
      shape.end(),
      static_cast<size_t>(1),
      std::multiplies<size_t>());
  const auto req_size = (numel + storage_offset) * element_size;
  TORCH_CHECK(
      req_size <= symmetric_memory::signal_pad_size,
      "XPUSymmetricMemory::get_signal_pad: the requested size (",
      req_size,
      " bytes) exceeds the allocated size (",
      symmetric_memory::signal_pad_size,
      " bytes)");
  auto data_ptr = reinterpret_cast<uint8_t*>(signal_pads_[rank]) +
      storage_offset * element_size;
  auto device = c10::Device(c10::DeviceType::XPU, local_device_idx_);
  auto options = at::TensorOptions().dtype(*dtype).device(device);
  return at::for_blob(data_ptr, shape)
      .options(options)
      .target_device(device)
      .make_tensor();
}

void check_channel(int channel, int world_size) {
  TORCH_CHECK(
      channel >= 0,
      "channel for barrier(), put_signal() and wait_signal() ",
      "must be greater than 0 (got ",
      channel,
      ")");
  const size_t num_channels =
      symmetric_memory::signal_pad_size / sizeof(uint32_t) * world_size;
  TORCH_CHECK(
      static_cast<size_t>(channel) < num_channels,
      "The maximum supported channel for barrier(), put_signal() and wait_signal() is ",
      num_channels - 1,
      " (got ",
      channel,
      ")");
}

void XPUSymmetricMemory::barrier(int channel, size_t timeout_ms) {
  check_channel(channel, world_size_);

  // Create SYCL queue for kernel execution
  auto& sycl_device = c10::xpu::get_raw_device(local_device_idx_);
  auto& sycl_context = c10::xpu::get_device_context();
  sycl::queue queue(sycl_context, sycl_device);

  // Capture member variables explicitly to avoid 'this' capture
  auto world_size = world_size_;
  auto rank = rank_;
  auto signal_pads_dev = signal_pads_dev_;

  // Submit kernel for barrier synchronization
  queue
      .submit([&](sycl::handler& h) {
        h.parallel_for<class barrier_kernel>(
            sycl::range<1>(world_size), [=](sycl::id<1> idx) {
              auto target_rank = idx[0];
              if (target_rank == rank) {
                return;
              }

              // Signal other ranks
              auto* target_signal =
                  reinterpret_cast<uint32_t*>(signal_pads_dev[target_rank]) +
                  world_size * channel + rank;
              auto signal_ref = sycl::atomic_ref<
                  uint32_t,
                  sycl::memory_order::acq_rel,
                  sycl::memory_scope::system>(*target_signal);
              signal_ref.store(1);

              // Wait for signal from other ranks
              auto* my_signal =
                  reinterpret_cast<uint32_t*>(signal_pads_dev[rank]) +
                  world_size * channel + target_rank;
              auto wait_ref = sycl::atomic_ref<
                  uint32_t,
                  sycl::memory_order::acq_rel,
                  sycl::memory_scope::system>(*my_signal);
              while (wait_ref.load() != 1) {
                // Busy wait - could be improved with better synchronization
              }
              wait_ref.store(0); // Reset for next use
            });
      })
      .wait();
}

void XPUSymmetricMemory::put_signal(
    int dst_rank,
    int channel,
    size_t timeout_ms) {
  check_channel(channel, world_size_);

  auto& sycl_device = c10::xpu::get_raw_device(local_device_idx_);
  auto& sycl_context = c10::xpu::get_device_context();
  sycl::queue queue(sycl_context, sycl_device);

  // Capture member variables explicitly
  auto world_size = world_size_;
  auto rank = rank_;
  auto signal_pads_dev = signal_pads_dev_;

  queue
      .submit([&](sycl::handler& h) {
        h.single_task<class put_signal_kernel>([=]() {
          auto* signal_addr =
              reinterpret_cast<uint32_t*>(signal_pads_dev[dst_rank]) +
              world_size * channel + rank;
          auto signal_ref = sycl::atomic_ref<
              uint32_t,
              sycl::memory_order::acq_rel,
              sycl::memory_scope::system>(*signal_addr);
          signal_ref.store(1);
        });
      })
      .wait();
}

void XPUSymmetricMemory::wait_signal(
    int src_rank,
    int channel,
    size_t timeout_ms) {
  check_channel(channel, world_size_);

  auto& sycl_device = c10::xpu::get_raw_device(local_device_idx_);
  auto& sycl_context = c10::xpu::get_device_context();
  sycl::queue queue(sycl_context, sycl_device);

  // Capture member variables explicitly
  auto world_size = world_size_;
  auto rank = rank_;
  auto signal_pads_dev = signal_pads_dev_;

  queue
      .submit([&](sycl::handler& h) {
        h.single_task<class wait_signal_kernel>([=]() {
          auto* signal_addr =
              reinterpret_cast<uint32_t*>(signal_pads_dev[rank]) +
              world_size * channel + src_rank;
          auto signal_ref = sycl::atomic_ref<
              uint32_t,
              sycl::memory_order::acq_rel,
              sycl::memory_scope::system>(*signal_addr);
          while (signal_ref.load() != 1) {
            // Busy wait - could be improved with better synchronization
          }
          signal_ref.store(0); // Reset for next use
        });
      })
      .wait();
}

int XPUSymmetricMemory::get_rank() {
  return rank_;
}

int XPUSymmetricMemory::get_world_size() {
  return world_size_;
}

void XPUSymmetricMemory::copy_buffer(
    at::Tensor src,
    at::Tensor dst,
    size_t size) {
  // Create SYCL queue for memory operations
  auto& sycl_device = c10::xpu::get_raw_device(local_device_idx_);
  auto& sycl_context = c10::xpu::get_device_context();
  sycl::queue queue(sycl_context, sycl_device);

  // Copy data using SYCL memcpy
  queue.memcpy(dst.data_ptr(), src.data_ptr(), size).wait();
}

XPUBlock::XPUBlock(
    c10::intrusive_ptr<XPUAllocationRef> alloc_ref,
    int device_idx,
    size_t block_size,
    size_t buffer_size,
    size_t signal_pad_offset,
    const std::optional<std::string>& group_name)
    : alloc_ref(std::move(alloc_ref)),
      device_idx(device_idx),
      block_size(block_size),
      buffer_size(buffer_size),
      signal_pad_offset(signal_pad_offset),
      default_group_name(std::move(group_name)) {}

void* XPUSymmetricMemoryAllocator::alloc(
    size_t size,
    int device_idx,
    const std::optional<std::string>& group_name) {
  size_t signal_pad_offset = at::round_up(size, 16UL);
  size_t block_size = signal_pad_offset + symmetric_memory::signal_pad_size;

  c10::xpu::set_device(device_idx);

  // Get SYCL queue and context in the correct way
  auto& sycl_device = c10::xpu::get_raw_device(device_idx);
  auto& sycl_context = c10::xpu::get_device_context();
  sycl::queue queue(sycl_context, sycl_device);

  // Get Level Zero context from SYCL context
  sycl::context ctx = queue.get_context();
  ze_context_handle_t ze_context =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
  ze_device_handle_t ze_device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);

  // Allocate device memory using Level Zero API
  ze_device_mem_alloc_desc_t alloc_desc = {};
  alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  alloc_desc.ordinal = 0; // Memory type ordinal
  alloc_desc.flags = 0;
  alloc_desc.pNext = nullptr;

  void* ptr = nullptr;
  ze_result_t result =
      zeMemAllocDevice(ze_context, &alloc_desc, block_size, 0, ze_device, &ptr);
  TORCH_CHECK(
      result == ZE_RESULT_SUCCESS, "Level Zero memory allocation failed");

  // Get IPC handle for sharing
  HandleType handle;
  result = zeMemGetIpcHandle(ze_context, ptr, &handle);
  TORCH_CHECK(
      result == ZE_RESULT_SUCCESS, "Failed to get Level Zero IPC handle");

  // Zero out the memory
  queue.memset(ptr, 0, block_size).wait();

  auto alloc_ref = c10::make_intrusive<XPUAllocationRef>(
      ptr, handle, block_size, device_idx);
  auto block = c10::make_intrusive<XPUBlock>(
      std::move(alloc_ref),
      device_idx,
      block_size,
      size,
      signal_pad_offset,
      group_name);

  {
    std::unique_lock lock(mutex_);
    ptr_to_block_.emplace(ptr, std::move(block));
  }
  return ptr;
}

void XPUSymmetricMemoryAllocator::free(void* ptr) {
  std::unique_lock lock(mutex_);
  ptr_to_block_.erase(ptr);
}

size_t XPUSymmetricMemoryAllocator::get_alloc_size(void* ptr) {
  auto block = find_block(ptr);
  TORCH_CHECK(
      block != nullptr,
      "XPUSymmetricMemoryAllocator::get_alloc_size: input must be allocated ",
      "via XPUSymmetricMemoryAllocator::alloc");
  return block->buffer_size;
}

struct XPURendezvousRequest {
  int device_idx;
  int pid;
  size_t block_size;
  size_t buffer_size;
  size_t signal_pad_offset;
  bool has_multicast_support;
};

void validate_xpu_rendezvous_requests(
    const std::vector<XPURendezvousRequest>& reqs,
    int world_size) {
  TORCH_CHECK(reqs.size() == (size_t)world_size);

  std::unordered_set<int> device_indices;
  device_indices.reserve(world_size);
  for (auto req : reqs) {
    device_indices.insert(req.device_idx);
  }
  if (!allow_overlapping_devices() &&
      device_indices.size() < (size_t)world_size) {
    TORCH_CHECK(
        false,
        "XPUSymmetricMemoryAllocator::rendezvous: ",
        "detected allocations from overlapping devices ",
        "from different ranks.");
  }

  for (int r = 1; r < world_size; ++r) {
    TORCH_CHECK(reqs[r].block_size == reqs[0].block_size);
    TORCH_CHECK(reqs[r].buffer_size == reqs[0].buffer_size);
    TORCH_CHECK(reqs[r].signal_pad_offset == reqs[0].signal_pad_offset);
  }
}

c10::intrusive_ptr<SymmetricMemory> XPUSymmetricMemoryAllocator::rendezvous(
    void* ptr,
    const std::optional<std::string>& group_name) {
  auto block = find_block(ptr);
  if (block == nullptr) {
    return nullptr;
  }

  // The group_name passed to rendezvous() takes precedence over
  // the default group_name specified during allocation.
  std::string group_name_;
  if (group_name.has_value() && group_name != "") {
    group_name_ = *group_name;
  } else {
    if (!block->default_group_name.has_value()) {
      TORCH_CHECK(
          false,
          "XPUSymmetricMemory::rendezvous: `group_name` is neither "
          "specified during allocation nor passed to rendezvous().");
    }
    group_name_ = *block->default_group_name;
  }

  auto it = block->symm_mems.find(group_name_);
  if (it != block->symm_mems.end()) {
    return it->second;
  }

  c10::xpu::set_device(block->device_idx);

  // Currently, XPUIpcChannel is using a file based socket for inter-process
  // communication
  XPUIpcChannel ipc_channel;
  auto group_info = get_group_info(group_name_);
  auto store = group_info.store;
  int rank = group_info.rank;
  int world_size = group_info.world_size;

  auto local_req = XPURendezvousRequest{
      .device_idx = block->device_idx,
      .pid = getpid(),
      .block_size = block->block_size,
      .buffer_size = block->buffer_size,
      .signal_pad_offset = block->signal_pad_offset,
      .has_multicast_support = false}; // XPU doesn't support multicast yet

  auto reqs = storeExchange.all_gather(store, rank, world_size, local_req);
  validate_xpu_rendezvous_requests(reqs, world_size);

  std::vector<int> pids(world_size);
  for (int r = 0; r < world_size; ++r) {
    pids[r] = reqs[r].pid;
  }

  auto imported_handles =
      ipc_channel.all_gather_handles(rank, pids, block->alloc_ref->handle);

  std::vector<void*> buffers(world_size, nullptr);
  std::vector<void*> signal_pads(world_size, nullptr);

  for (int r = 0; r < world_size; ++r) {
    if (r == rank) {
      buffers[r] = ptr;
      signal_pads[r] = (void*)((uintptr_t)ptr + block->signal_pad_offset);
      continue;
    }

    // Map remote memory using Level Zero IPC
    c10::xpu::set_device(block->device_idx);
    auto& sycl_device = c10::xpu::get_raw_device(block->device_idx);
    auto& sycl_context = c10::xpu::get_device_context();
    sycl::queue queue(sycl_context, sycl_device);

    // Get Level Zero context from SYCL context
    sycl::context ctx = queue.get_context();
    ze_context_handle_t ze_context =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    ze_device_handle_t ze_device =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);

    void* mapped_ptr = nullptr;
    ze_result_t result = zeMemOpenIpcHandle(
        ze_context, ze_device, imported_handles[r], 0, &mapped_ptr);
    TORCH_CHECK(
        result == ZE_RESULT_SUCCESS, "Failed to open Level Zero IPC handle");

    buffers[r] = mapped_ptr;
    signal_pads[r] = (void*)((uintptr_t)mapped_ptr + block->signal_pad_offset);
  }

  storeExchange.barrier(store, rank, world_size);

  std::vector<c10::intrusive_ptr<XPUAllocationRef>> alloc_refs;
  for (int r = 0; r < world_size; ++r) {
    if (r == rank) {
      alloc_refs.emplace_back(block->alloc_ref);
      continue;
    }
    // Create dummy handles for remote allocations
    HandleType dummy_handle = {};
    alloc_refs.push_back(c10::make_intrusive<XPUAllocationRef>(
        buffers[r], dummy_handle, block->block_size, block->device_idx));
  }

  auto symm_mem = c10::make_intrusive<XPUSymmetricMemory>(
      std::move(alloc_refs),
      std::move(buffers),
      std::move(signal_pads),
      block->buffer_size,
      block->device_idx,
      group_info.rank,
      group_info.world_size);

  block->symm_mems[group_name_] = symm_mem;
  return symm_mem;
}

bool XPUSymmetricMemoryAllocator::has_multicast_support(int device_idx) {
  // XPU currently doesn't support multicast memory
  return false;
}

c10::DeviceType XPUSymmetricMemoryAllocator::supported_device_type() {
  return c10::DeviceType::XPU;
}

std::string XPUSymmetricMemoryAllocator::name() {
  return "XPU";
}

c10::intrusive_ptr<XPUBlock> XPUSymmetricMemoryAllocator::find_block(
    void* ptr) {
  std::shared_lock lock(mutex_);
  auto it = ptr_to_block_.find(ptr);
  if (it == ptr_to_block_.end()) {
    return nullptr;
  }
  return it->second;
}

struct RegisterXPUSymmetricMemoryAllocator {
  RegisterXPUSymmetricMemoryAllocator() {
    auto allocator = c10::make_intrusive<XPUSymmetricMemoryAllocator>();
    // Query backend used for XPU
    if (getSymmMemBackendXPU() == "XPU") {
      // Direct set (static registration)
      register_allocator(c10::DeviceType::XPU, allocator);
    } else {
      // Register availability in case `set_backend` is called dynamically
      register_availability("XPU", allocator);
    }
  }
};

static RegisterXPUSymmetricMemoryAllocator register_allocator_;

} // namespace symmetric_memory
} // namespace c10d
