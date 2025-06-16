#include <xccl/XPUSymmetricMemory.hpp>
#include <xccl/XPUSymmetricMemoryUtils.hpp>

#include <ATen/ceil_div.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/error.h>

#include <sys/socket.h>
#include <unistd.h>

// todo: check this point
#include <level_zero/ze_api.h>

namespace c10d {
namespace symmetric_memory {

/* Start of XPUSymmetricMemory implementation */

// A set of exchange methods with prefix "XPUSymmetricMemory"
static StoreExchange storeExchange = StoreExchange("XPUSymmetricMemory");

AllocationRef::AllocationRef(
    void* ptr,
    HandleType handle,
    size_t block_size,
    int device_idx)
    : ptr(ptr),
      handle(handle),
      block_size(block_size),
      device_idx(device_idx) {}

AllocationRef::~AllocationRef() {
  if (is_finalizing()) {
    return;
  }
c10::Device local_device(c10::DeviceType::XPU, device_idx);
c10::DeviceGuard guard(local_device);
  c10::xpu::syncStreamsOnDevice();
}

XPUSymmetricMemory::XPUSymmetricMemory(
    std::vector<c10::intrusive_ptr<AllocationRef>> alloc_refs,
    std::vector<void*> buffers,
    std::vector<void*> signal_pads,
    HandleType mc_handle,
    void* mc_addr,
    size_t buffer_size,
    int local_device_idx,
    int rank,
    int world_size)
    : alloc_refs_(std::move(alloc_refs)),
      buffers_(std::move(buffers)),
      signal_pads_(std::move(signal_pads)),
      mc_handle_(mc_handle),
      mc_addr_(mc_addr),
      buffer_size_(buffer_size),
      local_device_idx_(local_device_idx),
      rank_(rank),
      world_size_(world_size) {
  const size_t arr_size = sizeof(void*) * world_size_;
  buffers_dev_ = reinterpret_cast<void**>(
      c10::xpu::XPUCachingAllocator::raw_alloc(arr_size));
  signal_pads_dev_ = reinterpret_cast<void**>(
      c10::xpu::XPUCachingAllocator::raw_alloc(arr_size));

  c10::Device local_device(c10::DeviceType::XPU, local_device_idx);
  c10::DeviceGuard guard(local_device);

  // todo: zl_debug
  at::xpu::getCurrentXPUStream().queue().memcpy(buffers_dev_, buffers_.data(), arr_size);
  at::xpu::getCurrentXPUStream().queue().memcpy(signal_pads_dev_, signal_pads_.data(), arr_size);
}

std::vector<void*> XPUSymmetricMemory::get_buffer_ptrs() {
  std::cout << "zl_debug in XPUSymmetricMemory::get_buffer_ptrs" << buffers_[0] << " ___ " << buffers_[1] << std::endl;
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
  return signal_pad_size;
}

bool XPUSymmetricMemory::has_multicast_support() {
  return mc_addr_ != nullptr;
}

void* XPUSymmetricMemory::get_multicast_ptr() {
  return mc_addr_;
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
  std::cout << "zl_debug in get_buffer " << rank << "___" << buffers_[rank] << "___" <<storage_offset * element_size << std::endl;
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
  if (sizes.size() != 0) {
    shape = sizes.vec();
  } else {
    shape.push_back(signal_pad_size / element_size);
  }

  const size_t numel = std::accumulate(
      shape.begin(),

      shape.end(),
      static_cast<size_t>(1),
      std::multiplies<size_t>());
  const auto req_size = (numel + storage_offset) * element_size;
  TORCH_CHECK(
      req_size <= signal_pad_size,
      "CUDASymmetricMemory::get_signal_pad: the requested size (",
      req_size,
      " bytes) exceeds the allocated size (",
      signal_pad_size,
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
  const size_t num_channels = signal_pad_size / sizeof(uint32_t) * world_size;
  TORCH_CHECK(
      static_cast<size_t>(channel) < num_channels,
      "The maximum supported channel for barrier(), put_signal() and wait_signal() is ",
      num_channels - 1,
      " (got ",
      channel,
      ")");
}

void XPUSymmetricMemory::barrier(int channel, size_t timeout_ms) {

//  LOG(ERROR) << "XPUSymmetricMemory::barrier not supported";
  check_channel(channel, world_size_);

  c10::Device local_device(c10::DeviceType::XPU, local_device_idx_);
  c10::DeviceGuard guard(local_device);

  sycl::queue current_queue = at::xpu::getCurrentXPUStream().queue();

//  current_queue.submit([&](handler& h) {
//    h.parallel_for(range<1>(world_size), [=](id<1> idx) {
//      int target_rank = idx[0];
//      if (target_rank == rank) {
//        return;
//      }
//      //todo: implement
////      bool put_success = try_put_signal<std::memory_order_release>(
////          signal_pads[target_rank] + world_size * channel + rank, timeout_ms);
////
////      bool wait_success = try_wait_signal<std::memory_order_acquire>(
////          signal_pads[rank] + world_size * channel + target_rank, timeout_ms);
//    });
//  });
}

void XPUSymmetricMemory::put_signal(
    int dst_rank,
    int channel,
    size_t timeout_ms) {

  LOG(ERROR) << "XPUSymmetricMemory::put_signal not supported";

//  check_channel(channel, world_size_);
//  c10::cuda::CUDAGuard guard(local_device_idx_);
//  put_signal_kernel<<<1, C10_WARP_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
//      reinterpret_cast<uint32_t**>(signal_pads_dev_),
//      dst_rank,
//      channel,
//      rank_,
//      world_size_,
//      timeout_ms);
//  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void XPUSymmetricMemory::wait_signal(
    int src_rank,
    int channel,
    size_t timeout_ms) {

    LOG(ERROR) << "XPUSymmetricMemory::wait_signal not supported";
//  check_channel(channel, world_size_);
//  c10::cuda::CUDAGuard guard(local_device_idx_);
//  wait_signal_kernel<<<1, C10_WARP_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
//      reinterpret_cast<uint32_t**>(signal_pads_dev_),
//      src_rank,
//      channel,
//      rank_,
//      world_size_,
//      timeout_ms);
//  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int XPUSymmetricMemory::get_rank() {
  return rank_;
}

int XPUSymmetricMemory::get_world_size() {
  return world_size_;
}

Block::Block(
    c10::intrusive_ptr<AllocationRef> alloc_ref,
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
  size_t block_size = signal_pad_offset + signal_pad_size;

  // 获取 SYCL/Level Zero context 和 device
  sycl::queue current_queue = at::xpu::getCurrentXPUStream().queue();
   sycl::context sycl_ctx = current_queue.get_context();
   sycl::device sycl_dev = current_queue.get_device();
   ze_context_handle_t ze_ctx =
    sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_ctx);
   ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_dev);

  std::cout << "zl_debug get context and device done " << std::endl;
  // 获取 granularity
  ze_physical_mem_desc_t phys_desc = {
      ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC, nullptr, 0, block_size};

  // 创建物理内存句柄
  ze_physical_mem_handle_t handle = nullptr;
  ze_result_t status = zePhysicalMemCreate(ze_ctx, ze_dev, &phys_desc, &handle);
  TORCH_CHECK(status == ZE_RESULT_SUCCESS, "zePhysicalMemCreate failed");

  std::cout << "zl_debug physical device memory allocation done " << std::endl;

  // 分配虚拟地址空间（只映射，不物理分配）
  void* ptr = nullptr;
  //map_block(&ptr, handle, block_size, device_idx);
  ze_device_mem_alloc_desc_t default_device_mem_alloc_desc = {
    .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
    .pNext = nullptr,
    .flags = 0,
    .ordinal = 0
};

  zeMemAllocDevice(ze_ctx, &default_device_mem_alloc_desc, size, 128, ze_dev, &ptr);

 at::Tensor xpu_tensor = at::empty({1000}, c10::TensorOptions().device(c10::kXPU).dtype(c10::kByte));

 uint8_t* raw_ptr = xpu_tensor.data_ptr<uint8_t>();
 std::cout << "zl_debug start copy to local " << std::endl;
 current_queue.memcpy(raw_ptr, ptr, 100).wait();
 std::cout << "zl_debug end copy to local " << std::endl;

 std::cout << "zl_debug map virtual to physical done " << std::endl;

  // 初始化（memset）
  //memset(ptr, 0, block_size);  // You may want zeCommandListMemset for GPU-based memset

  // 构造 Block 和 AllocationRef（假设这些结构未变）
  //auto alloc_ref = c10::make_intrusive<AllocationRef>(ptr, handle, block_size, device_idx);
  auto alloc_ref = c10::make_intrusive<AllocationRef>(ptr, ptr, block_size, device_idx);
  auto block = c10::make_intrusive<Block>(
      std::move(alloc_ref), device_idx, block_size, size, signal_pad_offset, group_name);
  std::cout << "zl_debug make block done " << std::endl;

  {
    std::unique_lock lock(mutex_);
    ptr_to_block_.emplace(ptr, std::move(block));
  }
  // check this ptr copy to sycl buffer


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

struct RendezvousRequest {
  int device_idx;
  int pid;
  size_t block_size;
  size_t buffer_size;
  size_t signal_pad_offset;
  bool has_multicast_support;
  size_t base_offset;
};

void validate_rendezvous_requests(
    const std::vector<RendezvousRequest>& reqs,
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

static bool check_group_multicast_support(
    const std::vector<RendezvousRequest>& reqs) {
  std::vector<size_t> ranks_with_multicast_support;
  for (size_t r = 0; r < reqs.size(); ++r) {
    if (reqs[r].has_multicast_support) {
      ranks_with_multicast_support.push_back(r);
    }
  }
  if (ranks_with_multicast_support.size() == reqs.size()) {
    return true;
  } else {
    // We don't expect this to happen. But we want to let the user to know if
    // this happens.
    if (ranks_with_multicast_support.size() != 0) {
      LOG(WARNING)
          << "Only a subset of ranks in the group has multicast support: "
          << ranks_with_multicast_support << " (world_size=" << reqs.size()
          << "). Skipping multicast initialization because this is unexpected.";
    }
    return false;
  }
}

//void XPUSymmetricMemoryAllocator::exchange_peer_ipc_mem(sycl::queue& queue, void* ptr)
//    {
//        // Step 1: Get base address of the pointer
//        sycl::context ctx = queue.get_context();
//        auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
//
//        void *base_addr;
//        size_t base_size;
//        ze_result_t status = zeMemGetAddressRange(l0_ctx, ptr, &base_addr, &base_size);
//        TORCH_CHECK(status == ZE_RESULT_SUCCESS, "zeMemGetAddressRange failed");
//
//        // Step 2: Get IPC mem handle from base address
//        alignas(64) exchange_contents send_buf;
//        alignas(64) exchange_contents recv_buf[world];
//
//        // fill in the exchange info
//        status = zeMemGetIpcHandle(l0_ctx, base_addr, &send_buf.ipc_handle);
//        TORCH_CHECK(status == ZE_RESULT_SUCCESS, "zeMemGetIpcHandle failed");
//        send_buf.offset = (char*)ptr - (char*)base_addr;
//        send_buf.pid = getpid();
//
//        // Step 3: Exchange the handles and offsets
//        memset(recv_buf, 0, sizeof(recv_buf));
//        // Overkill if we don't really needs all peer's handles
//        un_allgather(&send_buf, recv_buf, rank, world);
//
//        for (uint32_t i = 0; i < world; i++)
//        {
//            // Step 4: Prepare pid file descriptor of next process
//            auto* peer = recv_buf + i;
//            // Step 6: Open IPC handle of remote peer
//            auto l0_device
//                = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
//            void* peer_base;
//
//            status = zeMemOpenIpcHandle(
//                    l0_ctx, l0_device, peer->ipc_handle, ZE_IPC_MEMORY_FLAG_BIAS_CACHED, &peer_base);
//            TORCH_CHECK(status == ZE_RESULT_SUCCESS, "zeMemOpenIpcHandle failed");
//            buffers[i] = (char*)peer_base + peer->offset;
//            sync_buffer[i] = (char*)peer_base + peer->offset + data_size_per_buffer * sizeof(data_type);
//            offsets[i] = peer->offset;
//            ipc_handle[i] = send_buf.ipc_handle;
//        }
//    }

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
  // Treat empty string and std::nullopt the same as empty string seems to be
  // implicitly used that way
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

  c10::Device local_device(c10::DeviceType::XPU, block->device_idx);
  c10::DeviceGuard guard(local_device);

  // Currently, IpcChannel is using a file based socket for inter-process communication
  IpcChannel ipc_channel;
  auto group_info = get_group_info(group_name_);
  auto store = group_info.store;
  int rank = group_info.rank;
  int world_size = group_info.world_size;
  int block_fd;

  // Step 6: Open IPC handle of remote peer
  sycl::queue current_queue = at::xpu::getCurrentXPUStream().queue();
  sycl::context ctx = current_queue.get_context();
  auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
  sycl::device dev = current_queue.get_device();
  auto l0_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);

  ze_ipc_mem_handle_t ipc_handle;
  // convert to base address
  void *base_addr;
  size_t base_size;
  zeMemGetAddressRange(l0_ctx, ptr, &base_addr, &base_size);
  zeMemGetIpcHandle(l0_ctx, base_addr, &ipc_handle);
  size_t base_offset = (char*)ptr - (char*)base_addr;
  block_fd = *reinterpret_cast<int*>(&ipc_handle);

  auto local_req = RendezvousRequest{
      .device_idx = block->device_idx,
      .pid = getpid(),
      .block_size = block->block_size,
      .buffer_size = block->buffer_size,
      .signal_pad_offset = block->signal_pad_offset,
      .has_multicast_support = false,
      .base_offset = base_offset};
  auto reqs = storeExchange.all_gather(store, rank, world_size, local_req);
  validate_rendezvous_requests(reqs, world_size);

  std::vector<int> pids(world_size);
  for (int r = 0; r < world_size; ++r) {
    pids[r] = reqs[r].pid;
  }
  auto imported_fds = ipc_channel.all_gather_fds(rank, pids, block_fd);

  std::vector<HandleType> handles(world_size);
  std::vector<void*> buffers(world_size, nullptr);
  std::vector<void*> signal_pads(world_size, nullptr);

  for (int r = 0; r < world_size; ++r) {
    if (r == rank) {
      handles[r] = block->alloc_ref->handle;
      buffers[r] = ptr;
      std::cout << "zl_debug rendevous in rank = " << r  << " ptr: " <<  ptr << std::endl;
      signal_pads[r] = (void*)((uintptr_t)ptr + block->signal_pad_offset);
      continue;
    }

    ze_ipc_mem_handle_t peer_ipc_handle;
    int peer_fd = imported_fds[r];
    std::memcpy(&peer_ipc_handle, &peer_fd, sizeof(int));

    // Open IPC handle of remote peer
    void* peer_base;
    zeMemOpenIpcHandle(l0_ctx, l0_dev, peer_ipc_handle, ZE_IPC_MEMORY_FLAG_BIAS_CACHED, &peer_base);
    void* physical_buffer_ptr = (char*)peer_base + reqs[r].base_offset;
    //map_block(&buffers[r], physical_buffer_ptr, block->block_size, block->device_idx);
    buffers[r] = physical_buffer_ptr;

    //double check this buffer
    at::Tensor xpu_tensor = at::empty({1000}, c10::TensorOptions().device(c10::kXPU).dtype(c10::kByte));

     uint8_t* raw_ptr = xpu_tensor.data_ptr<uint8_t>();
     std::cout << "zl_debug start copy to local in rendevous" << std::endl;
     current_queue.memcpy(raw_ptr, physical_buffer_ptr, 100).wait();
     std::cout << "zl_debug end copy to local in rendevous in rank = " << r  << " ptr: " <<  physical_buffer_ptr << std::endl;

    int count = 256;
    auto host_ptr = (int *)sycl::malloc_host(512 * sizeof(int), current_queue);
    auto tmp_ptr = (int *)sycl::malloc_device(512 * sizeof(int), current_queue);
    std::cout << "Sync buffer content at " << address << ": ";
    current_queue.memcpy(tmp_ptr, physical_buffer_ptr, count * sizeof(int));
    current_queue.memcpy(host_ptr, tmp_ptr, count * sizeof(int));
    current_queue.wait();

    for (int i = 0; i < count; i++) {
        std::cout << host_ptr[i] << " ";
    }
     std::cout << std::flush;
     std::cout << "zl_debug print done " << std::flush;

//     at::Tensor cpu_tensor = xpu_tensor.to(c10::kCPU);
//     std::cout << "zl_debug peer rank = " << r << " data = " << cpu_tensor << std::endl;

    signal_pads[r] = (void*)((uintptr_t)buffers[r] + block->signal_pad_offset);
  }
  storeExchange.barrier(store, rank, world_size);

  HandleType mc_handle{};
  void* mc_addr = nullptr;
  bool group_has_multicast_support = check_group_multicast_support(reqs);
  //todo: not support multicast now
  std::vector<c10::intrusive_ptr<AllocationRef>> alloc_refs;
  for (int r = 0; r < world_size; ++r) {
    if (r == rank) {
      alloc_refs.emplace_back(block->alloc_ref);
      continue;
    }
    alloc_refs.push_back(c10::make_intrusive<AllocationRef>(
        buffers[r], handles[r], block->block_size, block->device_idx));
  }

  auto symm_mem = c10::make_intrusive<XPUSymmetricMemory>(
      std::move(alloc_refs),
      std::move(buffers),
      std::move(signal_pads),
      mc_handle,
      mc_addr,
      block->buffer_size,
      block->device_idx,
      group_info.rank,
      group_info.world_size);
  block->symm_mems[group_name_] = symm_mem;
  return symm_mem;
}

bool XPUSymmetricMemoryAllocator::has_multicast_support(int device_idx) {
  return device_has_multicast_support(device_idx);
}

c10::intrusive_ptr<Block> XPUSymmetricMemoryAllocator::find_block(void* ptr) {
  std::shared_lock lock(mutex_);
  auto it = ptr_to_block_.find(ptr);
  if (it == ptr_to_block_.end()) {
    return nullptr;
  }
  return it->second;
}

struct RegisterXPUSymmetricMemoryAllocator {
  RegisterXPUSymmetricMemoryAllocator() {
      register_allocator(
      c10::DeviceType::XPU,
      c10::make_intrusive<XPUSymmetricMemoryAllocator>());
  }
};

static RegisterXPUSymmetricMemoryAllocator register_allocator_;

} // namespace symmetric_memory
} // namespace c10d
