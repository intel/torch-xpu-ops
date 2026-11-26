#include <xccl/XPUSymmetricMemory.hpp>
#include <xccl/XPUSymmetricMemoryUtils.hpp>

#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>

// Requires Intel oneAPI DPC++/C++ Compiler >= 2026.0 for
// <sycl/ext/oneapi/experimental/ipc_memory.hpp>.

#include <xccl/ProcessGroupXCCL.hpp>

#include <ATen/Context.h> // Residency workaround only
#include <ATen/ceil_div.h>
#include <ATen/xpu/XPUContext.h>
#include <ATen/xpu/level_zero_stub/ATenLevelZero.h> // Residency workaround only
#include <c10/core/DeviceGuard.h>
#include <c10/util/error.h>
#include <c10/xpu/XPUCachingAllocator.h>

#include <sys/prctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <mutex>

namespace c10d {
namespace symmetric_memory {

namespace {

std::atomic<uint64_t> store_exchange_nonce{0};

thread_local StoreExchange storeExchange = []() {
  const auto nonce =
      store_exchange_nonce.fetch_add(1, std::memory_order_relaxed);
  return StoreExchange("XPUSymmetricMemory_" + std::to_string(nonce));
}();

} // namespace

AllocationRef::AllocationRef(
    void* ptr,
    HandleType handle,
    size_t block_size,
    int device_idx,
    bool local_allocation)
    : ptr(ptr),
      handle(handle),
      block_size(block_size),
      device_idx(device_idx),
      local_allocation(local_allocation) {}

#if XPU_SYMM_MEM_PHYSICAL_IPC
AllocationRef::AllocationRef(
    void* ptr,
    size_t block_size,
    int device_idx,
    bool local_allocation,
    sycl::ext::oneapi::experimental::physical_mem physmem)
    : ptr(ptr),
      handle(ptr),
      block_size(block_size),
      device_idx(device_idx),
      local_allocation(local_allocation),
      physmem(std::move(physmem)) {}
#endif

AllocationRef::~AllocationRef() {
  if (is_finalizing()) {
    return;
  }
#if XPU_SYMM_MEM_PHYSICAL_IPC
  if (physmem) {
    // This process owns the mapping, whether the memory is local or imported.
    namespace syclexp = sycl::ext::oneapi::experimental;
    c10::Device local_device(c10::DeviceType::XPU, device_idx);
    c10::DeviceGuard guard(local_device);
    c10::xpu::syncStreamsOnDevice();
    sycl::context ctx = physmem->get_context();
    syclexp::unmap(ptr, block_size, ctx);
    syclexp::free_virtual_mem(
        reinterpret_cast<uintptr_t>(ptr), block_size, ctx);
    return;
  }
#endif
  // Currently, we cannot free virtual memory exchanged from other device.
  // (SYCL `ipc_memory::close` is available but calling it during teardown
  // has been observed to hang on this stack; match the original L0 path
  // which also skips remote unmap.)
  if (!local_allocation) {
    return;
  }
  c10::Device local_device(c10::DeviceType::XPU, device_idx);
  c10::DeviceGuard guard(local_device);
  c10::xpu::syncStreamsOnDevice();
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();
  sycl::free(ptr, queue);
}

XPUPeerAllocInfo::XPUPeerAllocInfo(
    std::vector<c10::intrusive_ptr<AllocationRef>> alloc_refs,
    std::vector<void*> buffers,
    std::vector<void*> signal_pads,
    size_t buffer_size,
    int local_device_idx,
    int rank,
    int world_size,
    std::string group_name)
    : alloc_refs_(std::move(alloc_refs)),
      buffers_(std::move(buffers)),
      signal_pads_(std::move(signal_pads)),
      buffer_size_(buffer_size),
      local_device_idx_(local_device_idx),
      rank_(rank),
      world_size_(world_size),
      group_name_(std::move(group_name)) {
  const size_t arr_size = sizeof(void*) * world_size_;
  buffers_dev_ = reinterpret_cast<void**>(
      c10::xpu::XPUCachingAllocator::raw_alloc(arr_size));
  signal_pads_dev_ = reinterpret_cast<void**>(
      c10::xpu::XPUCachingAllocator::raw_alloc(arr_size));

  c10::Device local_device(c10::DeviceType::XPU, local_device_idx);
  c10::DeviceGuard guard(local_device);

  auto& queue = at::xpu::getCurrentXPUStream().queue();
  queue.memcpy(buffers_dev_, buffers_.data(), arr_size);
  queue.memcpy(signal_pads_dev_, signal_pads_.data(), arr_size);
  queue.wait();
}

// This is mostly a shallow copy that shares the pointer to `XPUPeerAllocInfo`
// which corresponds to the base Block. The XPUSymmetricMemory handle is
// specified by the offset to the base ptr.
XPUSymmetricMemory::XPUSymmetricMemory(
    const c10::intrusive_ptr<XPUPeerAllocInfo>& pai,
    size_t offset)
    : local_device_idx_(pai->local_device_idx_),
      rank_(pai->rank_),
      world_size_(pai->world_size_),
      pai_(pai),
      offset_(offset) {
  // offset is specific per symm_mem handle
  TORCH_INTERNAL_ASSERT(offset_ < pai_->buffer_size_, "offset out of range");
}

std::vector<void*> XPUSymmetricMemory::get_buffer_ptrs() {
  return pai_->buffers_;
}

std::vector<void*> XPUSymmetricMemory::get_signal_pad_ptrs() {
  return pai_->signal_pads_;
}

void** XPUSymmetricMemory::get_buffer_ptrs_dev() {
  return pai_->buffers_dev_;
}

void** XPUSymmetricMemory::get_signal_pad_ptrs_dev() {
  return pai_->signal_pads_dev_;
}

size_t XPUSymmetricMemory::get_buffer_size() {
  return pai_->buffer_size_;
}

size_t XPUSymmetricMemory::get_offset() {
  return offset_;
}

bool XPUSymmetricMemory::has_multicast_support() {
  return false;
}

void* XPUSymmetricMemory::get_multicast_ptr() {
  return nullptr;
}

void check_channel(int channel, int world_size) {
  TORCH_CHECK(
      channel >= 0,
      "channel for barrier(), put_signal() and wait_signal() ",
      "must be greater than or equal to 0 (got ",
      channel,
      ")");
  const size_t num_channels =
      get_signal_pad_size() / sizeof(uint32_t) / world_size;
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

  c10::Device local_device(c10::DeviceType::XPU, local_device_idx_);
  c10::DeviceGuard guard(local_device);

  auto stream = at::xpu::getCurrentXPUStream();
  barrier_impl_xpu(
      reinterpret_cast<uint32_t**>(pai_->signal_pads_dev_),
      channel,
      rank_,
      world_size_,
      timeout_ms,
      stream);
  return;
}

void XPUSymmetricMemory::put_signal(
    int dst_rank,
    int channel,
    size_t timeout_ms) {
  check_channel(channel, world_size_);

  c10::Device local_device(c10::DeviceType::XPU, local_device_idx_);
  c10::DeviceGuard guard(local_device);
  auto stream = at::xpu::getCurrentXPUStream();

  put_signal_impl_xpu(
      reinterpret_cast<uint32_t**>(pai_->signal_pads_dev_),
      dst_rank,
      channel,
      rank_,
      world_size_,
      timeout_ms,
      stream);
}

void XPUSymmetricMemory::wait_signal(
    int src_rank,
    int channel,
    size_t timeout_ms) {
  check_channel(channel, world_size_);

  c10::Device local_device(c10::DeviceType::XPU, local_device_idx_);
  c10::DeviceGuard guard(local_device);
  auto stream = at::xpu::getCurrentXPUStream();

  wait_signal_impl_xpu(
      reinterpret_cast<uint32_t**>(pai_->signal_pads_dev_),
      src_rank,
      channel,
      rank_,
      world_size_,
      timeout_ms,
      stream);
}

int XPUSymmetricMemory::get_rank() {
  return rank_;
}

int XPUSymmetricMemory::get_world_size() {
  return world_size_;
}

c10::Device XPUSymmetricMemory::get_device() {
  return c10::Device(c10::DeviceType::XPU, local_device_idx_);
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

#if XPU_SYMM_MEM_PHYSICAL_IPC
Block::~Block() {
  // The only release of the exported handle: rendezvous() reuses it in every
  // group, since the block cannot be exported again after a put.
  if (exported_handle && !is_finalizing()) {
    sycl::ext::oneapi::experimental::ipc::physical_memory::put(
        *exported_handle, alloc_ref->physmem->get_context());
  }
}
#endif

void* XPUSymmetricMemoryAllocator::alloc(
    size_t size,
    int device_idx,
    const std::optional<std::string>& group_name) {
  size_t signal_pad_offset = at::round_up(size, 16UL);
  size_t block_size = signal_pad_offset + get_signal_pad_size();

  c10::DeviceGuard device_guard(c10::Device(c10::DeviceType::XPU, device_idx));
  sycl::queue& current_queue = at::xpu::getCurrentXPUStream().queue();
  // Allocate directly from SYCL runtime instead of XPUCachingAllocator:
  // 1) keep behavior aligned with CUDA symmetric-memory implementation;
  // 2) avoid allocator-level expandable-memory remapping side effects on
  //    exchanged IPC handles/addresses;
  // 3) preserve flexibility for future handle-based features (for example,
  //    reconstructing multicast objects from physical/shared handles).
  void* ptr = nullptr;
  c10::intrusive_ptr<AllocationRef> alloc_ref;
#if XPU_SYMM_MEM_PHYSICAL_IPC
  // Each block is its own physical memory object, which peers import and map
  // themselves, as the CUDA backend does with cuMemCreate. Unlike USM shared
  // through ipc::memory, the importer's mapping is usable by kernels that load
  // the peer address from device memory, such as the signal kernels.
  sycl::device dev = current_queue.get_device();
  if (dev.has(sycl::aspect::ext_oneapi_ipc_physical_memory)) {
    namespace syclexp = sycl::ext::oneapi::experimental;
    sycl::context ctx = current_queue.get_context();
    // Work around a SYCL runtime bug (GSD-14390):
    // sycl_ext_oneapi_virtual_mem allows any multiple of get_mem_granularity,
    // but on Level Zero, sizes of 2 MiB and more must also be a multiple of
    // 2 MiB, the page size Level Zero uses for them.
    constexpr size_t kLargePageSize = 2 * 1024 * 1024;
    block_size =
        at::round_up(block_size, syclexp::get_mem_granularity(dev, ctx));
    if (block_size >= kLargePageSize) {
      block_size = at::round_up(block_size, kLargePageSize);
    }
    syclexp::physical_mem physmem(
        dev, ctx, block_size, syclexp::properties{syclexp::enable_ipc{}});
    uintptr_t va = syclexp::reserve_virtual_mem(block_size, ctx);
    ptr = physmem.map(va, block_size, syclexp::address_access_mode::read_write);
    alloc_ref = c10::make_intrusive<AllocationRef>(
        ptr, block_size, device_idx, true, std::move(physmem));
  }
#endif
  if (!ptr) {
    ptr = sycl::malloc_device(block_size, current_queue);
    alloc_ref = c10::make_intrusive<AllocationRef>(
        ptr, ptr, block_size, device_idx, true);
  }
  current_queue.memset(ptr, 0, block_size).wait();
  auto block = c10::make_intrusive<Block>(
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

struct RendezvousRequest {
  int device_idx;
  int pid;
  size_t block_size;
  size_t buffer_size;
  size_t signal_pad_offset;
  bool has_multicast_support;
  bool physical_ipc;
};

void validate_rendezvous_requests(
    const std::vector<RendezvousRequest>& reqs,
    int world_size) {
  TORCH_CHECK(reqs.size() == (size_t)world_size);

  std::unordered_set<int> device_indices;
  device_indices.reserve(world_size);
  for (const auto& req : reqs) {
    device_indices.insert(req.device_idx);
  }

  for (int r = 1; r < world_size; ++r) {
    TORCH_CHECK(
        reqs[r].physical_ipc == reqs[0].physical_ipc,
        "XPUSymmetricMemory::rendezvous: rank 0 shares its block as ",
        reqs[0].physical_ipc ? "physical memory" : "USM",
        " but rank ",
        r,
        " as ",
        reqs[r].physical_ipc ? "physical memory" : "USM",
        ". All ranks must support ext_oneapi_ipc_physical_memory, or none.");
    TORCH_CHECK(reqs[r].block_size == reqs[0].block_size);
    TORCH_CHECK(reqs[r].buffer_size == reqs[0].buffer_size);
    TORCH_CHECK(reqs[r].signal_pad_offset == reqs[0].signal_pad_offset);
  }
}

c10::intrusive_ptr<SymmetricMemory> XPUSymmetricMemoryAllocator::rendezvous(
    void* ptr,
    const std::optional<std::string>& group_name) {
  // In case of MemPool, the `ptr` passed in (i.e. tensor storage ptr) may not
  // be the same as the allocation base pointer, so we need to find the block
  // that covers the `ptr`
  size_t offset = 0;
  auto block = find_block_covering(ptr, offset);
  if (block == nullptr) {
    TORCH_WARN(
        "Pointer not within any SymmetricMemory allocation, "
        "is the tensor allocated from SymmetricMemory?");
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

  // If found, this block has been rendezvous'd by the given group
  auto it = block->symm_mems.find(group_name_);
  if (it != block->symm_mems.end()) {
    // Create symm mem handle for this tensor, specified by its offset
    return c10::make_intrusive<XPUSymmetricMemory>(it->second, offset);
  }

  c10::Device local_device(c10::DeviceType::XPU, block->device_idx);
  c10::DeviceGuard guard(local_device);

  auto group = resolve_process_group(group_name_);
  TORCH_CHECK(
      group != nullptr,
      "XPUSymmetricMemory::rendezvous: Could not resolve process group '",
      group_name_,
      "'. This can happen if rendezvous() is called before the process "
      "group is initialized or if the group name is incorrect.");
  auto rank = group->getRank();
  auto world_size = group->getSize();
  auto store = group->getStore();
  sycl::queue& current_queue = at::xpu::getCurrentXPUStream().queue();

  // SYCL/L0 IPC import uses `pidfd_getfd` between peer processes, which
  // requires PTRACE_MODE_ATTACH_REALCREDS on the target pid. With Yama
  // (/proc/sys/kernel/yama/ptrace_scope >= 1, the default on most distros),
  // only ancestor processes or those explicitly whitelisted via
  // PR_SET_PTRACER can attach. Sibling ranks spawned by a launcher are
  // neither ancestors nor descendants of each other, so
  // PR_SET_PTRACER(getppid()) is NOT sufficient -- it only authorizes the
  // launcher. We need PR_SET_PTRACER_ANY so any peer rank can import our
  // IPC handles. This is scoped to the lifetime of this process.
  static c10::once_flag prctl_once;
  c10::call_once(prctl_once, []() {
    (void)::prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
  });

  // Blocks allocated as physical memory are shared as physical memory objects;
  // each peer imports and maps them itself (see alloc()).
#if XPU_SYMM_MEM_PHYSICAL_IPC
  const bool physical_ipc = block->alloc_ref->physmem.has_value();
#else
  const bool physical_ipc = false;
#endif
  auto local_req = RendezvousRequest{
      .device_idx = block->device_idx,
      .pid = getpid(),
      .block_size = block->block_size,
      .buffer_size = block->buffer_size,
      .signal_pad_offset = block->signal_pad_offset,
      .has_multicast_support = false,
      .physical_ipc = physical_ipc};
  auto reqs = storeExchange.all_gather(store, rank, world_size, local_req);
  validate_rendezvous_requests(reqs, world_size);

  // Step 1: Get SYCL experimental IPC handle from the allocation base.
  // `ptr` may point into the middle of the block (MemPool case), so the handle
  // and all peer pointers are derived from the block's base address; the
  // per-tensor offset is carried by the XPUSymmetricMemory handle instead.
  void* base_ptr = block->alloc_ref->ptr;
  sycl::context ctx = current_queue.get_context();
  sycl::device dev = current_queue.get_device();

  // ipc_memory is deprecated in favor of ipc::memory since oneAPI 2026.2,
  // which older toolchains lack; keep ipc_memory until 2026.1 is dropped.
  C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated-declarations")
  namespace syclexp = sycl::ext::oneapi::experimental;
#if XPU_SYMM_MEM_PHYSICAL_IPC
  std::vector<c10::intrusive_ptr<AllocationRef>> peer_refs(world_size);
#endif
  std::vector<std::byte> local_handle_bytes;
#if XPU_SYMM_MEM_PHYSICAL_IPC
  if (physical_ipc) {
    // Export once per block and reuse the handle in every group. Releasing it
    // after a rendezvous and exporting again would give a peer that imports
    // this block for the first time, in a later group, a handle it cannot
    // open. The handle is released in ~Block().
    if (!block->exported_handle) {
      block->exported_handle =
          syclexp::ipc::physical_memory::get(*block->alloc_ref->physmem);
    }
    local_handle_bytes = block->exported_handle->data();
  }
#endif
  if (!physical_ipc) {
    local_handle_bytes = syclexp::ipc_memory::get(base_ptr, ctx).data();
  }
  std::vector<uint8_t> local_payload(
      reinterpret_cast<const uint8_t*>(local_handle_bytes.data()),
      reinterpret_cast<const uint8_t*>(local_handle_bytes.data()) +
          local_handle_bytes.size());

  // Step 2: Exchange IPC-handle bytes via store (variable-length payload).
  auto peer_handle_payloads =
      storeExchange.all_gather_bytes(store, rank, world_size, local_payload);

  // Step 3: Open peer IPC handles via SYCL API.
  std::vector<HandleType> handles(world_size);
  std::vector<void*> buffers(world_size, nullptr);
  std::vector<void*> signal_pads(world_size, nullptr);

  for (int r = 0; r < world_size; ++r) {
    if (r == rank) {
      handles[r] = base_ptr;
      buffers[r] = base_ptr;
      signal_pads[r] = (void*)((uintptr_t)base_ptr + block->signal_pad_offset);
      continue;
    }

    const auto& payload = peer_handle_payloads[r];
    syclexp::ipc_memory::handle_data_t peer_bytes(
        reinterpret_cast<const std::byte*>(payload.data()),
        reinterpret_cast<const std::byte*>(payload.data()) + payload.size());
    void* remote_base = nullptr;
#if XPU_SYMM_MEM_PHYSICAL_IPC
    if (physical_ipc) {
      // Import for the local device: imports for the exporting device are not
      // reliably visible to kernels on the local device with current drivers.
      syclexp::physical_mem physmem =
          syclexp::ipc::physical_memory::open(peer_bytes, ctx, dev);
      uintptr_t va = syclexp::reserve_virtual_mem(block->block_size, ctx);
      remote_base = physmem.map(
          va, block->block_size, syclexp::address_access_mode::read_write);
      peer_refs[r] = c10::make_intrusive<AllocationRef>(
          remote_base,
          block->block_size,
          block->device_idx,
          false,
          std::move(physmem));
      handles[r] = remote_base;
      buffers[r] = remote_base;
      signal_pads[r] =
          static_cast<char*>(remote_base) + block->signal_pad_offset;
      continue;
    }
#endif
    remote_base = syclexp::ipc_memory::open(peer_bytes, ctx, dev);
    // L0 does not make an imported IPC allocation resident on this device
    // until a kernel takes it as a direct argument. The signal kernels only
    // reach peer memory through pointers loaded from signal_pads_dev_, so
    // without this their first access page-faults and the device is lost.
    // This relies on opening peer memory on this device: opened on the
    // exporting device, residency calls on later blocks of a pooled
    // allocation map the first block's memory (driver bug, GSD-14207).
    // TODO: Remove this workaround, and the includes marked for it, once
    // oneAPI n-1 is 2026.2 or later, so that every build shares symmetric
    // memory as physical memory, or once IPC imports are usable through
    // indirect access.
    ze_result_t ze_status =
        at::globalContext().getLevelZero().zeContextMakeMemoryResident(
            sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx),
            sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev),
            remote_base,
            block->block_size);
    TORCH_CHECK(
        ze_status == ZE_RESULT_SUCCESS,
        "zeContextMakeMemoryResident failed for peer ",
        r,
        " with error ",
        ze_status);

    handles[r] = remote_base;
    buffers[r] = remote_base;
    signal_pads[r] = (void*)((uintptr_t)remote_base + block->signal_pad_offset);
  }
  C10_DIAGNOSTIC_POP()
  storeExchange.barrier(store, rank, world_size);

  std::vector<c10::intrusive_ptr<AllocationRef>> alloc_refs;
  for (int r = 0; r < world_size; ++r) {
    if (r == rank) {
      alloc_refs.emplace_back(block->alloc_ref);
      continue;
    }
#if XPU_SYMM_MEM_PHYSICAL_IPC
    if (peer_refs[r]) {
      alloc_refs.push_back(peer_refs[r]);
      continue;
    }
#endif
    alloc_refs.push_back(c10::make_intrusive<AllocationRef>(
        buffers[r], handles[r], block->block_size, block->device_idx, false));
  }

  auto pai = c10::make_intrusive<XPUPeerAllocInfo>(
      std::move(alloc_refs),
      std::move(buffers),
      std::move(signal_pads),
      block->buffer_size,
      block->device_idx,
      rank,
      world_size,
      group_name_);
  // Cache it with the group name
  block->symm_mems[group_name_] = pai;
  // Create symm mem handle for this tensor, specified by its offset
  return c10::make_intrusive<XPUSymmetricMemory>(pai, offset);
}

bool XPUSymmetricMemoryAllocator::has_multicast_support(int device_idx) {
  return false;
}

c10::DeviceType XPUSymmetricMemoryAllocator::supported_device_type() {
  return c10::DeviceType::XPU;
}

std::string XPUSymmetricMemoryAllocator::name() {
  return "XPU";
}

bool XPUSymmetricMemoryAllocator::has_allocation(void* ptr) {
  size_t offset = 0;
  return find_block_covering(ptr, offset) != nullptr;
}

c10::intrusive_ptr<Block> XPUSymmetricMemoryAllocator::find_block(void* ptr) {
  std::shared_lock lock(mutex_);
  auto it = ptr_to_block_.find(ptr);
  if (it == ptr_to_block_.end()) {
    return nullptr;
  }
  return it->second;
}

/* Search for a block that covers the given ptr, and write back the offset to
 * the base ptr; return nullptr if not found */
c10::intrusive_ptr<Block> XPUSymmetricMemoryAllocator::find_block_covering(
    void* ptr,
    size_t& offset) {
  std::shared_lock lock(mutex_);
  // In case of MemPool, tensor.storage().data_ptr() may not match exactly an
  // allocation's base address. Thus we perform the search by testing if the
  // former is within an allocation's range.
  auto ptr_int = reinterpret_cast<uintptr_t>(ptr);
  for (const auto& pair : ptr_to_block_) {
    // pair.first is the allocation base, which on XPU is also the start of the
    // user buffer (the signal pad sits at the end of the block).
    auto base_int = reinterpret_cast<uintptr_t>(pair.first);
    if (ptr_int < base_int) {
      continue;
    }
    auto candidate = ptr_int - base_int;
    if (candidate < pair.second->buffer_size) {
      offset = candidate;
      return pair.second;
    }
  }
  return nullptr;
}

struct RegisterXPUSymmetricMemoryAllocator {
  RegisterXPUSymmetricMemoryAllocator() {
    auto allocator = c10::make_intrusive<XPUSymmetricMemoryAllocator>();
    // Always register availability to support dynamic backend switching
    register_availability("XPU", allocator);
    // If this is the preferred backend, also set it as default
    if (getSymmMemBackendXPU() == "XPU") {
      register_allocator(c10::DeviceType::XPU, allocator);
    }
  }
};
static RegisterXPUSymmetricMemoryAllocator register_allocator_;

} // namespace symmetric_memory
} // namespace c10d
