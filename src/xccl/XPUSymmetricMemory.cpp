#include <xccl/ProcessGroupXCCL.hpp>
#include <xccl/XPUSymmetricMemory.hpp>
#include <xccl/XPUSymmetricMemoryUtils.hpp>

#include <ATen/ceil_div.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/error.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>

#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>

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

AllocationRef::~AllocationRef() {
  if (is_finalizing()) {
    return;
  }
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

  auto& queue = at::xpu::getCurrentXPUStream().queue();
  queue.memcpy(buffers_dev_, buffers_.data(), arr_size);
  queue.memcpy(signal_pads_dev_, signal_pads_.data(), arr_size);
  queue.wait();
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

size_t XPUSymmetricMemory::get_offset() {
  return 0;
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
      reinterpret_cast<uint32_t**>(signal_pads_dev_),
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
      reinterpret_cast<uint32_t**>(signal_pads_dev_),
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
      reinterpret_cast<uint32_t**>(signal_pads_dev_),
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
  void* ptr = sycl::malloc_device(block_size, current_queue);
  current_queue.memset(ptr, 0, block_size);
  auto alloc_ref = c10::make_intrusive<AllocationRef>(
      ptr, ptr, block_size, device_idx, true);
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

  auto local_req = RendezvousRequest{
      .device_idx = block->device_idx,
      .pid = getpid(),
      .block_size = block->block_size,
      .buffer_size = block->buffer_size,
      .signal_pad_offset = block->signal_pad_offset,
      .has_multicast_support = false};
  auto reqs = storeExchange.all_gather(store, rank, world_size, local_req);
  validate_rendezvous_requests(reqs, world_size);

  // Step 1: Get SYCL experimental IPC handle from the allocation base.
  // `ptr` is always a base pointer here: alloc() stores ptr_to_block_[ptr]
  // keyed by the malloc_device return value, and find_block() does an exact
  // lookup, so by the time rendezvous() is called we already have the base.
  sycl::context ctx = current_queue.get_context();
  sycl::device dev = current_queue.get_device();

  namespace syclexp = sycl::ext::oneapi::experimental;
  syclexp::ipc_memory::handle local_handle = syclexp::ipc_memory::get(ptr, ctx);
  syclexp::ipc_memory::handle_data_t local_handle_bytes = local_handle.data();
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
      handles[r] = ptr;
      buffers[r] = ptr;
      signal_pads[r] = (void*)((uintptr_t)ptr + block->signal_pad_offset);
      continue;
    }

    const auto& payload = peer_handle_payloads[r];
    syclexp::ipc_memory::handle_data_t peer_bytes(
        reinterpret_cast<const std::byte*>(payload.data()),
        reinterpret_cast<const std::byte*>(payload.data()) + payload.size());
    void* remote_base = syclexp::ipc_memory::open(peer_bytes, ctx, dev);

    handles[r] = remote_base;
    buffers[r] = remote_base;
    signal_pads[r] = (void*)((uintptr_t)remote_base + block->signal_pad_offset);
  }
  storeExchange.barrier(store, rank, world_size);

  HandleType mc_handle{};
  void* mc_addr = nullptr;

  std::vector<c10::intrusive_ptr<AllocationRef>> alloc_refs;
  for (int r = 0; r < world_size; ++r) {
    if (r == rank) {
      alloc_refs.emplace_back(block->alloc_ref);
      continue;
    }
    alloc_refs.push_back(c10::make_intrusive<AllocationRef>(
        buffers[r], handles[r], block->block_size, block->device_idx, false));
  }

  auto symm_mem = c10::make_intrusive<XPUSymmetricMemory>(
      std::move(alloc_refs),
      std::move(buffers),
      std::move(signal_pads),
      mc_handle,
      mc_addr,
      block->buffer_size,
      block->device_idx,
      rank,
      world_size);
  symm_mem->set_group_name(group_name_);
  block->symm_mems[group_name_] = symm_mem;
  return symm_mem;
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
