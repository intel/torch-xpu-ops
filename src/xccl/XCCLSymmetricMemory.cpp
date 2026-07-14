#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <xccl/ProcessGroupXCCL.hpp>
#include <xccl/XPUSymmetricMemoryUtils.hpp>

#include <ATen/ceil_div.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/util/error.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <comm/SYCLContext.h>

#include <oneapi/ccl.hpp>

#include <algorithm>
#include <map>
#include <mutex>

namespace c10d {
namespace symmetric_memory {

static StoreExchange storeExchange = StoreExchange("XCCLSymmetricMemory");

// Resolve the ccl::communicator that the XCCL ProcessGroup has created for
// `device_idx` within `group_name`. The XCCL symmetric-memory backend reuses
// the XCCL backend's communicator rather than bootstrapping its own, so the
// XCCL comm must already exist (i.e. a collective has run on the group, or the
// backend was eagerly initialized via `device_id` in `init_process_group`).
static ccl::communicator& getCclComm(
    const std::string& group_name,
    int device_idx) {
  auto group = resolve_process_group(group_name);
  auto backend = group->getBackend(c10::DeviceType::XPU);
  auto* pg = dynamic_cast<ProcessGroupXCCL*>(backend.get());
  TORCH_CHECK(
      pg != nullptr,
      "XCCL symmetric memory requires the XCCL backend, but the process "
      "group '",
      group_name,
      "' does not use it.");
  auto comm = pg->getXCCLComm(std::to_string(device_idx));
  TORCH_CHECK(
      comm != nullptr && comm->cclComm.has_value(),
      "XCCL symmetric memory: the XCCL communicator for device ",
      device_idx,
      " is not initialized yet. Run a collective on group '",
      group_name,
      "' first, or eagerly initialize the backend by passing `device_id` to "
      "`init_process_group`.");
  return comm->cclComm.value();
}

// Owns a combined oneCCL device allocation. Layout:
//   [0, round_up(buffer_size, 16))                 - user data buffer
//   [round_up(buffer_size, 16), total_size)        - signal pad
// A single ccl::mem_alloc region backs both, so a single window registration
// (done lazily at rendezvous) covers the whole thing.
struct XCCLAllocation {
  void* ptr;
  size_t buffer_size; // user-requested buffer size in bytes
  size_t signal_pad_size;
  int device_idx;

  XCCLAllocation(
      void* ptr,
      size_t buffer_size,
      size_t signal_pad_size,
      int device_idx)
      : ptr(ptr),
        buffer_size(buffer_size),
        signal_pad_size(signal_pad_size),
        device_idx(device_idx) {}

  // Non-copyable / non-movable to avoid a double free of `ptr`.
  XCCLAllocation(const XCCLAllocation&) = delete;
  XCCLAllocation& operator=(const XCCLAllocation&) = delete;
  XCCLAllocation(XCCLAllocation&&) = delete;
  XCCLAllocation& operator=(XCCLAllocation&&) = delete;

  ~XCCLAllocation() {
    if (is_finalizing()) {
      return;
    }
    c10::OptionalDeviceGuard guard;
    guard.reset_device(at::Device(at::DeviceType::XPU, device_idx));
    auto& queue = at::xpu::getCurrentSYCLQueue();
    auto ccl_stream = ccl::create_stream(queue);
    ccl::mem_free(ccl_stream, ptr);
  }
};

// Holds the per-group window registration and the resolved peer buffer / signal
// pad pointers for one allocation. Shared by all `XCCLSymmetricMemory` handles
// that view the same allocation under the same group (they differ only in
// offset).
class XCCLPeerAllocInfo : public c10::intrusive_ptr_target {
 public:
  XCCLPeerAllocInfo(XCCLAllocation* allocation, std::string group_name)
      : buffer_size_(allocation->buffer_size),
        device_idx_(allocation->device_idx),
        group_name_(std::move(group_name)) {
    c10::OptionalDeviceGuard guard;
    guard.reset_device(at::Device(at::DeviceType::XPU, device_idx_));

    auto group = resolve_process_group(group_name_);
    rank_ = group->getRank();
    world_size_ = group->getSize();

    ccl::communicator& comm = getCclComm(group_name_, device_idx_);

    const size_t aligned_buffer_size = at::round_up(buffer_size_, 16UL);
    const size_t total_size = aligned_buffer_size + allocation->signal_pad_size;

    // Register a single window over the combined buffer + signal pad region.
    win_ = ccl::comm_window_register(
        comm, allocation->ptr, total_size, CCL_WIN_COLL_SYMMETRIC);
    TORCH_CHECK(win_ != nullptr, "ccl::comm_window_register failed");

    // Resolve each peer's buffer pointer within the window. oneCCL returns
    // nullptr for peers that are not load/store accessible (e.g. over network).
    buffers_.resize(world_size_);
    signal_pads_.resize(world_size_);
    world_within_xpu_p2p_ = true;
    for (int r = 0; r < world_size_; ++r) {
      ccl::get_peer_device_pointer(win_, 0, r, &buffers_[r]);
      if (buffers_[r] == nullptr) {
        world_within_xpu_p2p_ = false;
        signal_pads_[r] = nullptr;
      } else {
        // All ranks use the same aligned_buffer_size, so a peer's signal pad
        // is simply its buffer base plus that offset.
        signal_pads_[r] =
            static_cast<char*>(buffers_[r]) + aligned_buffer_size;
      }
    }

    // Mirror the peer pointer arrays into device memory for kernel-side use.
    const size_t arr_size = sizeof(void*) * world_size_;
    auto& queue = at::xpu::getCurrentSYCLQueue();
    buffers_dev_ = reinterpret_cast<void**>(
        c10::xpu::XPUCachingAllocator::raw_alloc(arr_size));
    signal_pads_dev_ = reinterpret_cast<void**>(
        c10::xpu::XPUCachingAllocator::raw_alloc(arr_size));
    queue.memcpy(buffers_dev_, buffers_.data(), arr_size).wait();
    queue.memcpy(signal_pads_dev_, signal_pads_.data(), arr_size).wait();
  }

  XCCLPeerAllocInfo(const XCCLPeerAllocInfo&) = delete;
  XCCLPeerAllocInfo& operator=(const XCCLPeerAllocInfo&) = delete;
  XCCLPeerAllocInfo(XCCLPeerAllocInfo&&) = delete;
  XCCLPeerAllocInfo& operator=(XCCLPeerAllocInfo&&) = delete;

  ~XCCLPeerAllocInfo() override {
    if (is_finalizing()) {
      return;
    }
    c10::OptionalDeviceGuard guard;
    guard.reset_device(at::Device(at::DeviceType::XPU, device_idx_));

    if (win_ != nullptr) {
      // Best effort: the communicator may already be torn down at process
      // exit, in which case deregistration is unnecessary.
      try {
        ccl::communicator& comm = getCclComm(group_name_, device_idx_);
        ccl::comm_window_deregister(comm, win_); // sets win_ to nullptr
      } catch (const std::exception&) {
      }
      win_ = nullptr;
    }
    if (buffers_dev_ != nullptr) {
      c10::xpu::XPUCachingAllocator::raw_delete(
          static_cast<void*>(buffers_dev_));
      buffers_dev_ = nullptr;
    }
    if (signal_pads_dev_ != nullptr) {
      c10::xpu::XPUCachingAllocator::raw_delete(
          static_cast<void*>(signal_pads_dev_));
      signal_pads_dev_ = nullptr;
    }
  }

 private:
  size_t buffer_size_;
  int device_idx_;
  std::string group_name_;
  int rank_{0};
  int world_size_{0};
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  void** buffers_dev_ = nullptr;
  void** signal_pads_dev_ = nullptr;
  ccl::window_t win_ = nullptr;
  bool world_within_xpu_p2p_ = false;

  friend class XCCLSymmetricMemory;
};

class XCCLSymmetricMemory : public SymmetricMemory {
 public:
  XCCLSymmetricMemory(
      c10::intrusive_ptr<XCCLPeerAllocInfo> pai,
      size_t offset)
      : pai_(std::move(pai)), offset_(offset) {}

  ~XCCLSymmetricMemory() override = default;

  std::vector<void*> get_buffer_ptrs() override {
    return pai_->buffers_;
  }

  std::vector<void*> get_signal_pad_ptrs() override {
    return pai_->signal_pads_;
  }

  void** get_buffer_ptrs_dev() override {
    return pai_->buffers_dev_;
  }

  void** get_signal_pad_ptrs_dev() override {
    return pai_->signal_pads_dev_;
  }

  size_t get_buffer_size() override {
    return pai_->buffer_size_;
  }

  size_t get_offset() override {
    return offset_;
  }

  bool has_multicast_support() override {
    // oneCCL does not expose a multicast (multimem) device pointer today.
    return false;
  }

  void* get_multicast_ptr() override {
    return nullptr;
  }

  void barrier(int channel, size_t timeout_ms) override {
    // TODO: implement via ccl::LsaBarrierSession once signals are enabled.
  }

  void put_signal(int dst_rank, int channel, size_t timeout_ms) override {
    // TODO
  }

  void wait_signal(int src_rank, int channel, size_t timeout_ms) override {
    // TODO
  }

  int get_rank() override {
    return pai_->rank_;
  }

  int get_world_size() override {
    return pai_->world_size_;
  }

  c10::Device get_device() override {
    return c10::Device(c10::DeviceType::XPU, pai_->device_idx_);
  }

  bool world_within_direct_access() override {
    return pai_->world_within_xpu_p2p_;
  }

  // XCCL-specific accessor: the registered oneCCL window backing this
  // allocation. Exposed for symmetric-memory ops that call the device API.
  ccl::window_t get_window() {
    return pai_->win_;
  }

 private:
  c10::intrusive_ptr<XCCLPeerAllocInfo> pai_;
  size_t offset_{0}; // in bytes

  friend class XCCLSymmetricMemoryAllocator;
};

class XCCLSymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(
      size_t size,
      int device_idx,
      const std::optional<std::string>& group_name) override {
    TORCH_CHECK(
        !group_name.has_value(),
        "XCCLSymmetricMemoryAllocator::alloc must not be called with a "
        "group_name");
    c10::OptionalDeviceGuard guard;
    guard.reset_device(at::Device(at::DeviceType::XPU, device_idx));

    const size_t aligned_buffer_size = at::round_up(size, 16UL);
    const size_t signal_pad_size = get_signal_pad_size();
    const size_t total_size = aligned_buffer_size + signal_pad_size;

    auto& queue = at::xpu::getCurrentSYCLQueue();
    auto ccl_stream = ccl::create_stream(queue);
    void* ptr = nullptr;
    ccl::mem_alloc(ccl_stream, total_size, &ptr);
    TORCH_CHECK(ptr != nullptr || total_size == 0, "ccl::mem_alloc failed");

    // ccl::mem_alloc does not zero memory; clear the signal pad so the
    // CAS-based signalling protocol starts from a known state.
    if (ptr != nullptr) {
      queue.memset(static_cast<char*>(ptr) + aligned_buffer_size, 0, signal_pad_size)
          .wait();
    }

    std::lock_guard<std::mutex> lock(mutex_);
    allocations_.try_emplace(
        ptr,
        std::make_unique<XCCLAllocation>(
            ptr, size, signal_pad_size, device_idx));
    return ptr;
  }

  void free(void* ptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = symm_mems_.begin(); it != symm_mems_.end();) {
      if (std::get<0>(it->first) == ptr) {
        it = symm_mems_.erase(it);
      } else {
        ++it;
      }
    }
    allocations_.erase(ptr);
  }

  size_t get_alloc_size(void* ptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(ptr);
    TORCH_CHECK(
        it != allocations_.end(),
        ptr,
        " is not allocated with XCCLSymmetricMemoryAllocator");
    return it->second->buffer_size;
  }

  c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr,
      const std::optional<std::string>& group_name) override {
    TORCH_CHECK(
        group_name.has_value(),
        "XCCLSymmetricMemoryAllocator::rendezvous requires a group_name");
    std::lock_guard<std::mutex> lock(mutex_);

    {
      auto it = symm_mems_.find(std::make_tuple(ptr, *group_name));
      if (it != symm_mems_.end()) {
        return it->second;
      }
    }

    // The tensor pointer may fall inside an allocation (e.g. a view / slice).
    auto alloc_it = std::find_if(
        allocations_.begin(), allocations_.end(), [&](const auto& pair) {
          auto& allocation = pair.second;
          auto ptr_int = reinterpret_cast<uintptr_t>(ptr);
          auto base_ptr = reinterpret_cast<uintptr_t>(allocation->ptr);
          return ptr_int >= base_ptr &&
              ptr_int < base_ptr + allocation->buffer_size;
        });
    TORCH_CHECK(
        alloc_it != allocations_.end(),
        "Pointer not within any SymmetricMemory allocation, "
        "is the tensor allocated from SymmetricMemory?");

    auto& allocation = alloc_it->second;

    // One `XCCLPeerAllocInfo` (and its window registration) per
    // (allocation base, group). Handles at different offsets share it.
    auto base_key = std::make_tuple(allocation->ptr, *group_name);
    c10::intrusive_ptr<XCCLPeerAllocInfo> pai;
    auto base_it = symm_mems_.find(base_key);
    if (base_it != symm_mems_.end()) {
      pai = base_it->second->pai_;
    } else {
      pai = c10::make_intrusive<XCCLPeerAllocInfo>(
          allocation.get(), *group_name);
      auto base_symm_mem =
          c10::make_intrusive<XCCLSymmetricMemory>(pai, /*offset=*/0);
      symm_mems_[base_key] = base_symm_mem;
      if (ptr == allocation->ptr) {
        return base_symm_mem;
      }
    }

    const size_t offset = reinterpret_cast<uintptr_t>(ptr) -
        reinterpret_cast<uintptr_t>(allocation->ptr);
    auto symm_mem = c10::make_intrusive<XCCLSymmetricMemory>(pai, offset);
    symm_mems_[std::make_tuple(ptr, *group_name)] = symm_mem;
    return symm_mem;
  }

  bool has_multicast_support(int device_idx) override {
    return false;
  }

  c10::DeviceType supported_device_type() override {
    return c10::DeviceType::XPU;
  }

  std::string name() override {
    return "XCCL";
  }

 private:
  std::mutex mutex_;
  std::unordered_map<void*, std::unique_ptr<XCCLAllocation>> allocations_;
  std::map<
      std::tuple<void*, std::string>,
      c10::intrusive_ptr<XCCLSymmetricMemory>>
      symm_mems_;
};

struct RegisterXCCLSymmetricMemoryAllocator {
  RegisterXCCLSymmetricMemoryAllocator() {
    auto allocator = c10::make_intrusive<XCCLSymmetricMemoryAllocator>();
    // Always advertise availability so `set_backend("XCCL")` can find it.
    register_availability("XCCL", allocator);
    // Become the active XPU allocator when explicitly selected.
    if (getSymmMemBackendXPU() == "XCCL") {
      register_allocator(c10::DeviceType::XPU, allocator);
    }
  }
};

static RegisterXCCLSymmetricMemoryAllocator register_allocator_;

} // namespace symmetric_memory
} // namespace c10d
