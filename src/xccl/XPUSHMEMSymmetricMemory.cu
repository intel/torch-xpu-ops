#include <torch/csrc/distributed/c10d/xpu/utils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <xccl/nvshmem_extension.cpp>
#include <xccl/XPUSymmetricMemoryUtils.hpp>

#include <ATen/ceil_div.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUGuard.h>
#include <c10/util/error.h>

// Starting from NVSHMEM 3.3.9, nvshmem_host.h exists so that we can cleanly
// include only the nvshmem host library headers:
// #include <nvshmem_host.h>
// It translates into the following two lines:
#include <host/ishmem_api.h>
#include <host/ishmemx_api.h>
// For maximum compatibility, we use the "host/" style for now.

namespace c10d {
namespace symmetric_memory {

/* Start of NVSHMEMSymmetricMemory implementation */

static StoreExchange storeExchange = StoreExchange("XPUSHMEMSymmetricMemory");

struct XPUSHMEMAllocation {
  void* ptr;
  size_t buffer_size;
  int device_idx;

  XPUSHMEMAllocation(void* ptr, size_t buffer_size, int device_idx)
      : ptr(ptr), buffer_size(buffer_size), device_idx(device_idx) {}

  ~XPUSHMEMAllocation() {
    // Avoid calling CUDA functions after driver shutting down
    if (is_finalizing()) { // todo: what is it?
      return;
    }
    c10::xpu::XPUGuard guard(device_idx);
    ishmem_free(ptr);  // nvshmem_free has no return value
  }
};

// A class to hold the base pointers and signal pad pointers for a group of
// peers. One `NVSHMEMPeerAllocInfo` object can be shared by multiple
// `NVSHMEMSymmetricMemory` objects when latter reside on the same allocation
// and rendezvous over the same group. (The `NVSHMEMSymmetricMemory` objects may
// have different offsets compared to the base address.)
class XPUSHMEMPeerAllocInfo : public c10::intrusive_ptr_target {
 public:
  XPUSHMEMPeerAllocInfo(
      std::shared_ptr<XPUSHMEMAllocation> allocation,
      const std::string& group_name)
      : base_ptr_(allocation->ptr),
        buffer_size_(allocation->buffer_size) {
    // For logging only
    static int exchanged_n_times = 0;
    c10::xpu::XPUGuard guard(allocation->device_idx);

    auto global_rank = get_group_info("0").rank;
    GroupInfo& group_info = get_group_info(group_name);
    auto store = group_info.store;
    rank_ = group_info.rank;
    world_size_ = group_info.world_size;
    // Exchange rank to global rank mapping for this group.
    // If it is already available, skip the exchange.
    if (group_info.rank_to_global_rank.empty()) {
      group_info.rank_to_global_rank =
          storeExchange.all_gather(store, rank_, world_size_, global_rank);
      exchanged_n_times++;
      if (rank_ == 0) {
        LOG(INFO) << "[rank " << rank_ << "]"
                  << " rank_to_global_rank: " << group_info.rank_to_global_rank
                  << ", group_name: " << group_name
                  << ", exchanged_n_times: " << exchanged_n_times;
      }
    }
    TORCH_INTERNAL_ASSERT(!group_info.rank_to_global_rank.empty());
    rank_to_global_rank_ = group_info.rank_to_global_rank;

    world_within_xpu_p2p_ = true;
    for (int r = 0; r < world_size_; ++r) {
      auto peer_ptr = ishmem_ptr(
          base_ptr_, rank_to_global_rank_[r]);
      buffers_.push_back(peer_ptr);
      // If a peer is over network, `nvshmem_ptr` returns null
      if (peer_ptr == nullptr) {
        world_within_xpu_p2p_ = false;
      }
    }

    // TODO: use the same allocation for signal pad
    void* signal_pad_ptr = ishmem_malloc(signal_pad_size);
    TORCH_CHECK(signal_pad_ptr != nullptr, "ishmem_malloc failed");
    current_queue.memset(signal_pad_ptr, 0, signal_pad_size);

    for (int r = 0; r < world_size_; ++r) {
      signal_pads_.push_back(ishmem_ptr(
          signal_pad_ptr, rank_to_global_rank_[r]));
    }

    const size_t arr_size = sizeof(void*) * world_size_;
    buffers_dev_ = reinterpret_cast<void**>(
        c10::xpu::XPUCachingAllocator::raw_alloc(arr_size));
    signal_pads_dev_ = reinterpret_cast<void**>(
        c10::xpu::XPUCachingAllocator::raw_alloc(arr_size));
    rank_to_global_rank_dev_ = reinterpret_cast<int*>(
        c10::xpu::XPUCachingAllocator::raw_alloc(sizeof(int) * world_size_));

    at::xpu::getCurrentXPUStream().queue().memcpy(
        buffers_dev_, buffers_.data(), arr_size);
    at::xpu::getCurrentXPUStream().queue().memcpy(
        signal_pads_dev_, signal_pads_.data(), arr_size);
    at::xpu::getCurrentXPUStream().queue().memcpy(
        rank_to_global_rank_dev_, rank_to_global_rank_.data(), sizeof(int) * world_size_);
  }

 private:
  void* base_ptr_;
  size_t buffer_size_;
  int rank_;
  int world_size_;
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  void** buffers_dev_;
  void** signal_pads_dev_;
  std::vector<int> rank_to_global_rank_;
  int* rank_to_global_rank_dev_;
  // Whether the world is within CUDA P2P only, not network
  bool world_within_xpu_p2p_;

  friend class XPUSHMEMSymmetricMemory;
};

class XPUSHMEMSymmetricMemory : public SymmetricMemory {
 public:
  XPUSHMEMSymmetricMemory(
      std::shared_ptr<XPUSHMEMAllocation> allocation,
      const std::string& group_name)
      : allocation_(allocation),
        device_idx_(allocation->device_idx),
        group_name_(group_name) {
    // A handle stores two types of info:
    // (i) allocation's base ptrs and base signal pads, ours and peers'
    pai_ = c10::make_intrusive<XPUSHMEMPeerAllocInfo>(allocation, group_name);
    // (ii) offset of tensor compared to base ptr (in byte)
    offset_ = 0;
  }

  // Exact copy is not needed / supported
  XPUSHMEMSymmetricMemory(const XPUSHMEMSymmetricMemory& other) = delete;

  // Copy with offset is allowed
  // This is mostly a shallow copy that shares the pointer to `NVSHMEMPeerAllocInfo` which has been created by `other`
  XPUSHMEMSymmetricMemory(const XPUSHMEMSymmetricMemory& other, size_t offset)
      : allocation_(other.allocation_), device_idx_(other.device_idx_), group_name_(other.group_name_), pai_(other.pai_) {
    offset_ = offset;
  }

  ~XPUSHMEMSymmetricMemory() override{
      // TODO
  };

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

  size_t get_signal_pad_size() override {
    return signal_pad_size;
  };

  bool has_multicast_support() override {
    // TODO
    return false;
  }

  void* get_multicast_ptr() override {
    // TODO
    return nullptr;
  }

  size_t get_offset() override {
    return offset_;
  }

  void barrier(int channel, size_t timeout_ms) override {
    // TODO
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
    return c10::Device(c10::DeviceType::XPU, device_idx_);
  }

  const std::vector<int>& get_rank_to_global_rank() override {
    return pai_->rank_to_global_rank_;
  };

  int* get_rank_to_global_rank_dev() override {
    return pai_->rank_to_global_rank_dev_;
  };

  bool world_within_direct_access() {
    return pai_->world_within_xpu_p2p_;
  }

 private:
  std::shared_ptr<XPUSHMEMAllocation> allocation_;
  int device_idx_;
  std::string group_name_;
  c10::intrusive_ptr<XPUSHMEMPeerAllocInfo> pai_;
  size_t offset_{0};  // in byte
};

// Bootstrap based on user's setting for NCCL
// Long term, this may be a bit unclean; short term, it improves UX
static void maybe_initialize_env_vars() {
//   auto nccl_socket_if_name = c10::utils::get_env("NCCL_SOCKET_IFNAME");
//   auto nccl_hca_list = c10::utils::get_env("NCCL_IB_HCA");
//   auto nccl_ib_gid_index = c10::utils::get_env("NCCL_IB_GID_INDEX");
//   auto nvshmem_socket_if_name =
//       c10::utils::get_env("NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME");
//   auto nvshmem_hca_list = c10::utils::get_env("NCCL_IB_HCA");
//   auto nvshmem_ib_gid_index = c10::utils::get_env("NVSHMEM_IB_GID_INDEX");
//
//   if (!nvshmem_socket_if_name.has_value() && nccl_socket_if_name.has_value()) {
//     c10::utils::set_env(
//         "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME", nccl_socket_if_name->c_str());
//   }
//   if (!nvshmem_hca_list.has_value() && nccl_hca_list.has_value()) {
//     c10::utils::set_env("NVSHMEM_ENABLE_NIC_PE_MAPPING", "1");
//     c10::utils::set_env("NVSHMEM_HCA_LIST", nccl_hca_list->c_str());
//   }
//   if (!nvshmem_ib_gid_index.has_value() && nccl_ib_gid_index.has_value()) {
//     c10::utils::set_env("NVSHMEM_IB_GID_INDEX", nccl_ib_gid_index->c_str());
//   }
}

static void initialize_xpushmem_with_store(
    c10::intrusive_ptr<c10d::Store> store,
    int rank,
    int world_size,
    int device_idx) {
  static bool is_initialized = false;
  if (is_initialized) {
    return;
  }

  c10::xpu::XPUGuard guard(device_idx);
  maybe_initialize_env_vars();
  // Make sure the CUDA runtime is initialized.
//   cudaFree(nullptr); // todo: check?

  ishmemx_uniqueid_t unique_id;
  XPUSHMEM_CHECK(
      ishmemx_get_uniqueid(&unique_id), "intel_shmemx_get_uniqueid failed");

  // Using an existing store_all_gather due to laziness.
  // TODO(yifu): should use broadcast
  auto unique_ids = storeExchange.all_gather(store, rank, world_size, unique_id);

  ishmemx_init_attr_t attr;
  ishmemx_set_attr_uniqueid_args(rank, world_size, &unique_ids[0], &attr);

  XPUSHMEM_CHECK(
      ishmemx_init_attr(XPUSHMEMX_INIT_WITH_UNIQUEID, &attr),
      "ishmemx_init_attr failed");

  is_initialized = true;

  // Print version
  int major, minor;
  ::ishmem_info_get_version(&major, &minor);
  LOG(INFO) << "XPUSHMEM is available, version: " << major << '.' << minor;
}

class XPUSHMEMSymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(
      size_t size,
      int device_idx,
      const std::optional<std::string>& group_name) override {
    TORCH_CHECK(
        group_name == std::nullopt,
        "XPUSHMEMSymmetricMemoryAllocator::alloc "
        "must not be called with a group_name");
    c10::xpu::XPUGuard guard(device_idx);

    auto group_info = get_group_info("0");
    auto store = group_info.store;
    int rank = group_info.rank;
    int world_size = group_info.world_size;

    initialize_ishmem_with_store(store, rank, world_size, device_idx);
    auto ptr = ishmem_malloc(size);
    // If size is 0 (which is legal allocation request) we shouldn't error out
    TORCH_CHECK(ptr != nullptr || size == 0, "ishmem_malloc failed");
    auto allocation =
        std::make_shared<XPUSHMEMAllocation>(ptr, size, device_idx);
    // TODO: thread safety
    allocations_.try_emplace(ptr, std::move(allocation));
    return ptr;
  }

  void free(void* ptr) override {
    // TODO: thread safety
    allocations_.erase(ptr);
  };

  size_t get_alloc_size(void* ptr) override {
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
      TORCH_CHECK(
          false, ptr, " is not allocated with XPUSHMEMSymmetricMemoryAllocator");
    }
    return it->second->buffer_size;
  };

  c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr,
      const std::optional<std::string>& group_name) override {
    TORCH_CHECK(group_name.has_value());
    {
      auto it = symm_mems_.find(std::make_tuple(ptr, *group_name));
      if (it != symm_mems_.end()) {
        return it->second;
      }
    }
    // In case of MemPool, tensor.storage().data_ptr() may not match
    // exactly an allocation's base address. Thus we perform the search by
    // testing if the former is within an allocation's range.
    auto alloc_it = std::find_if(allocations_.begin(), allocations_.end(),
                               [&](const auto& pair){
                                  auto& allocation = pair.second;
                                  auto ptr_int = reinterpret_cast<uintptr_t>(ptr);
                                  auto base_ptr = reinterpret_cast<uintptr_t>(allocation->ptr);
                                  return ptr_int >= base_ptr && ptr_int < base_ptr + allocation->buffer_size; });
    TORCH_CHECK(alloc_it != allocations_.end(),
        "Pointer not within any SymmetricMemory allocation, "
        "is the tensor allocated from SymmetricMemory?");

    auto& allocation = alloc_it->second;

    // Search again using allocation base ptr (which is the key we use for caching, see below)
    auto it = symm_mems_.find(std::make_tuple(allocation->ptr, *group_name));
    c10::intrusive_ptr<XPUSHMEMSymmetricMemory> symm_mem;
    if (it != symm_mems_.end()) {
      // Base allocation has been rendezvoused
      symm_mem = it->second;
    } else {
      // Create a new rendezvous
      symm_mem =
          c10::make_intrusive<XPUSHMEMSymmetricMemory>(allocation, *group_name);
    }

    // Cache rendezvous using allocation's base address as key
    symm_mems_[std::make_tuple(allocation->ptr, *group_name)] = symm_mem;

    // TODO: change the `ptr` below to `tensor.data_ptr()` when adding support
    // for user slice/view operations. For MemPool support,
    // `tensor.storate().data_ptr()` is fine (today's `ptr`).

    // If the tensor's ptr happen to be the same as allocation ptr
    if (ptr == allocation->ptr) {
      return symm_mem;
    } else {
      // Return a copy of the SymmetricMemory with an offset. This is a
      // "shallow" copy adjusting the offset field in the handle.
      return c10::make_intrusive<XPUSHMEMSymmetricMemory>(*symm_mem, (uintptr_t)ptr - (uintptr_t)allocation->ptr);
    }
  };

  bool has_multicast_support(int device_idx) override {
    // TODO
    return false;
  };

  c10::DeviceType supported_device_type() override {
    return c10::DeviceType::XPU;
  }

  std::string name() override {
    return "XPUSHMEM";
  }

 private:
  std::unordered_map<void*, std::shared_ptr<XPUSHMEMAllocation>> allocations_;
  std::map<std::tuple<void*, std::string>, c10::intrusive_ptr<XPUSHMEMSymmetricMemory>>
      symm_mems_;
};

struct RegisterXPUSHMEMSymmetricMemoryAllocator {
  RegisterXPUSHMEMSymmetricMemoryAllocator() {
    auto allocator = c10::make_intrusive<XPUSHMEMSymmetricMemoryAllocator>();
    // Query backend used for CUDA tensor
    if (getSymmMemBackendXPU() == "XPUSHMEM") {
      // Direct set (static registration)
      register_allocator(
          c10::DeviceType::XPU,
          allocator);
    } else {
      // Register availability in case `set_backend` is called dynamically
      register_availability("XPUSHMEM", allocator);
    }
  }
};

static RegisterXPUSHMEMSymmetricMemoryAllocator register_allocator_;

} // namespace symmetric_memory
} // namespace c10d
