#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include "XPUSymmetricMemoryUtils.hpp"

#include <ATen/xpu/XPUContext.h>
#include <c10/util/error.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <comm/SYCLContext.h>

#include <ishmem.h>
#include <ishmemx.h>

#include <dlfcn.h>
#include <cstdlib>

namespace c10d {
namespace symmetric_memory {

constexpr int max_xpu_p2p_domain_size = 8;
constexpr int xpu_symm_max_nblocks = 32;
constexpr size_t xpu_signal_pad_size =
    xpu_symm_max_nblocks * max_xpu_p2p_domain_size * sizeof(uint32_t);

static StoreExchange storeExchange = StoreExchange("ISHMEMSymmetricMemory");

static bool ishmem_finalized = false;

static void finalize_ishmem_atexit() {
  if (!ishmem_finalized) {
    ishmem_finalized = true;
    ishmem_finalize();
  }
}

struct ISHMEMAllocation {
  void* ptr;
  size_t buffer_size;
  int device_idx;

  ISHMEMAllocation(void* ptr, size_t buffer_size, int device_idx)
      : ptr(ptr), buffer_size(buffer_size), device_idx(device_idx) {}

  ~ISHMEMAllocation() {
    if (is_finalizing() || ishmem_finalized) {
      return;
    }
    c10::OptionalDeviceGuard guard;
    guard.reset_device(at::Device(at::DeviceType::XPU, device_idx));
    ishmem_free(ptr);
  }
};

class ISHMEMPeerAllocInfo : public c10::intrusive_ptr_target {
 public:
  ISHMEMPeerAllocInfo(
      ISHMEMAllocation* allocation,
      const std::string& group_name)
      : base_ptr_(allocation->ptr),
        buffer_size_(allocation->buffer_size),
        device_idx_(allocation->device_idx) {
    static int exchanged_n_times = 0;
    c10::OptionalDeviceGuard guard;
    guard.reset_device(at::Device(at::DeviceType::XPU, allocation->device_idx));

    auto global_rank = get_group_info("0").rank;
    GroupInfo& group_info = get_group_info(group_name);
    auto store = group_info.store;
    rank_ = group_info.rank;
    world_size_ = group_info.world_size;
    if (group_info.rank_to_global_rank.empty()) {
      group_info.rank_to_global_rank =
          storeExchange.all_gather(store, rank_, world_size_, global_rank);
      exchanged_n_times++;
      if (rank_ == 0) {
        LOG(INFO) << "[rank " << rank_ << ']'
                  << " rank_to_global_rank: " << group_info.rank_to_global_rank
                  << ", group_name: " << group_name
                  << ", exchanged_n_times: " << exchanged_n_times;
      }
    }
    TORCH_INTERNAL_ASSERT(!group_info.rank_to_global_rank.empty());
    rank_to_global_rank_ = group_info.rank_to_global_rank;

    world_within_xpu_p2p_ = true;
    for (int r = 0; r < world_size_; ++r) {
      auto peer_ptr = ishmem_ptr(base_ptr_, rank_to_global_rank_[r]);
      buffers_.push_back(peer_ptr);
      if (peer_ptr == nullptr) {
        world_within_xpu_p2p_ = false;
      }
    }

    signal_pad_raw_ptr_ = ishmem_malloc(xpu_signal_pad_size);
    TORCH_CHECK(signal_pad_raw_ptr_ != nullptr, "ishmem_malloc failed");

    auto& queue = at::xpu::getCurrentSYCLQueue();
    queue.memset(signal_pad_raw_ptr_, 0, xpu_signal_pad_size).wait();

    for (int r = 0; r < world_size_; ++r) {
      signal_pads_.push_back(
          ishmem_ptr(signal_pad_raw_ptr_, rank_to_global_rank_[r]));
    }

    const size_t arr_size = sizeof(void*) * world_size_;
    buffers_dev_ = reinterpret_cast<void**>(
        c10::xpu::XPUCachingAllocator::raw_alloc(arr_size));
    signal_pads_dev_ = reinterpret_cast<void**>(
        c10::xpu::XPUCachingAllocator::raw_alloc(arr_size));

    queue.memcpy(buffers_dev_, buffers_.data(), arr_size).wait();
    queue.memcpy(signal_pads_dev_, signal_pads_.data(), arr_size).wait();

    rank_to_global_rank_dev_ = reinterpret_cast<int*>(
        c10::xpu::XPUCachingAllocator::raw_alloc(sizeof(int) * world_size_));
    queue
        .memcpy(
            rank_to_global_rank_dev_,
            rank_to_global_rank_.data(),
            sizeof(int) * world_size_)
        .wait();
  }

  ~ISHMEMPeerAllocInfo() {
    if (is_finalizing() || ishmem_finalized) {
      return;
    }
    c10::OptionalDeviceGuard guard;
    guard.reset_device(at::Device(at::DeviceType::XPU, device_idx_));

    if (signal_pad_raw_ptr_) {
      ishmem_free(signal_pad_raw_ptr_);
      signal_pad_raw_ptr_ = nullptr;
    }
    if (buffers_dev_) {
      c10::xpu::XPUCachingAllocator::raw_delete(buffers_dev_);
      buffers_dev_ = nullptr;
    }
    if (signal_pads_dev_) {
      c10::xpu::XPUCachingAllocator::raw_delete(signal_pads_dev_);
      signal_pads_dev_ = nullptr;
    }
    if (rank_to_global_rank_dev_) {
      c10::xpu::XPUCachingAllocator::raw_delete(rank_to_global_rank_dev_);
      rank_to_global_rank_dev_ = nullptr;
    }
  }

 private:
  void* base_ptr_;
  size_t buffer_size_;
  int device_idx_;
  int rank_;
  int world_size_;
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  void* signal_pad_raw_ptr_ = nullptr;
  void** buffers_dev_ = nullptr;
  void** signal_pads_dev_ = nullptr;
  std::vector<int> rank_to_global_rank_;
  int* rank_to_global_rank_dev_ = nullptr;
  bool world_within_xpu_p2p_;

  friend class ISHMEMSymmetricMemory;
};

class ISHMEMSymmetricMemory : public SymmetricMemory {
 public:
  ISHMEMSymmetricMemory(
      ISHMEMAllocation* allocation,
      const std::string& group_name)
      : device_idx_(allocation->device_idx), group_name_(group_name) {
    pai_ = c10::make_intrusive<ISHMEMPeerAllocInfo>(allocation, group_name);
    offset_ = 0;
  }

  ISHMEMSymmetricMemory(const ISHMEMSymmetricMemory& other) = delete;

  ISHMEMSymmetricMemory(const ISHMEMSymmetricMemory& other, size_t offset)
      : device_idx_(other.device_idx_),
        group_name_(other.group_name_),
        pai_(other.pai_) {
    offset_ = offset;
  }

  ~ISHMEMSymmetricMemory() override{
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
    return xpu_signal_pad_size;
  };

  bool has_multicast_support() override {
    return false;
  }

  void* get_multicast_ptr() override {
    return nullptr;
  }

  size_t get_offset() override {
    return offset_;
  }

  void barrier(int channel, size_t timeout_ms) override {
    ishmem_barrier_all();
  }

  void put_signal(int dst_rank, int channel, size_t timeout_ms) override {
    // TODO: Implement signal mechanism for ISHMEM
  }

  void wait_signal(int src_rank, int channel, size_t timeout_ms) override {
    // TODO: Implement signal mechanism for ISHMEM
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

  bool world_within_direct_access() override {
    return pai_->world_within_xpu_p2p_;
  }

 private:
  int device_idx_;
  std::string group_name_;
  c10::intrusive_ptr<ISHMEMPeerAllocInfo> pai_;
  size_t offset_{0}; // in byte
};

static void initialize_ishmem_with_store(
    c10::intrusive_ptr<c10d::Store> store,
    int rank,
    int world_size,
    int device_idx) {
  static bool is_initialized = false;
  if (is_initialized) {
    return;
  }
  c10::OptionalDeviceGuard guard;
  guard.reset_device(at::Device(at::DeviceType::XPU, device_idx));

  // Check if MPI is already initialized (e.g. by mpi4py).
  bool mpi_already_initialized = false;
  using MPI_Initialized_fn = int (*)(int*);
  auto mpi_initialized_fn = reinterpret_cast<MPI_Initialized_fn>(
      dlsym(RTLD_DEFAULT, "MPI_Initialized"));
  if (mpi_initialized_fn) {
    int flag = 0;
    mpi_initialized_fn(&flag);
    mpi_already_initialized = (flag != 0);
  }
  LOG(WARNING) << "Thiago's last version 5:46" << std::endl;

  // For torchrun/MPCP: set I_MPI_MPCP_RANK and I_MPI_MPCP if not set.
  if (!mpi_already_initialized && !getenv("I_MPI_MPCP_RANK")) {
    const char* local_rank = getenv("LOCAL_RANK");
    if (local_rank) {
      setenv("I_MPI_MPCP_RANK", local_rank, 1);
    }
  }
  if (!mpi_already_initialized && !getenv("I_MPI_MPCP")) {
    setenv("I_MPI_MPCP", "1", 1);
  }

  // UID-based init: ishmemx_get_uniqueid reads MASTER_ADDR from env.
  // initialize_runtime=true lets ISHMEM call MPI_Init (with MPCP support);
  // initialize_runtime=false when mpi4py already initialized MPI.
  ishmemx_uniqueid_t unique_id;
  memset(&unique_id, 0, sizeof(unique_id));

  if (rank == 0) {
    int ret = ishmemx_get_uniqueid(&unique_id);
    TORCH_CHECK(ret == 0, "ishmemx_get_uniqueid failed with error: ", ret);
  }

  std::vector<ishmemx_uniqueid_t> unique_ids =
      storeExchange.all_gather(store, rank, world_size, unique_id);

  ishmemx_attr_t attr;
  attr.initialize_runtime = !mpi_already_initialized;
  attr.use_uid = true;
  attr.nranks = world_size;
  attr.uid = &unique_ids[0];
  ishmemx_init_attr(&attr);

  TORCH_CHECK(
      ishmem_my_pe() == rank,
      "ISHMEM initialization failed: rank mismatch, expected ",
      rank,
      " got ",
      ishmem_my_pe());

  is_initialized = true;

  // Register ishmem_finalize via Py_AtExit (LIFO order ensures it runs
  // before mpi4py's MPI_Finalize, preventing proxy thread segfault).
  using PyAtExitFn = int (*)(void (*)(void));
  auto py_atexit =
      reinterpret_cast<PyAtExitFn>(dlsym(RTLD_DEFAULT, "Py_AtExit"));
  if (py_atexit != nullptr) {
    py_atexit(finalize_ishmem_atexit);
  } else {
    LOG(WARNING) << "Py_AtExit not found, falling back to std::atexit. "
                 << "ISHMEM cleanup at exit may not be orderly.";
    std::atexit(finalize_ishmem_atexit);
  }

  int major, minor;
  ishmem_info_get_version(&major, &minor);
  LOG(INFO) << "ISHMEM initialized, version: " << major << '.' << minor
            << ", rank: " << rank << "/" << world_size;
}

class ISHMEMSymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(
      size_t size,
      int device_idx,
      const std::optional<std::string>& group_name) override {
    c10::OptionalDeviceGuard guard;
    guard.reset_device(at::Device(at::DeviceType::XPU, device_idx));

    auto group_info = get_group_info("0");
    auto store = group_info.store;
    int rank = group_info.rank;
    int world_size = group_info.world_size;

    initialize_ishmem_with_store(store, rank, world_size, device_idx);
    auto ptr = ishmem_malloc(size);
    TORCH_CHECK(ptr != nullptr || size == 0, "ishmem_malloc failed");
    allocations_.try_emplace(
        ptr, std::make_unique<ISHMEMAllocation>(ptr, size, device_idx));
    return ptr;
  }

  void free(void* ptr) override {
    for (auto it = symm_mems_.begin(); it != symm_mems_.end();) {
      if (std::get<0>(it->first) == ptr) {
        it = symm_mems_.erase(it);
      } else {
        ++it;
      }
    }
    allocations_.erase(ptr);
  };

  size_t get_alloc_size(void* ptr) override {
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
      TORCH_CHECK(
          false, ptr, " is not allocated with ISHMEMSymmetricMemoryAllocator");
    }
    return it->second->buffer_size;
  };

  c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr,
      const std::optional<std::string>& group_name) override {
    std::string actual_group_name = group_name.has_value() ? *group_name : "0";

    {
      auto it = symm_mems_.find(std::make_tuple(ptr, actual_group_name));
      if (it != symm_mems_.end()) {
        return it->second;
      }
    }
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
    auto it =
        symm_mems_.find(std::make_tuple(allocation->ptr, actual_group_name));
    c10::intrusive_ptr<ISHMEMSymmetricMemory> symm_mem;
    if (it != symm_mems_.end()) {
      symm_mem = it->second;
    } else {
      symm_mem = c10::make_intrusive<ISHMEMSymmetricMemory>(
          allocation.get(), actual_group_name);
    }

    symm_mems_[std::make_tuple(allocation->ptr, actual_group_name)] = symm_mem;

    if (ptr == allocation->ptr) {
      return symm_mem;
    } else {
      return c10::make_intrusive<ISHMEMSymmetricMemory>(
          *symm_mem, (uintptr_t)ptr - (uintptr_t)allocation->ptr);
    }
  };

  bool has_multicast_support(int device_idx) override {
    return false;
  };

  c10::DeviceType supported_device_type() override {
    return c10::DeviceType::XPU;
  }

  std::string name() override {
    return "ISHMEM";
  }

 private:
  std::unordered_map<void*, std::unique_ptr<ISHMEMAllocation>> allocations_;
  std::map<
      std::tuple<void*, std::string>,
      c10::intrusive_ptr<ISHMEMSymmetricMemory>>
      symm_mems_;
};

struct RegisterISHMEMSymmetricMemoryAllocator {
  RegisterISHMEMSymmetricMemoryAllocator() {
    auto allocator = c10::make_intrusive<ISHMEMSymmetricMemoryAllocator>();
    register_availability("ISHMEM", allocator);
    if (getSymmMemBackendXPU() == "ISHMEM") {
      register_allocator(c10::DeviceType::XPU, allocator);
    }
  }
};

static RegisterISHMEMSymmetricMemoryAllocator register_allocator_;

} // namespace symmetric_memory
} // namespace c10d
