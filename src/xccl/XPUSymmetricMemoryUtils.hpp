#pragma once

#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <xccl/XPUSymmetricMemoryTypes.hpp>

namespace c10d {
namespace symmetric_memory {

// Query environment variable to get the backend used for XPU Symmetric Memory.
std::string getSymmMemBackendXPU();

// Check if XPU device supports multicast (Level Zero IPC extensions)
bool xpu_device_has_multicast_support(int device_idx);

bool allow_overlapping_devices();

// XPU-specific IPC Channel using Level Zero IPC handles
class XPUIpcChannel {
 public:
  XPUIpcChannel();
  ~XPUIpcChannel();

  void send_handle(int dst_pid, const HandleType& handle);
  HandleType recv_handle();

  std::vector<HandleType> all_gather_handles(
      int rank,
      const std::vector<int>& pids,
      const HandleType& handle);

  HandleType broadcast_handles(
      int rank,
      int src_rank,
      const std::vector<int>& pids,
      const HandleType& handle);

 private:
  static std::string get_socket_name(int pid);
  std::string socket_name_;
  int socket_;
};

// A set of store-based exchange methods with a preset prefix for XPU
class XPUStoreExchange {
 public:
  XPUStoreExchange(const std::string& store_prefix)
      : store_prefix_(store_prefix) {}

  template <typename T>
  std::vector<T> all_gather(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int world_size,
      T val) {
    static_assert(std::is_trivially_copyable_v<T>);

    std::vector<std::string> peer_keys;
    peer_keys.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
      std::ostringstream oss;
      oss << store_prefix_ << "/" << seq_id_ << "/" << r;
      peer_keys.push_back(oss.str());
    }
    ++seq_id_;

    {
      std::vector<uint8_t> payload(
          reinterpret_cast<uint8_t*>(&val),
          reinterpret_cast<uint8_t*>(&val) + sizeof(T));
      store->set(peer_keys[rank], payload);
    }

    std::vector<T> peer_vals;
    peer_vals.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
      if (r == rank) {
        peer_vals.push_back(val);
        continue;
      }
      store->wait({peer_keys[r]});
      auto payload = store->get(peer_keys[r]);
      TORCH_CHECK(payload.size() == sizeof(T));
      T peer_val{};
      std::memcpy(&peer_val, payload.data(), sizeof(T));
      peer_vals.push_back(peer_val);
    }
    return peer_vals;
  }

  void barrier(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int world_size) {
    all_gather(store, rank, world_size, 0);
  }

 private:
  const std::string store_prefix_;
  size_t seq_id_ = 0;
};

// Map a Level Zero memory handle to virtual address
void map_xpu_block(
    void** ptr,
    const HandleType& handle,
    size_t size,
    int device_idx);

} // namespace symmetric_memory
} // namespace c10d
