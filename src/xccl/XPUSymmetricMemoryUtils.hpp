#pragma once

#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <xccl/XPUSymmetricMemoryTypes.hpp>

#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

#include <c10/util/error.h>

namespace c10d {
namespace symmetric_memory {

std::string getSymmMemBackendXPU();

class IpcChannel {
 public:
  IpcChannel();
  ~IpcChannel();

  void send_fd(int dst_pid, int fd);
  int recv_fd();

  std::vector<int> all_gather_fds(
      int rank,
      const std::vector<int>& pids,
      int fd);

  int broadcast_fds(
      int rank,
      int src_rank,
      const std::vector<int>& pids,
      int fd);

 private:
  static std::string get_socket_name(int pid);

  std::string socket_name_;
  int socket_;
};

class StoreExchange {
 public:
  StoreExchange(const std::string& store_prefix)
      : store_prefix_(store_prefix) {}

  // Put template function in header file so that compiler can easily access it.
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
    // TODO: implement an efficient one?
    all_gather(store, rank, world_size, 0);
  }

  // Variable-length byte all_gather (used to exchange SYCL IPC handles, whose
  // serialized size is opaque and may differ from platform to platform).
  std::vector<std::vector<uint8_t>> all_gather_bytes(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int world_size,
      const std::vector<uint8_t>& payload) {
    std::vector<std::string> peer_keys;
    peer_keys.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
      std::ostringstream oss;
      oss << store_prefix_ << "/bytes/" << seq_id_ << "/" << r;
      peer_keys.push_back(oss.str());
    }
    ++seq_id_;

    store->set(peer_keys[rank], payload);

    std::vector<std::vector<uint8_t>> peer_vals(world_size);
    peer_vals[rank] = payload;
    for (int r = 0; r < world_size; ++r) {
      if (r == rank) {
        continue;
      }
      store->wait({peer_keys[r]});
      peer_vals[r] = store->get(peer_keys[r]);
    }
    return peer_vals;
  }

 private:
  const std::string store_prefix_;
  size_t seq_id_ = 0;
};

} // namespace symmetric_memory
} // namespace c10d
