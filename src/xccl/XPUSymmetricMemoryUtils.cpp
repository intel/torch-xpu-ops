#include <xccl/XPUSymmetricMemoryUtils.hpp>

#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <c10/xpu/XPUFunctions.h>

#include <level_zero/ze_api.h>
// #include <sycl/sycl.hpp>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <sstream>

namespace c10d {
namespace symmetric_memory {

std::string getSymmMemBackendXPU() {
  static auto val = c10::utils::get_env("TORCH_SYMMMEM");
  if (val.has_value()) {
    TORCH_CHECK(
        val.value() == "XPU",
        "TORCH_SYMMMEM environment variable must be 'XPU'.");
    return val.value();
  }
  return "XPU";
}

bool device_has_multicast_support(int device_idx) {
  // XPU/Level Zero doesn't currently have equivalent multicast support like
  // CUDA
  return false;
}

bool allow_overlapping_devices() {
  return c10::utils::check_env("TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES") ==
      true;
}

XPUIpcChannel::XPUIpcChannel() {
  socket_name_ = get_socket_name(getpid());

  socket_ = socket(AF_UNIX, SOCK_STREAM, 0);
  TORCH_CHECK(socket_ != -1, "Failed to create socket");

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, socket_name_.c_str(), sizeof(addr.sun_path) - 1);

  unlink(socket_name_.c_str());

  int result = bind(socket_, (struct sockaddr*)&addr, sizeof(addr));
  TORCH_CHECK(result != -1, "Failed to bind socket");

  result = listen(socket_, 128);
  TORCH_CHECK(result != -1, "Failed to listen on socket");
}

XPUIpcChannel::~XPUIpcChannel() {
  if (socket_ != -1) {
    close(socket_);
  }
  unlink(socket_name_.c_str());
}

std::string XPUIpcChannel::get_socket_name(int pid) {
  std::ostringstream oss;
  oss << "/tmp/torch_xpu_symm_mem_" << pid;
  return oss.str();
}

void XPUIpcChannel::send_handle(int dst_pid, const HandleType& handle) {
  std::string dst_socket_name = get_socket_name(dst_pid);

  int client_socket = socket(AF_UNIX, SOCK_STREAM, 0);
  TORCH_CHECK(client_socket != -1, "Failed to create client socket");

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, dst_socket_name.c_str(), sizeof(addr.sun_path) - 1);

  // Retry connection with backoff
  int retries = 10;
  while (retries > 0) {
    if (connect(client_socket, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
      break;
    }
    retries--;
    if (retries > 0) {
      usleep(100000); // 100ms
    }
  }
  TORCH_CHECK(retries > 0, "Failed to connect to destination socket");

  // Send the Level Zero IPC handle
  ssize_t sent = send(client_socket, &handle, sizeof(handle), 0);
  TORCH_CHECK(sent == sizeof(handle), "Failed to send handle");

  close(client_socket);
}

HandleType XPUIpcChannel::recv_handle() {
  int client_socket = accept(socket_, nullptr, nullptr);
  TORCH_CHECK(client_socket != -1, "Failed to accept connection");

  HandleType handle;
  ssize_t received = recv(client_socket, &handle, sizeof(handle), MSG_WAITALL);
  TORCH_CHECK(received == sizeof(handle), "Failed to receive handle");

  close(client_socket);
  return handle;
}

std::vector<HandleType> XPUIpcChannel::all_gather_handles(
    int rank,
    const std::vector<int>& pids,
    const HandleType& handle) {
  std::vector<HandleType> handles(pids.size());
  handles[rank] = handle;

  // Send to all other ranks
  for (size_t i = 0; i < pids.size(); ++i) {
    if (static_cast<int>(i) != rank) {
      send_handle(pids[i], handle);
    }
  }

  // Receive from all other ranks
  for (size_t i = 0; i < pids.size(); ++i) {
    if (static_cast<int>(i) != rank) {
      handles[i] = recv_handle();
    }
  }

  return handles;
}

HandleType XPUIpcChannel::broadcast_handles(
    int rank,
    int src_rank,
    const std::vector<int>& pids,
    const HandleType& handle) {
  if (rank == src_rank) {
    // Send to all other ranks
    for (size_t i = 0; i < pids.size(); ++i) {
      if (static_cast<int>(i) != rank) {
        send_handle(pids[i], handle);
      }
    }
    return handle;
  } else {
    // Receive from source rank
    return recv_handle();
  }
}

void map_xpu_block(
    void** ptr,
    const HandleType& handle,
    size_t size,
    int device_idx) {
#ifdef USE_XPU
  auto& sycl_context = c10::xpu::get_device_context();
  auto& sycl_device = c10::xpu::get_raw_device(device_idx);

  ze_context_handle_t ze_context =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_context);
  ze_device_handle_t ze_device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);

  ze_result_t result =
      zeMemOpenIpcHandle(ze_context, ze_device, handle, 0, ptr);
  TORCH_CHECK(
      result == ZE_RESULT_SUCCESS,
      "Failed to map XPU block via Level Zero IPC");
#else
  TORCH_CHECK(false, "XPU support not available");
#endif
}

} // namespace symmetric_memory
} // namespace c10d
