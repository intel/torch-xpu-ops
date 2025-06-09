#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

#include <c10/util/error.h>

#include <xccl/XPUSymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

namespace c10d::symmetric_memory {

bool device_has_multicast_support(int device_idx) {
  return true;
}

bool allow_overlapping_devices() {
  return true;
}

IpcChannel::IpcChannel()
    : socket_name_(get_socket_name(getpid())),
      socket_(socket(AF_UNIX, SOCK_DGRAM, 0)) {
  // On success, a file descriptor for the new socket is returned.
  //  On error, -1 is returned, and errno is set to indicate the error.
  TORCH_CHECK(
      socket_ != -1, "Failed to create socket: ", c10::utils::str_error(errno));

  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  std::copy(socket_name_.begin(), socket_name_.end(), addr.sun_path);

  TORCH_CHECK(
      bind(socket_, (struct sockaddr*)&addr, SUN_LEN(&addr)) == 0,
      "Failed to bind socket: ",
      c10::utils::str_error(errno));
}

IpcChannel::~IpcChannel() {
  close(socket_);
  unlink(socket_name_.c_str());
}

void IpcChannel::send_fd(int dst_pid, int fd) {
  // Because file descriptors are process-local kernel objects, and we can’t
  // pass them via normal socket payloads (like write() or send()).  Unix domain
  // sockets provide a mechanism to pass actual FDs via sendmsg()/recvmsg().
  // Define destination socket address
  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  auto socket_name = get_socket_name(dst_pid);
  std::copy(socket_name.begin(), socket_name.end(), addr.sun_path);

  // Prepare data to send
  // Data being sent is "fd", the value of fd will be sent as auxiliary data
  // (control message)
  struct iovec io = {.iov_base = (void*)("fd"), .iov_len = 2};

  // Prepare control message data buffer and zero it out
  // NOLINTNEXTLINE(*array*)
  char cbuf[CMSG_SPACE(sizeof(int))];
  memset(cbuf, 0, sizeof(cbuf));

  // Create message header
  struct msghdr msg {
    // destination socket address and size of it
    // message content in msg_iov and number of such structs (1 in our case)
    // auxiliary data with the value of fd and size of it
    .msg_name = (void*)&addr, .msg_namelen = sizeof(struct sockaddr_un),
    .msg_iov = &io, .msg_iovlen = 1, .msg_control = cbuf,
    .msg_controllen = sizeof(cbuf)
  };

  // This points to the first control message header
  // With SCM_RIGHTS we let the kernel know that we are passing file
  // descriptors.
  auto cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  // Specify socket level message
  cmsg->cmsg_level = SOL_SOCKET;
  // SCM_RIGHTS is the type used to pass file descriptors
  cmsg->cmsg_type = SCM_RIGHTS;

  if (fd != -1) {
    std::copy(
        reinterpret_cast<const char*>(&fd),
        reinterpret_cast<const char*>(&fd) + sizeof(fd),
        reinterpret_cast<char*>(CMSG_DATA(cmsg)));
  } else {
    msg.msg_controllen = 0;
  }

  // Finally send the the message
  TORCH_CHECK(
      sendmsg(socket_, &msg, 0) > 0,
      "Failed to send fd: ",
      c10::utils::str_error(errno));
}

int IpcChannel::recv_fd() {
  // Prepare buffer for regular message "fd"
  // NOLINTNEXTLINE(*array*)
  char buf[2];
  memset(&buf, 0, sizeof(buf));
  struct iovec io = {.iov_base = (void*)buf, .iov_len = sizeof(buf)};

  // Prepare buffer for control message and zero it out
  // NOLINTNEXTLINE(*array*)
  char cbuf[CMSG_SPACE(sizeof(int))];
  memset(cbuf, 0, sizeof(cbuf));

  // Define socket address to receive on: family AF_UNIX means unix domain
  // socket
  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  std::copy(socket_name_.begin(), socket_name_.end(), addr.sun_path);

  // Prepare message header
  struct msghdr msg = {
      .msg_name = (void*)&addr,
      .msg_namelen = sizeof(struct sockaddr_un),
      .msg_iov = &io,
      .msg_iovlen = 1,
      .msg_control = cbuf,
      .msg_controllen = sizeof(cbuf)};

  // Recieve message on socket_
  TORCH_CHECK(
      recvmsg(socket_, &msg, 0) > 0,
      "Failed to receive fd: ",
      c10::utils::str_error(errno));

  if (msg.msg_controllen == 0) {
    return -1;
  }

  // Extract control message and validate its content
  auto cmsg = CMSG_FIRSTHDR(&msg);
  TORCH_CHECK(cmsg != nullptr);
  TORCH_CHECK(cmsg->cmsg_len == CMSG_LEN(sizeof(int)));
  TORCH_CHECK(cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS);
  return *reinterpret_cast<int*>(CMSG_DATA(cmsg));
}

std::vector<int> IpcChannel::all_gather_fds(
    int rank,
    const std::vector<int>& pids,
    int fd) {
  int world_size = (int)pids.size();
  std::vector<int> fds(pids.size());
  fds[rank] = fd;

  int dst_rank = (rank + 1) % world_size;
  for (int step = 1; step < world_size; ++step) {
    int src_rank = (rank + world_size - step) % world_size;
    send_fd(pids[dst_rank], fd);
    fd = recv_fd();
    fds[src_rank] = fd;
  }
  return fds;
}

int IpcChannel::broadcast_fds(
    int rank,
    int src_rank,
    const std::vector<int>& pids,
    int fd) {
  int world_size = (int)pids.size();

  if (rank == src_rank) {
    for (int dst_rank = 0; dst_rank < (int)world_size; ++dst_rank) {
      if (dst_rank == rank) {
        continue;
      }
      send_fd(pids[dst_rank], fd);
    }
    return fd;
  }
  return recv_fd();
}

std::string IpcChannel::get_socket_name(int pid) {
  const char* tmp_dir = "/tmp";
  for (const char* env_var : {"TMPDIR", "TMP", "TEMP", "TEMPDIR"}) {
    if (const char* path = getenv(env_var)) {
      tmp_dir = path;
      break;
    }
  }
  std::ostringstream oss;
  oss << tmp_dir << "/symm_mem-" << pid;
  return oss.str();
}

void map_block(
    void** ptr,
    c10d::symmetric_memory::HandleType handle,
    size_t size,
    int device_idx) {
   sycl::queue current_queue = at::xpu::getCurrentXPUStream().queue();
   sycl::device sycl_ctx = current_queue.get_device();
   ze_context_handle_t ze_context =
    sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_ctx);

  // 1. Reserve virtual address space
  void* virtual_ptr = nullptr;
  ze_result_t status = zeVirtualMemReserve(
      ze_context,            // context
      nullptr,               // let L0 pick virtual address
      size,                  // size
      &virtual_ptr           // out: reserved address
  );
  TORCH_CHECK(status == ZE_RESULT_SUCCESS, "zeVirtualMemReserve failed");

  // 2. Map physical memory to virtual address
  status = zeVirtualMemMap(
      ze_context,
      virtual_ptr,           // virtual memory to map to
      size,
      handle,                // physical memory handle
      0                      // flags
  );
  TORCH_CHECK(status == ZE_RESULT_SUCCESS, "zeVirtualMemMap failed");

  // 3. Set access attributes
  ze_memory_access_attribute_t access = ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE;
  status = zeVirtualMemSetAccessAttribute(
      ze_context,
      virtual_ptr,
      size,
      access
  );
  TORCH_CHECK(status == ZE_RESULT_SUCCESS, "zeVirtualMemSetAccessAttribute failed");

  // 4. Return pointer
  *ptr = virtual_ptr;
}

} // namespace c10d::symmetric_memory
