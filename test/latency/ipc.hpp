#pragma once

#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

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


IpcChannel::IpcChannel()
    : socket_name_(get_socket_name(getpid())),
      socket_(socket(AF_UNIX, SOCK_DGRAM, 0)) {
  // On success, a file descriptor for the new socket is returned.
  //  On error, -1 is returned, and errno is set to indicate the error.
  if (socket_ == -1) {
    throw std::runtime_error(std::string("Failed to create socket: ") + std::strerror(errno));
  }

  struct sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::copy(socket_name_.begin(), socket_name_.end(), addr.sun_path);

  if (bind(socket_, (struct sockaddr*)&addr, SUN_LEN(&addr)) != 0) {
    throw std::runtime_error(std::string("Failed to bind socket: ") + std::strerror(errno));
  }
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
  struct sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  auto socket_name = get_socket_name(dst_pid);
  std::copy(socket_name.begin(), socket_name.end(), addr.sun_path);

  // Prepare data to send
  // Data being sent is "fd", the value of fd will be sent as auxiliary data
  // (control message)
  struct iovec io = {.iov_base = (void*)"fd", .iov_len = 2};

  // Prepare control message data buffer and zero it out
  // NOLINTNEXTLINE(*array*)
  char cbuf[CMSG_SPACE(sizeof(int))];
  memset(cbuf, 0, sizeof(cbuf));

  // Create message header
  struct msghdr msg{};
  msg.msg_name = (void*)&addr;
  msg.msg_namelen = sizeof(struct sockaddr_un);
  msg.msg_iov = &io;
  msg.msg_iovlen = 1;
  msg.msg_control = cbuf;
  msg.msg_controllen = sizeof(cbuf);

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

  // Finally send the message
  if (sendmsg(socket_, &msg, 0) <= 0) {
    throw std::runtime_error(std::string("Failed to send fd: ") + std::strerror(errno));
  }
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
  struct sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::copy(socket_name_.begin(), socket_name_.end(), addr.sun_path);

  // Prepare message header
  struct msghdr msg{};
  msg.msg_name = (void*)&addr;
  msg.msg_namelen = sizeof(struct sockaddr_un);
  msg.msg_iov = &io;
  msg.msg_iovlen = 1;
  msg.msg_control = cbuf;
  msg.msg_controllen = sizeof(cbuf);

  // Receive message on socket_
  if (recvmsg(socket_, &msg, 0) <= 0) {
    throw std::runtime_error(std::string("Failed to receive fd: ") + std::strerror(errno));
  }

  if (msg.msg_controllen == 0) {
    return -1;
  }

  // Extract control message and validate its content
  auto cmsg = CMSG_FIRSTHDR(&msg);
  if (cmsg == nullptr) {
    throw std::runtime_error("recvmsg: control message header is null");
  }
  if (cmsg->cmsg_len != CMSG_LEN(sizeof(int))) {
    throw std::runtime_error("recvmsg: unexpected cmsg_len");
  }
  if (cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) {
    throw std::runtime_error("recvmsg: unexpected cmsg_level or cmsg_type");
  }
  return *reinterpret_cast<int*>(CMSG_DATA(cmsg));
}

std::vector<int> IpcChannel::all_gather_fds(
    int rank,
    const std::vector<int>& pids,
    int fd) {
  int world_size = static_cast<int>(pids.size());
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
  int world_size = static_cast<int>(pids.size());

  if (rank == src_rank) {
    for (int dst_rank = 0; dst_rank < world_size; ++dst_rank) {
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