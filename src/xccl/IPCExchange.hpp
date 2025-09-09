#include <poll.h>
#include <stddef.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>
#include <future>
#include <system_error>
#include "xccl/ze_symbol.hpp"

#include <sycl/sycl.hpp>

#include <pwd.h>
#include <stdio.h>
#include <unistd.h>
#include <chrono> // for std::chrono::milliseconds
#include <thread> // for std::this_thread::sleep_for

#define ELE_COUNT 128

struct exchange_contents {
  // first 4-byte is file descriptor for drmbuf or gem object
  union {
    ze_ipc_mem_handle_t ipc_handle;
    int fd = -1;
  };
  size_t offset = 0;
  int pid = -1;
};

#define sysCheck(x)                                                  \
  if (x == -1) {                                                     \
    throw std::system_error(std::make_error_code(std::errc(errno))); \
  }

// We can't inherit it from cmsghdr because flexible array member
struct exchange_fd {
  char obscure[CMSG_LEN(sizeof(int)) - sizeof(int)];
  int fd;

  exchange_fd(int cmsg_level, int cmsg_type, int fd) : fd(fd) {
    auto* cmsg = reinterpret_cast<cmsghdr*>(obscure);
    cmsg->cmsg_len = sizeof(exchange_fd);
    cmsg->cmsg_level = cmsg_level;
    cmsg->cmsg_type = cmsg_type;
  }

  exchange_fd() : fd(-1) {
    memset(obscure, 0, sizeof(obscure));
  };
};

void un_send_fd(int sock, int fd, int rank, size_t offset) {
  iovec iov[1];
  msghdr msg;
  auto rank_offset = std::make_pair(rank, offset);

  iov[0].iov_base = &rank_offset;
  iov[0].iov_len = sizeof(rank_offset);
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;
  msg.msg_name = nullptr;
  msg.msg_namelen = 0;

  exchange_fd cmsg(SOL_SOCKET, SCM_RIGHTS, fd);

  msg.msg_control = &cmsg;
  msg.msg_controllen = sizeof(exchange_fd);
  sysCheck(sendmsg(sock, &msg, 0));
}

std::tuple<int, int, size_t> un_recv_fd(int sock) {
  iovec iov[1];
  msghdr msg;
  std::pair<int, size_t> rank_offset;

  iov[0].iov_base = &rank_offset;
  iov[0].iov_len = sizeof(rank_offset);
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;
  msg.msg_name = nullptr;
  msg.msg_namelen = 0;

  exchange_fd cmsg;
  msg.msg_control = &cmsg;
  msg.msg_controllen = sizeof(exchange_fd);
  int n_recv = recvmsg(sock, &msg, 0);
  sysCheck(n_recv);
  // assert(n_recv == sizeof(int));

  return std::make_tuple(cmsg.fd, rank_offset.first, rank_offset.second);
}

int prepare_socket(const char* sockname) {
  sockaddr_un un;
  memset(&un, 0, sizeof(un));
  un.sun_family = AF_UNIX;
  strcpy(un.sun_path, sockname);

  auto sock = socket(AF_UNIX, SOCK_STREAM, 0);
  sysCheck(sock);

  int on = 1;
  sysCheck(ioctl(sock, FIONBIO, &on));

  auto size = offsetof(sockaddr_un, sun_path) + strlen(un.sun_path);
  sysCheck(bind(sock, (sockaddr*)&un, size));

  return sock;
}

int server_listen(const char* sockname) {
  unlink(sockname);
  auto sock = prepare_socket(sockname);
  sysCheck(listen(sock, 10));

  return sock;
}

int serv_accept(int listen_sock) {
  sockaddr_un un;

  socklen_t len = sizeof(un);
  auto accept_sock = accept(listen_sock, (sockaddr*)&un, &len);
  sysCheck(accept_sock);

  return accept_sock;
}

bool wait_for_socket_file(const char* path, int max_seconds = 10) {
  struct stat buffer;
  for (int i = 0; i < max_seconds * 10; ++i) {
    if (stat(path, &buffer) == 0) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return false;
}

int client_connect(const char* server, const char* client) {
  if (!wait_for_socket_file(server, 10)) {
    std::cerr << "Error: timeout waiting for server socket file: " << server
              << std::endl;
    exit(EXIT_FAILURE);
  }
  auto sock = prepare_socket(client);
  sockaddr_un sun;
  memset(&sun, 0, sizeof(sun));
  sun.sun_family = AF_UNIX;
  strcpy(sun.sun_path, server);
  auto len = offsetof(sockaddr_un, sun_path) + strlen(server);
  const int max_retries = 50;
  int retry = 0;
  int ret = -1;
  while (retry < max_retries) {
    ret = connect(sock, (sockaddr*)&sun, len);
    if (ret == 0)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    retry++;
  }
  if (ret != 0) {
    perror("connect failed");
    exit(EXIT_FAILURE);
  }

  // sysCheck(connect(sock, (sockaddr*)&sun, len));
  return sock;
}

void un_allgather(
    exchange_contents* send_buf,
    exchange_contents recv_buf[],
    int rank,
    int world) {
  const char* servername_prefix = "/tmp/open-peer-ipc-mem-server-rank_";
  const char* clientname_prefix = "/tmp/open-peer-ipc-mem-client-rank_";
  char server_name[64];
  /* get username to make server_name unique */
  auto uid = getuid();
  auto pwd = getpwuid(uid);
  snprintf(
      server_name,
      sizeof(server_name),
      "%s%d_%s",
      servername_prefix,
      rank,
      pwd->pw_name);
  unlink(server_name);
  auto s_listen = server_listen(server_name);

  pollfd fdarray[world];
  int recv_socks[world - 1];

  for (auto& pollfd : fdarray)
    pollfd.fd = -1;
  std::fill(recv_socks, recv_socks + world - 1, -1);

  auto fd_guard = [&]() {
    for (int i = 0, j = 0; i < world; ++i) {
      if (i != rank && recv_socks[j] != -1)
        sysCheck(close(recv_socks[j++]));
      if (fdarray[i].fd != -1)
        sysCheck(close(fdarray[i].fd));
    }
  };

  struct guard__ {
    using F = decltype(fd_guard);
    F f;
    guard__(const F& f) : f(f) {}
    ~guard__() {
      f();
    }
  } free_fd(fd_guard);

  // connect to all ranks
  for (int i = 0; i < world; ++i) {
    if (rank == i) {
      fdarray[i].fd = s_listen;
      fdarray[i].events = POLLIN;
      fdarray[i].revents = 0;
    } else {
      char peer_name[64];
      char client_name[64];

      snprintf(
          client_name,
          sizeof(client_name),
          "%s%d-%d_%s",
          clientname_prefix,
          rank,
          i,
          pwd->pw_name);
      unlink(client_name);

      snprintf(
          peer_name,
          sizeof(peer_name),
          "%s%d_%s",
          servername_prefix,
          i,
          pwd->pw_name);
      fdarray[i].fd = client_connect(peer_name, client_name);
      fdarray[i].events = POLLOUT;
      fdarray[i].revents = 0;
    }
  }

  // std::future<std::tuple<int, int, size_t>> future_fds[world -1];
  int slot = 0;
  uint32_t send_progress = 1 << rank;

  while (slot < world - 1 || send_progress != (1 << world) - 1) {
    sysCheck(ppoll(fdarray, world, nullptr, nullptr));

    for (int i = 0; i < world; ++i) {
      if (i == rank && (fdarray[i].revents & POLLIN)) {
        // auto accept_sock = serv_accept(fdarray[i].fd);
        // future_fds[slot ++] = std::async(
        //     std::launch::async, [=]() {
        //     struct sock_guard{
        //       int sock;
        //       sock_guard(int sock) : sock(sock) {}
        //       ~guard_sock() {sysCheck(close(sock));}
        //     } release(accept_sock);
        //     auto ret = un_recv_fd(accept_sock);
        //     return ret;});
        recv_socks[slot++] = serv_accept(fdarray[i].fd);
      } else if (
          (send_progress & (1 << i)) == 0 && fdarray[i].revents & POLLOUT) {
        un_send_fd(fdarray[i].fd, send_buf->fd, rank, send_buf->offset);
        send_progress |= 1 << i;
      }
    }
  }

  for (int i = 0; i < world - 1; ++i) {
    // future_fds[i].wait();
    // auto [fd, peer, offset] = future_fds[i].get();
    auto [fd, peer, offset] = un_recv_fd(recv_socks[i]);
    recv_buf[peer].fd = fd;
    recv_buf[peer].offset = offset;
  }

  recv_buf[rank] = *send_buf;
}

template <
    typename data_type,
    uint32_t max_rank = 8,
    uint32_t max_buffer = 1024 /*KB*/>
class allreducer {
 public:
  allreducer() {
    initialized = false;
    size_per_buffer = 0;
    buffer_index = 0;
  }
  allreducer(const allreducer&) = delete;
  allreducer& operator=(const allreducer&) = delete;
  allreducer(allreducer&& other) noexcept {
    *this = std::move(other);
  }
  allreducer& operator=(allreducer&& other) noexcept {
    if (this != &other) {
      initialized = other.initialized;
      rank = other.rank;
      world = other.world;
      std::memcpy(buffers, other.buffers, sizeof(buffers));
      std::memcpy(offsets, other.offsets, sizeof(offsets));
      std::memcpy(ipc_handle, other.ipc_handle, sizeof(ipc_handle));

      other.initialized = false;
    }
    return *this;
  }
  ~allreducer() {
    if (initialized) {
      std::cerr << "Warning: allreducer destroyed without calling release()"
                << std::endl;
    }
  }

  void init(sycl::queue& queue, uint32_t rank_in, uint32_t world_in) {
    if (initialized)
      return;

    if (!load_level_zero_library()) {
      throw std::runtime_error("Failed to initialize Level Zero");
    }

    zeCheck_dynamic(zeInit_dynamic(0));
    int tmp_rank, tmp_world;

    tmp_world = world_in;
    tmp_rank = rank_in;

    rank = tmp_rank;
    world = tmp_world;
    initialized = true;
  }
  void allreduce(sycl::queue& queue, void* inout_buffer, uint32_t size) {}
  void release(sycl::queue& queue) {
    if (!initialized)
      return;
    try {
      auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
          queue.get_context());
      for (int i = 0; i < world; i++) {
        if (i != rank) {
          zeCheck_dynamic(zeMemCloseIpcHandle_dynamic(
              l0_ctx, (char*)buffers[i] - offsets[i]));
        }
      }
    } catch (const std::exception& e) {
      std::cerr << "Warning: Level Zero cleanup failed: " << e.what()
                << std::endl;
    }
    sycl::free(buffers[rank], queue);
    initialized = false;
  }

  void debug_print_buffer(sycl::queue& queue, int* address, int count) {
    auto host_ptr = (int*)sycl::malloc_host(count * sizeof(int), queue);
    auto tmp_ptr = (int*)sycl::malloc_device(count * sizeof(int), queue);

    queue.memcpy(tmp_ptr, address, count * sizeof(int));
    queue.memcpy(host_ptr, tmp_ptr, count * sizeof(int));

    queue.wait();

    for (int i = 0; i < count; i++) {
      std::cout << host_ptr[i] << " ";
    }
    std::cout << std::endl;
  }
  // buffer_size as element size
  void exchange_peer_ipc_mem(sycl::queue& queue, void* ptr) {
    if (!load_level_zero_library()) {
      throw std::runtime_error("Level Zero not available");
    }

    // Step 1: Get base address of the pointer
    sycl::context ctx = queue.get_context();
    auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

    void* base_addr;
    size_t base_size;
    zeCheck_dynamic(
        zeMemGetAddressRange_dynamic(l0_ctx, ptr, &base_addr, &base_size));

    // Step 2: Get IPC mem handle from base address
    alignas(64) exchange_contents send_buf;
    alignas(64) exchange_contents recv_buf[world];

    // fill in the exchange info
    zeCheck_dynamic(
        zeMemGetIpcHandle_dynamic(l0_ctx, base_addr, &send_buf.ipc_handle));
    send_buf.offset = (char*)ptr - (char*)base_addr;

    send_buf.pid = getpid();

    // Step 3: Exchange the handles and offsets
    memset(recv_buf, 0, sizeof(recv_buf));
    // Overkill if we don't really needs all peer's handles
    un_allgather(&send_buf, recv_buf, rank, world);
    for (uint32_t i = 0; i < world; i++) {
      // Step 4: Prepare pid file descriptor of next process
      auto* peer = recv_buf + i;
      // Step 6: Open IPC handle of remote peer
      auto l0_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
          queue.get_device());
      void* peer_base;

      zeCheck_dynamic(zeMemOpenIpcHandle_dynamic(
          l0_ctx,
          l0_device,
          peer->ipc_handle,
          ZE_IPC_MEMORY_FLAG_BIAS_CACHED,
          &peer_base));

      buffers[i] = (char*)peer_base + peer->offset;
      offsets[i] = peer->offset;
      ipc_handle[i] = send_buf.ipc_handle;
    }
  }

  bool initialized;
  void* buffers[max_rank];
  void* sync_buffer[max_rank];
  size_t offsets[max_rank];
  ze_ipc_mem_handle_t ipc_handle[max_rank];
  int rank, world;
  int size_per_buffer;
  int data_size_per_buffer;
  int buffer_index;
};
