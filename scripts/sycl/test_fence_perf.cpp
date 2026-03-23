/*
 * Intel GPU P2P Store & Check Benchmark using Level-Zero IPC
 *
 * This file benchmarks the latency of P2P flag signaling:
 *   Signal remote flag -> Wait for local flag from remote
 *
 * Compilation:
 *   icpx -fsycl -fsycl-targets=spir64 test_p2p_store_check.cpp -lmpi -lze_loader -o test_p2p_store_check
 *
 * Run:
 *   mpiexec -n 2 ./test_p2p_store_check [num_iterations]
 *
 * Single kernel per iteration: store local -> signal remote -> wait signal -> check remote
 */

#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <mpi.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <pwd.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>

#define ZE_CHECK(cmd) do {                           \
    ze_result_t e = cmd;                             \
    if( e != ZE_RESULT_SUCCESS ) {                   \
        printf("Level-Zero error at %s:%d code=%d\n", \
                __FILE__,__LINE__, e);                \
        exit(EXIT_FAILURE);                           \
    }                                                 \
} while(0)

// Helper structure for passing file descriptors via Unix sockets
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
  }
};

void send_fd(int sock, int fd, int rank, size_t offset) {
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

  if (sendmsg(sock, &msg, 0) == -1) {
    perror("sendmsg failed");
    exit(1);
  }
}

std::tuple<int, int, size_t> recv_fd(int sock) {
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

  if (recvmsg(sock, &msg, 0) == -1) {
    perror("recvmsg failed");
    exit(1);
  }

  return std::make_tuple(cmsg.fd, rank_offset.first, rank_offset.second);
}

int create_server_socket(const char* sockname) {
  unlink(sockname);

  sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, sockname, sizeof(addr.sun_path) - 1);

  int sock = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sock == -1) {
    perror("socket creation failed");
    exit(1);
  }

  auto size = offsetof(sockaddr_un, sun_path) + strlen(addr.sun_path);
  if (bind(sock, (sockaddr*)&addr, size) == -1) {
    perror("bind failed");
    exit(1);
  }

  if (listen(sock, 10) == -1) {
    perror("listen failed");
    exit(1);
  }

  return sock;
}

int connect_to_server(const char* sockname) {
  sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, sockname, sizeof(addr.sun_path) - 1);

  int sock = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sock == -1) {
    perror("socket creation failed");
    exit(1);
  }

  auto len = offsetof(sockaddr_un, sun_path) + strlen(addr.sun_path);

  for (int i = 0; i < 50; i++) {
    if (connect(sock, (sockaddr*)&addr, len) == 0) {
      return sock;
    }
    usleep(100000);
  }

  perror("connect failed after retries");
  exit(1);
}

static sycl::queue create_queue(int local_rank) {
    auto platforms = sycl::platform::get_platforms();
    for (const auto &platform : platforms) {
        if (platform.get_backend() == sycl::backend::ext_oneapi_level_zero) {
            return sycl::queue(platform.get_devices()[local_rank],
                             {sycl::property::queue::in_order{},
                              sycl::property::queue::enable_profiling{}});
        }
    }
    throw std::runtime_error("Level-Zero platform not found.");
}

int main(int argc, char** argv) {
    // ========== Step 1: Initialize MPI and Level-Zero ==========
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Parse command-line arguments
    int num_iterations = 100;          // default 100 iterations
    int warmup_iterations = 10;        // warmup iterations

    if (argc > 1) num_iterations = std::atoi(argv[1]);

    ZE_CHECK(zeInit(0));

    if (rank == 0) {
        std::cout << "=== P2P Flag Signaling Benchmark ===" << std::endl;
        std::cout << "  num_iterations:  " << num_iterations << std::endl;
        std::cout << "  warmup:          " << warmup_iterations << std::endl;
        std::cout << "  world_size:      " << world_size << std::endl;
    }

    // ========== Step 2: Create SYCL queue and get Level-Zero handles ==========
    auto compute_queue = create_queue(rank);
    auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        compute_queue.get_context());
    auto l0_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        compute_queue.get_device());

    std::cout << "[Rank " << rank << "] Using device: "
              << compute_queue.get_device().get_info<sycl::info::device::name>() << std::endl;

    // ========== Step 3: Allocate device memory ==========
    // Sync flag buffer (1 element per rank)
    int32_t *local_flag = sycl::malloc_device<int32_t>(1, compute_queue);
    compute_queue.memset(local_flag, 0, sizeof(int32_t)).wait();

    // ========== Step 4: Get base address and calculate offset ==========
    void* flag_base_addr;
    size_t flag_base_size;
    ZE_CHECK(zeMemGetAddressRange(l0_ctx, local_flag, &flag_base_addr, &flag_base_size));
    size_t flag_offset = (char*)local_flag - (char*)flag_base_addr;

    // ========== Step 5: Get IPC handle ==========
    ze_ipc_mem_handle_t flag_ipc_handle;
    ZE_CHECK(zeMemGetIpcHandle(l0_ctx, flag_base_addr, &flag_ipc_handle));
    int my_flag_fd = *reinterpret_cast<int*>(&flag_ipc_handle);

    // ========== Step 6: Setup Unix socket and exchange IPC handles ==========
    auto uid = getuid();
    auto pwd = getpwuid(uid);
    char server_name[128];
    snprintf(server_name, sizeof(server_name),
             "/tmp/ipc-store-check-rank_%d_%s", rank, pwd->pw_name);

    int server_sock = create_server_socket(server_name);

    MPI_Barrier(MPI_COMM_WORLD);

    int remote_rank = (rank + 1) % world_size;
    char remote_server[128];
    snprintf(remote_server, sizeof(remote_server),
             "/tmp/ipc-store-check-rank_%d_%s", remote_rank, pwd->pw_name);

    // Send flag IPC handle
    int client_sock = connect_to_server(remote_server);
    send_fd(client_sock, my_flag_fd, rank, flag_offset);

    int accept_sock = accept(server_sock, nullptr, nullptr);
    auto [remote_flag_fd, remote_rank_id, remote_flag_offset] = recv_fd(accept_sock);
    close(accept_sock);
    close(client_sock);

    // ========== Step 7: Open remote IPC handle ==========
    ze_ipc_mem_handle_t remote_flag_ipc_handle = flag_ipc_handle;
    *reinterpret_cast<int*>(&remote_flag_ipc_handle) = remote_flag_fd;

    void* remote_flag_base;
    ZE_CHECK(zeMemOpenIpcHandle(l0_ctx, l0_device, remote_flag_ipc_handle,
                                ZE_IPC_MEMORY_FLAG_BIAS_CACHED, &remote_flag_base));

    // local_flag: remote writes here to signal me
    // remote_flag: I write here to signal remote
    int32_t* remote_flag = (int32_t*)((char*)remote_flag_base + remote_flag_offset);

    std::cout << "[Rank " << rank << "] Local flag: " << local_flag
              << ", Remote flag: " << remote_flag << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    // ========== Step 8: Benchmark - Single Kernel: Signal + Wait Flag ==========
    std::vector<double> kernel_times_us;
    kernel_times_us.reserve(num_iterations);

    int total_iters = warmup_iterations + num_iterations;

    for (int iter = 0; iter < total_iters; iter++) {
        // Clear flag
        compute_queue.memset(local_flag, 0, sizeof(int32_t)).wait();
        MPI_Barrier(MPI_COMM_WORLD);

        // Single kernel: signal remote flag -> wait local flag
        auto event = compute_queue.submit([&](sycl::handler &h) {
            h.single_task([=]() {
                // Signal remote: write 1 to remote_flag
                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                *remote_flag = 1;
                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);

                // Wait for remote to signal us: spin on local_flag
                while (true) {
                    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                    if (*local_flag == 1) break;
                }
            });
        });
        event.wait();

        bool is_warmup = (iter < warmup_iterations);
        if (!is_warmup) {
            auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
            double kernel_us = static_cast<double>(end - start) / 1e3;  // ns -> us
            kernel_times_us.push_back(kernel_us);
        }

        if (iter < 3) {
            const char* tag = is_warmup ? "WARMUP" : "BENCH";
            std::cout << "[Rank " << rank << "] " << tag
                      << " iter=" << iter << " PASS" << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ========== Step 9: Print Timing Statistics ==========
    std::sort(kernel_times_us.begin(), kernel_times_us.end());
    double sum = std::accumulate(kernel_times_us.begin(), kernel_times_us.end(), 0.0);
    double avg = sum / kernel_times_us.size();
    double median = kernel_times_us[kernel_times_us.size() / 2];
    double p95 = kernel_times_us[(size_t)(kernel_times_us.size() * 0.95)];
    double p99 = kernel_times_us[(size_t)(kernel_times_us.size() * 0.99)];
    double min_val = kernel_times_us.front();
    double max_val = kernel_times_us.back();

    if (rank == 0) std::cout << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "[Rank " << rank << "] Flag Signaling Latency (" << num_iterations << " iters):" << std::endl;
    printf("  avg=%8.2f us  median=%8.2f us  min=%8.2f us  max=%8.2f us  p95=%8.2f us  p99=%8.2f us\n",
           avg, median, min_val, max_val, p95, p99);

    std::cout << std::endl;

    // ========== Step 10: Cleanup ==========
    ZE_CHECK(zeMemCloseIpcHandle(l0_ctx, remote_flag_base));
    sycl::free(local_flag, compute_queue);
    close(server_sock);
    unlink(server_name);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "=== Benchmark completed ===" << std::endl;
    }

    MPI_Finalize();
    return 0;
}

