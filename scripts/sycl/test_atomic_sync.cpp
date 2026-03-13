/*
 * Intel GPU Symmetric Memory Demo using Level-Zero IPC
 * 
 * This file reproduces sycl queue.submit hanging issue between two devices. 
 * Whilst sycl queue.submit works fine with address from local device. It hangs when read/write IPC address from remote device, even if the two processes are on the same node.
 * As common sense, the queue.submit should never hang though the kernel itself may hang.
 * 
 * Compilation:
 *   icpx -fsycl -fsycl-targets=spir64 sycl_submit_hang.cpp -lmpi -lze_loader
 * 
 * Run:
 *   mpiexec -n 2 ./a.out
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

// Send file descriptor over Unix socket using SCM_RIGHTS mechanism
// This is the ONLY correct way to pass IPC handles between processes
void send_fd(int sock, int fd, int rank, size_t offset) {
  iovec iov[1];
  msghdr msg;
  auto rank_offset = std::make_pair(rank, offset);

  // Attach rank and offset as regular data
  iov[0].iov_base = &rank_offset;
  iov[0].iov_len = sizeof(rank_offset);
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;
  msg.msg_name = nullptr;
  msg.msg_namelen = 0;

  // Attach file descriptor as ancillary data (control message)
  exchange_fd cmsg(SOL_SOCKET, SCM_RIGHTS, fd);
  msg.msg_control = &cmsg;
  msg.msg_controllen = sizeof(exchange_fd);
  
  if (sendmsg(sock, &msg, 0) == -1) {
    perror("sendmsg failed");
    exit(1);
  }
}

// Receive file descriptor from Unix socket
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

// Create Unix domain socket server for receiving IPC handles
int create_server_socket(const char* sockname) {
  unlink(sockname);  // Remove old socket file if exists
  
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

// Connect to remote rank's socket server
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
  
  // Retry connection with timeout (remote server may not be ready yet)
  for (int i = 0; i < 50; i++) {
    if (connect(sock, (sockaddr*)&addr, len) == 0) {
      return sock;
    }
    usleep(100000); // 100ms
  }
  
  perror("connect failed after retries");
  exit(1);
}

// Create SYCL queue for specific GPU device
static sycl::queue create_queue(int local_rank) {
    auto platforms = sycl::platform::get_platforms();
    for (const auto &platform : platforms) {
        if (platform.get_backend() == sycl::backend::ext_oneapi_level_zero) {
            return sycl::queue(platform.get_devices()[local_rank],
                             {sycl::property::queue::in_order{}});
        }
    }
    throw std::runtime_error("Level-Zero platform not found.");
}

// Store value with release fence (for put_signal)
// Order: store first, then release fence to flush the store
inline void store_release(int32_t* addr, int32_t val) {
  *addr = val;
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
}

// Load value with acquire fence (for get_signal/wait_signal)
// Order: acquire fence first, then load to see the latest value
inline int32_t load_acquire(int32_t* addr) {
  sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
  int32_t val = *addr;
  return val;
}


int main(int argc, char** argv) {
    // ========== Step 1: Initialize MPI and Level-Zero ==========
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ZE_CHECK(zeInit(0));

    std::cout << "\n=== Rank " << rank << " / " << world_size << " ===" << std::endl;

    // ========== Step 2: Create SYCL queue and get Level-Zero handles ==========
    auto compute_queue = create_queue(rank);
    auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        compute_queue.get_context());
    auto l0_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        compute_queue.get_device());

    // ========== Step 3: Allocate and initialize device memory ==========
    constexpr size_t num_elements = 2;
    constexpr size_t bytes = num_elements * sizeof(int32_t);

    int32_t *local_ptr = sycl::malloc_device<int32_t>(num_elements, compute_queue);
    compute_queue.memset(local_ptr, 0, bytes).wait();

    std::cout << "[Step 3] Allocated memory at " << local_ptr 
              << ", initialized with value " << (rank + 1) * 100 << std::endl;

    // ========== Step 4: Get base address and calculate offset ==========
    // Level-Zero IPC works with base addresses, so we need to calculate offset
    void* base_addr;
    size_t base_size;
    ZE_CHECK(zeMemGetAddressRange(l0_ctx, local_ptr, &base_addr, &base_size));
    size_t offset = (char*)local_ptr - (char*)base_addr;

    std::cout << "[Step 4] Base address: " << base_addr 
              << ", Offset: " << offset << " bytes" << std::endl;

    // ========== Step 5: Get IPC handle and extract file descriptor ==========
    ze_ipc_mem_handle_t ipc_handle;
    ZE_CHECK(zeMemGetIpcHandle(l0_ctx, base_addr, &ipc_handle));

    // CRITICAL: IPC handle contains a file descriptor that MUST be passed
    // via Unix socket SCM_RIGHTS, not via MPI!
    int my_fd = *reinterpret_cast<int*>(&ipc_handle);
    std::cout << "[Step 5] Got IPC handle with fd: " << my_fd << std::endl;

    // ========== Step 6: Create Unix socket server for IPC exchange ==========
    auto uid = getuid();
    auto pwd = getpwuid(uid);
    char server_name[128];
    snprintf(server_name, sizeof(server_name), 
             "/tmp/ipc-demo-rank_%d_%s", rank, pwd->pw_name);
    
    int server_sock = create_server_socket(server_name);
    std::cout << "[Step 6] Created socket server: " << server_name << std::endl;

    // Synchronize before starting IPC exchange
    MPI_Barrier(MPI_COMM_WORLD);

    // ========== Step 7: Exchange IPC handles with remote rank ==========
    int remote_rank = (rank + 1) % world_size;
    char remote_server[128];
    snprintf(remote_server, sizeof(remote_server),
             "/tmp/ipc-demo-rank_%d_%s", remote_rank, pwd->pw_name);
    
    // Connect to remote rank's server
    int client_sock = connect_to_server(remote_server);
    std::cout << "[Step 7] Connected to rank " << remote_rank << std::endl;

    // Send our IPC handle to remote rank
    send_fd(client_sock, my_fd, rank, offset);
    std::cout << "[Step 7] Sent IPC handle to rank " << remote_rank << std::endl;

    // Receive IPC handle from remote rank
    int accept_sock = accept(server_sock, nullptr, nullptr);
    auto [remote_fd, remote_rank_id, remote_offset] = recv_fd(accept_sock);
    
    std::cout << "[Step 7] Received IPC handle (fd=" << remote_fd 
              << ") from rank " << remote_rank_id << std::endl;

    // ========== Step 8: Reconstruct and open remote IPC handle ==========
    // Reconstruct IPC handle using the received file descriptor
    ze_ipc_mem_handle_t remote_ipc_handle = ipc_handle;
    *reinterpret_cast<int*>(&remote_ipc_handle) = remote_fd;

    // Open IPC handle to get remote memory pointer
    // Use BIAS_CACHED for better performance
    void* remote_base;
    ZE_CHECK(zeMemOpenIpcHandle(l0_ctx, l0_device, remote_ipc_handle,
                                ZE_IPC_MEMORY_FLAG_BIAS_CACHED, &remote_base));
    
    int32_t* remote_ptr = (int32_t*)((char*)remote_base + remote_offset);
    std::cout << "[Step 8] Opened remote memory at " << remote_ptr << std::endl;

    // ========== Step 9: Access both local and remote memory ==========
    // std::vector<int32_t> host_local(num_elements);
    // std::vector<int32_t> host_remote(num_elements);
    
    // compute_queue.memset(host_local.data(), local_ptr, bytes).wait();
    // compute_queue.memset(host_remote.data(), remote_ptr, bytes).wait();

    // std::cout << "[Step 9] SUCCESS! Local[0]=" << host_local[0]
    //           << ", Remote[0]=" << host_remote[0] 
    //           << " (from rank " << remote_rank_id << ")" << std::endl;
    for (int i = 0; i < 100000; i++) {
      std::cout << "node: " << rank << ", running iteration " << i << "======================" <<std::endl;
      // barrier_sync
      compute_queue.submit([&](sycl::handler &h) {
        sycl::stream out(1024, 256, h);
        std::cout << "node: " << rank <<", barrier_async submit 1" << std::endl;
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(num_elements), sycl::range<1>(num_elements)), [=](sycl::nd_item<1> item) {
            int target_rank = item.get_local_id(0);
            if (target_rank == rank) {
                return;
            }
             if (target_rank < world_size) {
               if (target_rank == rank ) {
                   return;
               }
               // step1: put signal - write to remote buffer
//                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
//                int32_t *sync_buffer_dst = remote_ptr + rank;
//                *sync_buffer_dst = 1; // to global memory

               int32_t *sync_buffer_dst = remote_ptr + rank;
                while (load_acquire(sync_buffer_dst) != 0) {}
                // Set signal to 1 with release semantics
                store_release(sync_buffer_dst, 1);

                 // step2: wait for my own signal to be updated
                sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                int32_t *wait_ptr = local_ptr + target_rank;
                while (load_acquire(wait_ptr) != 1) {}
                store_release(wait_ptr, 0);
             }
        });
        std::cout << "node: " << rank <<", barrier_async submit 2" << std::endl;
      });
      std::cout << "node: " << rank <<", barrier_async submitted" << std::endl;
      compute_queue.wait();
      std::cout << "node: " << rank << ", iteration " << i << " done======================" <<std::endl;
    }

    // ========== Step 10: Cleanup ==========
    ZE_CHECK(zeMemCloseIpcHandle(l0_ctx, remote_base));
    sycl::free(local_ptr, compute_queue);
    close(accept_sock);
    close(client_sock);
    close(server_sock);
    unlink(server_name);

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "\n=== Demo completed successfully! ===" << std::endl;
        std::cout << "Key takeaway: IPC handles must be passed via Unix sockets, not MPI!" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
