/**
 * Test case for P2P cache coherency between ranks using MPI + Level-Zero IPC.
 *
 * Key Technical Points:
 * - Level-Zero IPC handles contain file descriptors that MUST be passed via
 *   Unix Domain Sockets with SCM_RIGHTS (not via MPI!)
 *
 * Build: icpx -fsycl -I${MPI_HOME}/include -L${MPI_HOME}/lib -lmpi -lze_loader -o test_p2p_cache_coherency test_p2p_cache_coherency.cpp
 * Run: mpirun -np 2 ./test_p2p_cache_coherency
 */

#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <mpi.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <pwd.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>

constexpr size_t BUFFER_SIZE = 1024;
constexpr uint32_t RANK0_VALUE = 100;
constexpr uint32_t RANK1_LOCAL_VALUE = 111;
constexpr int NUM_ITERATIONS = 1;

#define ZE_CHECK(cmd) do {                           \
    ze_result_t e = cmd;                             \
    if (e != ZE_RESULT_SUCCESS) {                    \
        std::cerr << "Level-Zero error at " << __FILE__ << ":" << __LINE__ \
                  << " code=" << e << std::endl;     \
        MPI_Abort(MPI_COMM_WORLD, 1);                \
    }                                                \
} while(0)

// LSC fence for Intel GPUs - evict cache
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
#define __LscFlushCache() \
    __asm__ __volatile__("lsc_fence.ugm.evict.gpu")
#else
#define __LscFlushCache() do {} while(0)
#endif

// ==================== Unix Socket IPC Helpers ====================

int create_socket(const std::string& sockname) {
    unlink(sockname.c_str());
    sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sockname.c_str(), sizeof(addr.sun_path) - 1);

    int sock = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (sock == -1) { perror("socket"); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (bind(sock, (sockaddr*)&addr, sizeof(addr)) == -1) { perror("bind"); MPI_Abort(MPI_COMM_WORLD, 1); }
    return sock;
}

void send_fd(int socket, const std::string& remote_sockname, int fd) {
    sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, remote_sockname.c_str(), sizeof(addr.sun_path) - 1);

    char buf[3] = "fd";
    iovec io = {.iov_base = buf, .iov_len = 2};
    char cbuf[CMSG_SPACE(sizeof(int))];
    memset(cbuf, 0, sizeof(cbuf));

    msghdr msg = {
        .msg_name = (void*)&addr, .msg_namelen = sizeof(addr),
        .msg_iov = &io, .msg_iovlen = 1,
        .msg_control = cbuf, .msg_controllen = sizeof(cbuf)
    };

    auto cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    memcpy(CMSG_DATA(cmsg), &fd, sizeof(int));

    for (int retry = 0; retry < 100; retry++) {
        if (sendmsg(socket, &msg, 0) > 0) return;
        if (errno == ENOENT || errno == ECONNREFUSED) { usleep(10000); continue; }
        break;
    }
    perror("sendmsg"); MPI_Abort(MPI_COMM_WORLD, 1);
}

int recv_fd(int socket) {
    char buf[2];
    iovec io = {.iov_base = buf, .iov_len = sizeof(buf)};
    char cbuf[CMSG_SPACE(sizeof(int))];
    memset(cbuf, 0, sizeof(cbuf));

    msghdr msg = {
        .msg_name = nullptr, .msg_namelen = 0,
        .msg_iov = &io, .msg_iovlen = 1,
        .msg_control = cbuf, .msg_controllen = sizeof(cbuf)
    };

    if (recvmsg(socket, &msg, 0) <= 0) { perror("recvmsg"); MPI_Abort(MPI_COMM_WORLD, 1); }

    auto cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg || cmsg->cmsg_type != SCM_RIGHTS) {
        std::cerr << "Invalid control message" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int fd;
    memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
    return fd;
}

// ==================== Level-Zero Helpers ====================

ze_device_handle_t get_ze_device(const sycl::device& dev) {
    return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
}

ze_context_handle_t get_ze_context(const sycl::context& ctx) {
    return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
}

std::vector<sycl::device> get_sycl_devices() {
    std::vector<sycl::device> devices;
    auto platform_list = sycl::platform::get_platforms();
    for (const auto& platform : platform_list) {
        bool is_level_zero = platform.get_backend() == sycl::backend::ext_oneapi_level_zero;
        if (!is_level_zero) continue;
        auto device_list = platform.get_devices();
        constexpr auto partition_by_affinity =
            sycl::info::partition_property::partition_by_affinity_domain;
        constexpr auto next_partitionable =
            sycl::info::partition_affinity_domain::next_partitionable;
        for (auto& device : device_list) {
            if (device.is_gpu()) {
                auto max_sub_devices = device.get_info<sycl::info::device::partition_max_sub_devices>();
                if (max_sub_devices == 0) {
                    devices.push_back(device);
                } else {
                    auto sub_devices = device.create_sub_devices<partition_by_affinity>(next_partitionable);
                    devices.insert(devices.end(), sub_devices.begin(), sub_devices.end());
                }
            }
        }
    }
    return devices;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    ZE_CHECK(zeInit(0));

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 2) {
        if (rank == 0) std::cerr << "This test requires exactly 2 MPI ranks." << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::vector<sycl::device> devices = get_sycl_devices();
    if (devices.size() < 2) {
        if (rank == 0) std::cerr << "Need at least 2 GPU devices/tiles." << std::endl;
        MPI_Finalize();
        return 1;
    }

    sycl::device my_device = devices[rank];
    sycl::context ctx(my_device);
    sycl::queue queue(ctx, my_device, {sycl::property::queue::in_order{}});
    auto l0_ctx = get_ze_context(ctx);
    auto l0_dev = get_ze_device(my_device);

    std::cout << "[Rank " << rank << "] Using device: "
              << my_device.get_info<sycl::info::device::name>() << std::endl;

    // Create Unix socket for IPC exchange
    auto uid = getuid();
    auto pwd = getpwuid(uid);
    char my_sockname[128], remote_sockname[128];
    snprintf(my_sockname, sizeof(my_sockname), "/tmp/p2p-test-rank%d_%s", rank, pwd->pw_name);
    snprintf(remote_sockname, sizeof(remote_sockname), "/tmp/p2p-test-rank%d_%s", 1 - rank, pwd->pw_name);
    int my_sock = create_socket(my_sockname);

    MPI_Barrier(MPI_COMM_WORLD);
    int mismatch_count = 0;

    // ===== SETUP PHASE: Allocate buffers and exchange IPC handles (once) =====
    uint32_t* data_buffer = nullptr;
    uint32_t* flag_buffer = nullptr;
    uint32_t* result = nullptr;
    uint32_t* remote_data = nullptr;
    uint32_t* remote_flag = nullptr;
    void* remote_data_base = nullptr;
    void* remote_flag_base = nullptr;

    if (rank == 1) {
        // Rank 1: Allocate buffers
        data_buffer = sycl::malloc_device<uint32_t>(BUFFER_SIZE, queue);
        flag_buffer = sycl::malloc_device<uint32_t>(BUFFER_SIZE, queue);
        result = sycl::malloc_host<uint32_t>(BUFFER_SIZE, queue);

        queue.memset(data_buffer, 0, BUFFER_SIZE * sizeof(uint32_t)).wait();
        queue.memset(flag_buffer, 0, BUFFER_SIZE * sizeof(uint32_t)).wait();

        // Pollute cache with local value
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(BUFFER_SIZE, 64),
                [=](sycl::nd_item<1> item) {
                    size_t i = item.get_global_linear_id();
                    data_buffer[i] = RANK1_LOCAL_VALUE;
                });
        }).wait();

        // Get IPC handles
        void* data_base; size_t data_size;
        void* flag_base; size_t flag_size;
        ZE_CHECK(zeMemGetAddressRange(l0_ctx, data_buffer, &data_base, &data_size));
        ZE_CHECK(zeMemGetAddressRange(l0_ctx, flag_buffer, &flag_base, &flag_size));

        size_t data_offset = (char*)data_buffer - (char*)data_base;
        size_t flag_offset = (char*)flag_buffer - (char*)flag_base;

        std::cout << "[Rank 1] data_buffer=" << data_buffer << ", base=" << data_base
                  << ", offset=" << data_offset << std::endl;
        std::cout << "[Rank 1] flag_buffer=" << flag_buffer << ", base=" << flag_base
                  << ", offset=" << flag_offset << std::endl;

        ze_ipc_mem_handle_t data_ipc, flag_ipc;
        ZE_CHECK(zeMemGetIpcHandle(l0_ctx, data_base, &data_ipc));
        ZE_CHECK(zeMemGetIpcHandle(l0_ctx, flag_base, &flag_ipc));

        int data_fd = *reinterpret_cast<int*>(&data_ipc);
        int flag_fd = *reinterpret_cast<int*>(&flag_ipc);
        send_fd(my_sock, remote_sockname, data_fd);
        send_fd(my_sock, remote_sockname, flag_fd);
        MPI_Send(&data_offset, sizeof(size_t), MPI_BYTE, 0, 10, MPI_COMM_WORLD);
        MPI_Send(&flag_offset, sizeof(size_t), MPI_BYTE, 0, 11, MPI_COMM_WORLD);
        std::cout << "[Rank 1] Sent IPC handles" << std::endl;

    } else { // rank == 0
        int data_fd = recv_fd(my_sock);
        int flag_fd = recv_fd(my_sock);
        size_t data_offset, flag_offset;
        MPI_Recv(&data_offset, sizeof(size_t), MPI_BYTE, 1, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&flag_offset, sizeof(size_t), MPI_BYTE, 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "[Rank 0] Received IPC handles, offsets: data=" << data_offset
                  << ", flag=" << flag_offset << std::endl;

        ze_ipc_mem_handle_t data_ipc, flag_ipc;
        memset(&data_ipc, 0, sizeof(data_ipc));
        memset(&flag_ipc, 0, sizeof(flag_ipc));
        *reinterpret_cast<int*>(&data_ipc) = data_fd;
        *reinterpret_cast<int*>(&flag_ipc) = flag_fd;

        ZE_CHECK(zeMemOpenIpcHandle(l0_ctx, l0_dev, data_ipc, ZE_IPC_MEMORY_FLAG_BIAS_CACHED, &remote_data_base));
        ZE_CHECK(zeMemOpenIpcHandle(l0_ctx, l0_dev, flag_ipc, ZE_IPC_MEMORY_FLAG_BIAS_CACHED, &remote_flag_base));

        remote_data = (uint32_t*)((char*)remote_data_base + data_offset);
        remote_flag = (uint32_t*)((char*)remote_flag_base + flag_offset);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ===== ITERATION PHASE: Rank 0 updates, Rank 1 checks =====
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        uint32_t expected_flag = iter + 1;
        uint32_t expected_data = RANK0_VALUE + iter;

        if (rank == 0) {
            std::cout << "\n=== Iteration " << iter << ": writing data=" << expected_data
                      << ", flag=" << expected_flag << " ===" << std::endl;

            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::nd_range<1>(BUFFER_SIZE, 64),
                    [=](sycl::nd_item<1> item) {
                        size_t i = item.get_global_linear_id();
                        remote_data[i] = expected_data;
                        __LscFlushCache();
                        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::system,
                                         sycl::access::address_space::global_space> flag_ref(remote_flag[i]);
                        flag_ref.store(expected_flag);
                    });
            }).wait();
        } else if (rank == 1) {
            constexpr uint32_t MAX_SPIN = 100000000;
            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::nd_range<1>(BUFFER_SIZE, 64),
                    [=](sycl::nd_item<1> item) {
                        size_t i = item.get_global_linear_id();
                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);

                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::system,
                                         sycl::access::address_space::global_space> flag_ref(flag_buffer[i]);
                        uint32_t spin = 0;
                        while (flag_ref.load() < expected_flag && spin < MAX_SPIN) { spin++; }

                        result[i] = data_buffer[i];
                    });
            }).wait();

            int local_mismatch = 0;
            for (size_t i = 0; i < BUFFER_SIZE; ++i) {
                if (result[i] != expected_data) {
                    if (local_mismatch < 3) {
                        std::cout << "[Rank 1] [" << i << "] Expected: " << expected_data
                                  << ", Got: " << result[i] << std::endl;
                    }
                    local_mismatch++;
                }
            }

            if (local_mismatch > 0) {
                std::cout << "[Rank 1] Iter " << iter << " FAIL: " << local_mismatch
                          << "/" << BUFFER_SIZE << " mismatches" << std::endl;
                mismatch_count++;
            } else {
                std::cout << "[Rank 1] Iter " << iter << " PASS" << std::endl;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ===== CLEANUP =====
    if (rank == 1) {
        sycl::free(data_buffer, queue);
        sycl::free(flag_buffer, queue);
        sycl::free(result, queue);
    } else {
        ZE_CHECK(zeMemCloseIpcHandle(l0_ctx, remote_data_base));
    }

    // Cleanup socket
    close(my_sock);
    unlink(my_sockname);

    // Summary
    if (rank == 1) {
        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Iterations with cache coherency issues: " << mismatch_count
                  << "/" << NUM_ITERATIONS << std::endl;
        if (mismatch_count > 0) {
            std::cout << "WARNING: Cache coherency issues detected! "
                      << "Consider using uncached load on the reader side." << std::endl;
        }
    }

    MPI_Finalize();
    return mismatch_count > 0 ? 1 : 0;
}
