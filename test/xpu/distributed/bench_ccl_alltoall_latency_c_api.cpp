/*
 * bench_ccl_alltoall_latency_c_api.cpp (v2 - with onecclSetDevice fix)
 *
 * AllToAll kernel-time latency benchmark using oneCCL **C API** (NCCL-like).
 * Requires oneAPI 2026.0+ (oneCCL 2022.0).
 *
 * C API call:
 *   onecclAllToAll(send, recv, count, dtype, comm, stream)
 *   where count = elements per rank (each rank sends count elements to every other rank)
 *   Total send = count * ws, Total recv = count * ws
 *
 * busBW factor: (n-1)/n (each rank sends (n-1)*count elements out, receives (n-1)*count in)
 * Size column reports total bytes per rank (count * ws * sizeof(dtype))
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>
#include <oneapi/ccl.h>
#include <mpi.h>

using bf16 = sycl::ext::oneapi::bfloat16;

#define ONECCL_CHECK(stmt)                                                    \
    do {                                                                      \
        onecclResult_t _r = (stmt);                                           \
        if (_r != onecclSuccess) {                                            \
            std::fprintf(stderr, "oneCCL error %d at %s:%d (%s)\n",           \
                         (int)_r, __FILE__, __LINE__, #stmt);                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
    } while (0)

static std::string fmt_size(double bytes) {
    char buf[32];
    if      (bytes >= (double)(1LL<<30)) std::snprintf(buf,sizeof(buf),"%.2f GB",bytes/(double)(1LL<<30));
    else if (bytes >= (double)(1LL<<20)) std::snprintf(buf,sizeof(buf),"%.2f MB",bytes/(double)(1LL<<20));
    else if (bytes >= (double)(1LL<<10)) std::snprintf(buf,sizeof(buf),"%.2f KB",bytes/(double)(1LL<<10));
    else                                  std::snprintf(buf,sizeof(buf),"%.0f B", bytes);
    return std::string(buf);
}

// AllToAll busBW factor: (n-1)/n
static double busbw_factor(int ws) {
    double n = static_cast<double>(ws);
    return (n - 1.0) / n;
}

static double algbw_gbs(double bytes, double us) {
    return us > 0.0 ? bytes / (us * 1e-6) / 1e9 : 0.0;
}

static void print_table_header() {
    std::printf("  %-12s  %10s %10s %10s %10s  %12s\n",
        "Size", "avg_us", "min_us", "max_us", "var_us", "busBW(GB/s)");
    std::printf("  %-12s  %10s %10s %10s %10s  %12s\n",
        "------------", "----------", "----------", "----------", "----------",
        "------------");
}

static void print_table_row(double bytes, double avg_us, double min_us,
                            double max_us, double var_us, int ws) {
    double alg = algbw_gbs(bytes, avg_us);
    double bus = alg * busbw_factor(ws);
    std::printf("  %-12s  %10.2f %10.2f %10.2f %10.2f  %12.3f\n",
        fmt_size(bytes).c_str(), avg_us, min_us, max_us, var_us, bus);
}

struct BenchResult {
    std::vector<double> barrier_us;
    double prefill_us;
};

template<typename SubmitFn>
static BenchResult bench_loop(sycl::queue& q,
                              int warmup, int loop,
                              MPI_Comm comm,
                              bf16* prefill_buf,
                              size_t prefill_n,
                              int prefill_reps,
                              SubmitFn submit) {
    for (int i = 0; i < warmup; ++i) {
        submit();
        q.wait();
    }
    MPI_Barrier(comm);

    std::vector<sycl::event> pre_evs;
    std::vector<sycl::event> post_evs;
    pre_evs.reserve(loop);
    post_evs.reserve(loop);

    sycl::event prefill_ev;
    {
        auto* p = prefill_buf;
        int   reps = prefill_reps;
        prefill_ev = q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>{prefill_n}, [=](sycl::id<1> id) {
                bf16 v = p[id[0]];
                for (int r = 0; r < reps; ++r)
                    v = v * bf16(1.001f) + bf16(0.001f);
                p[id[0]] = v;
            });
        });
    }

    for (int i = 0; i < loop; ++i) {
        pre_evs.push_back(q.single_task([]{} ));
        submit();
        post_evs.push_back(q.single_task([]{} ));
    }
    q.wait();
    MPI_Barrier(comm);

    const int measure = std::min(loop, 50);
    const int skip    = loop - measure;
    std::vector<double> barrier_us(measure);
    for (int i = 0; i < measure; ++i) {
        uint64_t pe = pre_evs[skip + i].get_profiling_info<
            sycl::info::event_profiling::command_end>();
        uint64_t ps = post_evs[skip + i].get_profiling_info<
            sycl::info::event_profiling::command_start>();
        barrier_us[i] = (ps >= pe)
                        ? static_cast<double>(ps - pe) / 1e3
                        : 0.0;
    }

    uint64_t pf0 = prefill_ev.get_profiling_info<
        sycl::info::event_profiling::command_start>();
    uint64_t pf1 = prefill_ev.get_profiling_info<
        sycl::info::event_profiling::command_end>();
    double prefill_us = (pf1 >= pf0) ? static_cast<double>(pf1 - pf0) / 1e3 : 0.0;

    return {barrier_us, prefill_us};
}

int main(int argc, char** argv) {
    int min_log2 = 12, max_log2 = 28, step = 1;
    int warmup = 20, loop = 100;
    size_t prefill_n = 64ULL * 1024 * 1024;
    int    prefill_reps = 100;

    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i],"--min")         && i+1<argc) min_log2     = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--max")         && i+1<argc) max_log2     = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--step")        && i+1<argc) step         = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--warmup")      && i+1<argc) warmup       = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--loop")        && i+1<argc) loop         = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--prefill")     && i+1<argc) prefill_n    = (size_t)std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i],"--prefill-reps")&& i+1<argc) prefill_reps = std::atoi(argv[++i]);
    }

    MPI_Init(&argc, &argv);
    int rank, ws;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ws);

    // Get local rank for device selection
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);

    // CRITICAL: Set device BEFORE comm init (following acceptance_sycl.cpp pattern)
    ONECCL_CHECK(onecclSetDevice(local_rank));

    // Create SYCL queue for the selected device
    auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (local_rank >= (int)gpu_devices.size()) {
        std::fprintf(stderr, "Error: local_rank %d >= num_gpus %zu\n", local_rank, gpu_devices.size());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    sycl::queue q{gpu_devices[local_rank],
                  sycl::property_list{sycl::property::queue::enable_profiling{},
                                      sycl::property::queue::in_order{}}};
    if (rank == 0) {
        auto dev = q.get_device();
        std::printf("[rank %d] %s\n", rank, dev.get_info<sycl::info::device::name>().c_str());
        std::fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Create oneCCL communicator (C API)
    onecclComm_t comm;
    {
        onecclUniqueId uid;
        if (rank == 0) ONECCL_CHECK(onecclGetUniqueId(&uid));
        MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
        ONECCL_CHECK(onecclCommInitRank(&comm, ws, uid, rank));
    }

    // For AllToAll: each rank has count elements to send to each other rank
    // Total send buffer = count * ws elements
    // Total recv buffer = count * ws elements
    // We use 2^lg as the total elements per rank (count * ws)
    // So count = numel / ws
    const int64_t max_numel = (int64_t)1 << max_log2;

    bf16* buf_send  = sycl::malloc_device<bf16>(max_numel, q);
    bf16* buf_recv  = sycl::malloc_device<bf16>(max_numel, q);
    bf16* prefill   = sycl::malloc_device<bf16>(prefill_n, q);
    q.memset(buf_send, 0, max_numel * sizeof(bf16));
    q.memset(buf_recv, 0, max_numel * sizeof(bf16));
    q.memset(prefill,  0, prefill_n * sizeof(bf16));
    q.wait();

    const int measure = std::min(loop, 50);
    const int kFields = measure + 1;
    std::vector<double> gbuf(rank == 0 ? ws * kFields : 1, 0.0);
    std::vector<double> my_g(kFields, 0.0);

    if (rank == 0) {
        const std::string sep(90, '=');
        std::printf("\n%s\n", sep.c_str());
        std::printf("  AllToAll Latency Benchmark  (oneCCL C API / barrier timing)\n");
        std::printf("  world_size=%d  dtype=bfloat16  warmup=%d  loop=%d  measure(last)=%d\n",
                    ws, warmup, loop, measure);
        std::printf("  prefill: %zu M bf16 elements × %d reps/elem\n",
                    prefill_n >> 20, prefill_reps);
        std::printf("  sizes (total per rank): 2^%d .. 2^%d bf16 elements  (%s .. %s)\n",
                    min_log2, max_log2,
                    fmt_size((double)((int64_t)1<<min_log2)*2).c_str(),
                    fmt_size((double)((int64_t)1<<max_log2)*2).c_str());
        std::printf("  per-rank send to each peer = total / %d\n", ws);
        std::printf("  Timing method B (barrier): post.cmd_start − pre.cmd_end\n");
        std::printf("%s\n", sep.c_str());
        std::fflush(stdout);
    }

    struct RowData { double bytes; double avg_us, min_us, max_us, var_us; };
    std::vector<RowData> table_rows;

    for (int lg = min_log2; lg <= max_log2; lg += step) {
        const int64_t numel = (int64_t)1 << lg;  // total elements per rank
        if (numel % ws != 0) continue;
        const int64_t count = numel / ws;        // elements to send to each peer
        const double  bytes = static_cast<double>(numel) * 2.0;  // total bytes per rank

        BenchResult res = bench_loop(q, warmup, loop, MPI_COMM_WORLD,
                                     prefill, prefill_n, prefill_reps,
            [&]() {
                ONECCL_CHECK(onecclAllToAll(
                    buf_send, buf_recv,
                    static_cast<size_t>(count),
                    onecclBfloat16,
                    comm,
                    static_cast<void*>(&q)));
            });

        for (int i = 0; i < measure; ++i) my_g[i] = res.barrier_us[i];
        my_g[measure] = res.prefill_us;
        MPI_Gather(my_g.data(), kFields, MPI_DOUBLE,
                   (rank == 0) ? gbuf.data() : nullptr, kFields, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);

        if (rank == 0) {
            double sum_avg = 0, sum_min = 0, sum_max = 0, sum_var = 0;
            for (int i = 0; i < measure; ++i) {
                double iter_min = gbuf[i];
                double iter_max = iter_min;
                double iter_sum = iter_min;
                for (int r = 1; r < ws; ++r) {
                    double val = gbuf[r * kFields + i];
                    iter_min = std::min(iter_min, val);
                    iter_max = std::max(iter_max, val);
                    iter_sum += val;
                }
                sum_min += iter_min; sum_max += iter_max;
                sum_avg += iter_sum / ws;
                sum_var += (iter_max - iter_min);
            }
            table_rows.push_back({bytes,
                sum_avg / measure, sum_min / measure,
                sum_max / measure, sum_var / measure});
            std::fflush(stdout);
        }
    }

    if (rank == 0) {
        std::printf("\n\n  -- alltoall latency summary (C API / barrier) --\n\n");
        print_table_header();
        for (const auto& row : table_rows)
            print_table_row(row.bytes, row.avg_us, row.min_us,
                            row.max_us, row.var_us, ws);
        std::printf("\n");
        std::fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    q.wait();
    sycl::free(buf_send, q);
    sycl::free(buf_recv, q);
    sycl::free(prefill,  q);
    q.wait();
    ONECCL_CHECK(onecclCommDestroy(comm));
    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return 0;
}
