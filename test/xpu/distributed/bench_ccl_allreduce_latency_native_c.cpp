/*
 * bench_ccl_allreduce_latency_native_c.cpp
 *
 * Allreduce kernel-time latency benchmark for small messages (native C API / SYCL).
 *
 * This is the C API variant of bench_ccl_allreduce_latency.cpp.
 * Instead of the oneCCL C++ API (oneapi/ccl.hpp), it uses the oneCCL C API
 * (oneapi/ccl.h) with functions like onecclAllReduce(), onecclCommInitRank(),
 * etc.  See ../../../src/xccl/xccl.cpp for reference usage.
 *
 * Since the C API does not return ccl::event objects, the ccl_event-based
 * timing (Method A) is not available.  Only barrier-based timing (Method B:
 * post_barrier.command_start − pre_barrier.command_end) is used.
 *
 * Measurement strategy
 * ─────────────────────
 * For each message size:
 *   1. warmup iterations (fully synchronous, q.wait() every iter)
 *   2. Submit a large "prefill" element-wise kernel to keep the GPU busy
 *      while the host dispatches the entire timing loop.
 *   3. Fire-and-forget loop (NO per-iter wait):
 *         for i in [0, loop):
 *             pre_barrier  (single_task)
 *             onecclAllReduce(...)
 *             post_barrier (single_task)
 *   4. q.wait()  — single sync point for all GPU work
 *   5. Discard the first (loop - 50) iterations; only use last 50 for stats.
 *   6. Read profiling timestamps from barrier sycl::events:
 *         per_iter_us[i] = post_evs[i].command_start − pre_evs[i].command_end
 *   7. MPI_Gather per-rank per-iter data to rank 0.
 *   8. Output:
 *        • per-rank per-iter detail (measured iterations, all ranks)
 *        • clean summary table: avg_us, min_us, max_us, algBW, busBW
 *
 * Prefill kernel
 * ──────────────
 * A parallel_for over `prefill_n` bf16 elements, each doing `prefill_reps`
 * rounds of `v = v * 1.001 + 0.001`.  Tune so GPU time ≈ loop × dispatch_us.
 * Default: prefill_n = 64M, prefill_reps = 100  → ~5-10 ms on BMG.
 *
 * CLI
 * ───
 *   --min  N          log2 min numel (default 12 → 4K elems = 8 KB)
 *   --max  N          log2 max numel (default 28 → 256M elems = 512 MB)
 *   --step N          stride in log2 (default 1)
 *   --warmup N        warmup iters   (default 20)
 *   --loop   N        timed iters    (default 100)
 *   --prefill-n N     prefill elements (default 67108864 = 64M)
 *   --prefill-reps N  compute reps per element (default 100)
 *
 * Build
 * ─────
 *   icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -std=c++17 \
 *        -I${CCL_ROOT}/include -I${I_MPI_ROOT}/include \
 *        -L${CCL_ROOT}/lib -L${I_MPI_ROOT}/lib \
 *        -lccl -lmpi \
 *        bench_ccl_allreduce_latency_native_c.cpp -o bench_ccl_allreduce_latency_native_c
 *
 * Run
 * ───
 *   ZE_AFFINITY_MASK=0,1,2,3 CCL_ATL_TRANSPORT=mpi \
 *   mpirun -n 4 ./bench_ccl_allreduce_latency_native_c
 *   mpirun -n 4 ./bench_ccl_allreduce_latency_native_c --min 4 --max 14 --loop 200
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>
#include <oneapi/ccl.h>
#include <mpi.h>

using bf16 = sycl::ext::oneapi::bfloat16;

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------
static std::string fmt_size(double bytes) {
    char buf[32];
    if      (bytes >= (double)(1LL<<30)) std::snprintf(buf,sizeof(buf),"%.2f GB",bytes/(double)(1LL<<30));
    else if (bytes >= (double)(1LL<<20)) std::snprintf(buf,sizeof(buf),"%.2f MB",bytes/(double)(1LL<<20));
    else if (bytes >= (double)(1LL<<10)) std::snprintf(buf,sizeof(buf),"%.2f KB",bytes/(double)(1LL<<10));
    else                                  std::snprintf(buf,sizeof(buf),"%.0f B", bytes);
    return std::string(buf);
}

// BusBW correction: allreduce = 2*(n-1)/n
static double busbw_factor(int ws) {
    double n = static_cast<double>(ws);
    return 2.0 * (n - 1.0) / n;
}

// AlgBW in GB/s (base-10)
static double algbw_gbs(double bytes, double us) {
    return us > 0.0 ? bytes / (us * 1e-6) / 1e9 : 0.0;
}

static void print_table_header() {
    std::printf("  %-12s  %10s %10s %10s %10s  %12s\n",
        "Size", "avg_us", "min_us", "max_us", "var_us",
        "busBW(GB/s)");
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

// ---------------------------------------------------------------------------
// Bench result (barrier timing only, C API has no ccl::event)
// ---------------------------------------------------------------------------
struct BenchResult {
    std::vector<double> barrier_us;   // barrier: post.start - pre.end (µs)
    double prefill_us;                // GPU execution time of the prefill kernel (µs)
};

// ---------------------------------------------------------------------------
// bench_allreduce: fire-and-forget loop, timing via barrier kernels
// ---------------------------------------------------------------------------
template<typename SubmitFn>
static BenchResult bench_allreduce(sycl::queue& q,
                                   int warmup, int loop,
                                   MPI_Comm comm,
                                   bf16* prefill_buf,
                                   size_t prefill_n,
                                   int prefill_reps,
                                   SubmitFn submit) {
    // ── warmup (fully synchronous) ─────────────────────────────────────────
    for (int i = 0; i < warmup; ++i) {
        submit();
        q.wait();
    }
    q.wait();
    MPI_Barrier(comm);

    // ── allocate event storage ────────────────────────────────────────────
    std::vector<sycl::event> pre_evs;
    std::vector<sycl::event> post_evs;
    pre_evs.reserve(loop);
    post_evs.reserve(loop);

    // ── prefill kernel: keeps GPU busy during host loop dispatch ──────────
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

    // ── fire-and-forget timed loop with barrier kernels ────────────────────
    for (int i = 0; i < loop; ++i) {
        // pre-barrier: single_task on the in-order queue
        pre_evs.push_back(q.single_task([]{} ));
        // allreduce via C API
        submit();
        // post-barrier: single_task on the in-order queue
        post_evs.push_back(q.single_task([]{} ));
    }

    // ── single synchronize ────────────────────────────────────────────────
    q.wait();
    MPI_Barrier(comm);

    // ── read profiling timestamps, only use the LAST 50 iterations ────────
    const int measure = std::min(loop, 50);
    const int skip    = loop - measure;  // discard early iterations

    std::vector<double> barrier_us(measure);

    for (int i = 0; i < measure; ++i) {
        // Barrier-based: post.command_start - pre.command_end
        uint64_t pre_end = pre_evs[skip + i].get_profiling_info<
            sycl::info::event_profiling::command_end>();
        uint64_t post_start = post_evs[skip + i].get_profiling_info<
            sycl::info::event_profiling::command_start>();
        barrier_us[i] = (post_start >= pre_end)
                        ? static_cast<double>(post_start - pre_end) / 1e3
                        : 0.0;
    }

    // ── prefill GPU time ───────────────────────────────────────────────────
    uint64_t pf0 = prefill_ev.get_profiling_info<
        sycl::info::event_profiling::command_start>();
    uint64_t pf1 = prefill_ev.get_profiling_info<
        sycl::info::event_profiling::command_end>();
    double prefill_us = (pf1 >= pf0) ? static_cast<double>(pf1 - pf0) / 1e3 : 0.0;

    return {barrier_us, prefill_us};
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int min_log2    = 12;   // 4K elements = 8 KB
    int max_log2    = 28;   // 256M elements = 512 MB
    int step        = 1;
    int warmup      = 20;
    int loop        = 100;
    size_t prefill_n    = 64ULL * 1024 * 1024;  // 64M bf16 elements
    int    prefill_reps = 100;

    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i],"--min")         && i+1<argc) min_log2     = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--max")         && i+1<argc) max_log2     = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--step")        && i+1<argc) step         = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--warmup")      && i+1<argc) warmup       = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--loop")        && i+1<argc) loop         = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--prefill-n")   && i+1<argc) prefill_n    = static_cast<size_t>(std::atoll(argv[++i]));
        else if (!std::strcmp(argv[i],"--prefill-reps")&& i+1<argc) prefill_reps = std::atoi(argv[++i]);
    }

    // ── MPI ────────────────────────────────────────────────────────────────
    int mpi_provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_provided);
    int rank, ws;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ws);

    // ── SYCL: one GPU per rank, profiling enabled ─────────────────────────
    auto devs = sycl::platform{sycl::gpu_selector_v}.get_devices();
    if ((int)devs.size() < ws) {
        if (rank == 0)
            std::fprintf(stderr, "bench: need %d GPUs visible, found %zu\n",
                         ws, devs.size());
        MPI_Finalize();
        return 1;
    }
    sycl::device  dev = devs[rank];
    sycl::context ctx(dev);
    sycl::queue   q{ctx, dev,
                    {sycl::property::queue::in_order{},
                     sycl::property::queue::enable_profiling{}}};

    // ── oneCCL C API init ─────────────────────────────────────────────────
    onecclUniqueId uniqueId;
    if (rank == 0) {
        onecclResult_t res = onecclGetUniqueId(&uniqueId);
        if (res != onecclSuccess) {
            std::fprintf(stderr, "onecclGetUniqueId failed: %s\n",
                         onecclGetErrorString(res));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Set the device for this rank
    onecclResult_t res = onecclSetDevice(static_cast<uint32_t>(rank));
    if (res != onecclSuccess) {
        std::fprintf(stderr, "rank %d: onecclSetDevice failed: %s\n",
                     rank, onecclGetErrorString(res));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Create communicator
    onecclComm_t ccl_comm = nullptr;
    res = onecclCommInitRank(&ccl_comm, static_cast<size_t>(ws), uniqueId, rank);
    if (res != onecclSuccess) {
        std::fprintf(stderr, "rank %d: onecclCommInitRank failed: %s\n",
                     rank, onecclGetErrorString(res));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ── Allocate buffers ──────────────────────────────────────────────────
    const int64_t max_numel = (int64_t)1 << max_log2;
    bf16* buf_in    = sycl::malloc_device<bf16>(max_numel, q);
    bf16* buf_out   = sycl::malloc_device<bf16>(max_numel, q);
    bf16* prefill   = sycl::malloc_device<bf16>(prefill_n, q);
    q.memset(buf_in,  0, max_numel * sizeof(bf16));
    q.memset(buf_out, 0, max_numel * sizeof(bf16));
    q.memset(prefill, 0, prefill_n * sizeof(bf16));
    q.wait();

    // ── Gather buffer: [barrier_per_us[0..m-1], prefill_us]
    //    Only last 50 iterations are measured; earlier ones are discarded.
    const int measure = std::min(loop, 50);
    const int kFields = measure + 1;
    std::vector<double> gbuf(rank == 0 ? ws * kFields : 1, 0.0);
    std::vector<double> my_g(kFields, 0.0);

    // ── Banner ────────────────────────────────────────────────────────────
    if (rank == 0) {
        const std::string sep(90, '=');
        std::printf("\n%s\n", sep.c_str());
        std::printf("  Allreduce Latency Benchmark  "
                    "(oneCCL C API / barrier timing)\n");
        std::printf("  world_size=%d  dtype=bfloat16  warmup=%d  loop=%d  measure(last)=%d\n",
                    ws, warmup, loop, measure);
        std::printf("  prefill: %zu M bf16 elements × %d reps/elem\n",
                    prefill_n >> 20, prefill_reps);
        std::printf("  sizes: 2^%d .. 2^%d bf16 elements  (%s .. %s)\n",
                    min_log2, max_log2,
                    fmt_size((double)((int64_t)1<<min_log2)*2).c_str(),
                    fmt_size((double)((int64_t)1<<max_log2)*2).c_str());
        std::printf("  Timing method: barrier (post_barrier.command_start − pre_barrier.command_end)\n");
        std::printf("  NOTE: C API does not return ccl::event; ccl_event timing unavailable.\n");
        std::printf("%s\n", sep.c_str());
        std::fflush(stdout);
    }

    // ── Summary table accumulator ─────────────────────────────────────────
    struct RowData {
        double bytes;
        double avg_us, min_us, max_us, var_us;
    };
    std::vector<RowData> table_rows;

    if (rank == 0) {
        std::printf("\n");
        std::fflush(stdout);
    }

    // ── Size loop ─────────────────────────────────────────────────────────
    for (int lg = min_log2; lg <= max_log2; lg += step) {
        const int64_t numel = (int64_t)1 << lg;
        const double  bytes = static_cast<double>(numel) * 2.0;  // bf16 = 2 bytes

        BenchResult bres = bench_allreduce(q, warmup, loop, MPI_COMM_WORLD,
                                          prefill, prefill_n, prefill_reps,
            [&]() {
                onecclAllReduce(
                    static_cast<void*>(buf_in),
                    static_cast<void*>(buf_out),
                    static_cast<size_t>(numel),
                    onecclBfloat16,
                    onecclSum,
                    ccl_comm,
                    static_cast<void*>(&q));
            });

        // Pack: [barrier_per_us[0..m-1], prefill_us]
        for (int i = 0; i < measure; ++i) my_g[i] = bres.barrier_us[i];
        my_g[measure] = bres.prefill_us;

        MPI_Gather(my_g.data(), kFields, MPI_DOUBLE,
                   (rank == 0) ? gbuf.data() : nullptr, kFields, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::string sz = fmt_size(bytes);
            std::printf("\n  [allreduce %12s]  prefill_us:", sz.c_str());
            for (int r = 0; r < ws; ++r) {
                const double* rv = gbuf.data() + r * kFields;
                std::printf(" r%d=%.0f", r, rv[measure]);
            }
            std::printf("\n");

            // Print per-rank per-iteration raw results
            for (int r = 0; r < ws; ++r) {
                const double* rv = gbuf.data() + r * kFields;
                std::printf("    r%d barrier: [", r);
                for (int i = 0; i < measure; ++i) {
                    if (i > 0) std::printf("  ");
                    std::printf("%.2f", rv[i]);
                }
                std::printf("]\n");
            }

            // Per-iteration: min/max across ranks, then average over iterations
            double sum_avg = 0.0, sum_min = 0.0, sum_max = 0.0, sum_var = 0.0;
            for (int i = 0; i < measure; ++i) {
                double iter_min = gbuf[0 * kFields + i];
                double iter_max = gbuf[0 * kFields + i];
                double iter_sum = gbuf[0 * kFields + i];
                for (int r = 1; r < ws; ++r) {
                    double val = gbuf[r * kFields + i];
                    iter_min = std::min(iter_min, val);
                    iter_max = std::max(iter_max, val);
                    iter_sum += val;
                }
                sum_min += iter_min;
                sum_max += iter_max;
                sum_avg += iter_sum / ws;
                sum_var += (iter_max - iter_min);
            }

            table_rows.push_back({bytes,
                sum_avg / measure, sum_min / measure,
                sum_max / measure, sum_var / measure});
            std::fflush(stdout);
        }
    }  // sizes

    // ── Summary table ─────────────────────────────────────────────────────
    if (rank == 0) {
        std::printf("\n\n  -- allreduce latency summary (C API / barrier timing) --\n\n");
        print_table_header();
        for (const auto& row : table_rows)
            print_table_row(row.bytes, row.avg_us, row.min_us,
                            row.max_us, row.var_us, ws);

        std::printf("\n");
        std::fflush(stdout);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────
    MPI_Barrier(MPI_COMM_WORLD);
    q.wait();
    sycl::free(buf_in,  q);
    sycl::free(buf_out, q);
    sycl::free(prefill, q);
    q.wait();
    onecclCommDestroy(ccl_comm);
    MPI_Finalize();
    return 0;
}
