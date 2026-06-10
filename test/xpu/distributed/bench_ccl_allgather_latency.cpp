/*
 * bench_ccl_allgather_latency.cpp
 *
 * Allgather kernel-time latency benchmark (native C++/SYCL/oneCCL).
 *
 * Measurement strategy
 * ─────────────────────
 *   Mirrors bench_ccl_allreduce_latency.cpp.  Differences:
 *     • Calls ccl::allgatherv(send_buf, send_count, recv_buf, recv_counts, ...)
 *       where send_count = numel / ws (per-rank chunk) and recv buffer holds
 *       the concatenated output (numel total bf16 elements).
 *     • busBW correction for allgather = (n-1)/n.
 *     • "Size" column reports the **total output size** in bytes
 *       (i.e. numel * sizeof(bf16)), matching how allreduce reports its
 *       message size.  Each rank actually moves (n-1)/n of that on the wire.
 *
 * CLI / Build / Run: identical to bench_ccl_allreduce_latency.cpp.
 *
 *   icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -std=c++17 \
 *        -I${CCL_ROOT}/include -I${I_MPI_ROOT}/include \
 *        -L${CCL_ROOT}/lib -L${I_MPI_ROOT}/lib \
 *        -lccl -lmpi \
 *        bench_ccl_allgather_latency.cpp -o bench_ccl_allgather_latency
 *
 *   ZE_AFFINITY_MASK=0,1,2,3 CCL_ATL_TRANSPORT=mpi \
 *   mpirun -n 4 ./bench_ccl_allgather_latency
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>
#include <oneapi/ccl.hpp>
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

// BusBW correction: allgather = (n-1)/n
static double busbw_factor(int ws) {
    double n = static_cast<double>(ws);
    return (n - 1.0) / n;
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
// Bench result
// ---------------------------------------------------------------------------
struct BenchResult {
    std::vector<double> per_us;       // ccl event: command_end - command_start (µs)
    std::vector<double> barrier_us;   // barrier: post.start - pre.end (µs)
    double prefill_us;                // GPU execution time of the prefill kernel (µs)
};

// ---------------------------------------------------------------------------
// bench_allgather: fire-and-forget loop, timing via ccl::event + barrier kernels
// ---------------------------------------------------------------------------
template<typename SubmitFn>
static BenchResult bench_allgather(sycl::queue& q,
                                   int warmup, int loop,
                                   MPI_Comm comm,
                                   bf16* prefill_buf,
                                   size_t prefill_n,
                                   int prefill_reps,
                                   SubmitFn submit) {
    // ── warmup (fully synchronous) ─────────────────────────────────────────
    for (int i = 0; i < warmup; ++i) {
        submit().wait();
    }
    q.wait();
    MPI_Barrier(comm);

    // ── allocate event storage ────────────────────────────────────────────
    std::vector<ccl::event>  ccl_evs;
    std::vector<sycl::event> pre_evs;
    std::vector<sycl::event> post_evs;
    ccl_evs.reserve(loop);
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
        pre_evs.push_back(q.single_task([]{} ));
        ccl_evs.push_back(submit());
        post_evs.push_back(q.single_task([]{} ));
    }

    // ── single synchronize ────────────────────────────────────────────────
    q.wait();
    MPI_Barrier(comm);

    // ── read profiling timestamps, only use the LAST 50 iterations ────────
    const int measure = std::min(loop, 50);
    const int skip    = loop - measure;

    std::vector<double> per_us(measure);
    std::vector<double> barrier_us(measure);

    for (int i = 0; i < measure; ++i) {
        // Method A: ccl event native profiling
        sycl::event& ev = ccl_evs[skip + i].get_native();
        uint64_t t_start = ev.get_profiling_info<
            sycl::info::event_profiling::command_start>();
        uint64_t t_end = ev.get_profiling_info<
            sycl::info::event_profiling::command_end>();
        per_us[i] = (t_end >= t_start)
                    ? static_cast<double>(t_end - t_start) / 1e3
                    : 0.0;

        // Method B: barrier-based
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

    return {per_us, barrier_us, prefill_us};
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int min_log2    = 12;
    int max_log2    = 28;
    int step        = 1;
    int warmup      = 20;
    int loop        = 100;
    size_t prefill_n    = 64ULL * 1024 * 1024;
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

    // ── oneCCL init ───────────────────────────────────────────────────────
    ccl::init();
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type kvs_addr{};
    if (rank == 0) { kvs = ccl::create_main_kvs(); kvs_addr = kvs->get_address(); }
    MPI_Bcast(kvs_addr.data(), kvs_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank != 0) kvs = ccl::create_kvs(kvs_addr);
    auto ccl_dev    = ccl::create_device(dev);
    auto ccl_ctx    = ccl::create_context(ctx);
    auto ccl_comm   = ccl::create_communicator(ws, rank, ccl_dev, ccl_ctx, kvs);
    auto ccl_stream = ccl::create_stream(q);

    // ── Allocate buffers ──────────────────────────────────────────────────
    // numel = total output size in bf16 elements; each rank sends numel/ws.
    const int64_t max_numel = (int64_t)1 << max_log2;
    if (max_numel % ws != 0) {
        if (rank == 0)
            std::fprintf(stderr, "bench: max_numel (%lld) must be divisible by ws (%d)\n",
                         (long long)max_numel, ws);
        MPI_Finalize();
        return 1;
    }
    bf16* buf_chunk = sycl::malloc_device<bf16>(max_numel / ws, q);
    bf16* buf_out   = sycl::malloc_device<bf16>(max_numel,      q);
    bf16* prefill   = sycl::malloc_device<bf16>(prefill_n,      q);
    q.memset(buf_chunk, 0, (max_numel / ws) * sizeof(bf16));
    q.memset(buf_out,   0, max_numel * sizeof(bf16));
    q.memset(prefill,   0, prefill_n * sizeof(bf16));
    q.wait();

    // ── Gather buffer ─────────────────────────────────────────────────────
    const int measure = std::min(loop, 50);
    const int kFields = 2 * measure + 1;
    std::vector<double> gbuf(rank == 0 ? ws * kFields : 1, 0.0);
    std::vector<double> my_g(kFields, 0.0);

    // ── Banner ────────────────────────────────────────────────────────────
    if (rank == 0) {
        const std::string sep(90, '=');
        std::printf("\n%s\n", sep.c_str());
        std::printf("  Allgather Latency Benchmark  "
                    "(oneCCL native / dual timing)\n");
        std::printf("  world_size=%d  dtype=bfloat16  warmup=%d  loop=%d  measure(last)=%d\n",
                    ws, warmup, loop, measure);
        std::printf("  prefill: %zu M bf16 elements × %d reps/elem\n",
                    prefill_n >> 20, prefill_reps);
        std::printf("  sizes (total output): 2^%d .. 2^%d bf16 elements  (%s .. %s)\n",
                    min_log2, max_log2,
                    fmt_size((double)((int64_t)1<<min_log2)*2).c_str(),
                    fmt_size((double)((int64_t)1<<max_log2)*2).c_str());
        std::printf("  per-rank chunk = total / %d  (send_count for allgatherv)\n", ws);
        std::printf("  Timing method A (ccl_event): ccl::event command_end − command_start\n");
        std::printf("  Timing method B (barrier):   post_barrier.command_start − pre_barrier.command_end\n");
        std::printf("%s\n", sep.c_str());
        std::fflush(stdout);
    }

    // ── Summary table accumulator ─────────────────────────────────────────
    struct RowData {
        double bytes;
        double avg_us, min_us, max_us, var_us;       // method A
        double b_avg_us, b_min_us, b_max_us, b_var_us; // method B
    };
    std::vector<RowData> table_rows;

    if (rank == 0) {
        std::printf("\n");
        std::fflush(stdout);
    }

    // ── Size loop ─────────────────────────────────────────────────────────
    for (int lg = min_log2; lg <= max_log2; lg += step) {
        const int64_t numel = (int64_t)1 << lg;
        if (numel % ws != 0) continue;  // need divisible chunks

        const int64_t chunk = numel / ws;
        const double  bytes = static_cast<double>(numel) * 2.0;  // total output bytes
        const std::vector<size_t> ag_counts(ws, static_cast<size_t>(chunk));

        BenchResult res = bench_allgather(q, warmup, loop, MPI_COMM_WORLD,
                                          prefill, prefill_n, prefill_reps,
            [&]() -> ccl::event {
                return ccl::allgatherv(
                    buf_chunk, static_cast<size_t>(chunk),
                    buf_out,   ag_counts,
                    ccl::datatype::bfloat16,
                    ccl_comm, ccl_stream);
            });

        // Pack: [ccl_per_us[0..m-1], barrier_per_us[0..m-1], prefill_us]
        for (int i = 0; i < measure; ++i) my_g[i] = res.per_us[i];
        for (int i = 0; i < measure; ++i) my_g[measure + i] = res.barrier_us[i];
        my_g[2 * measure] = res.prefill_us;

        MPI_Gather(my_g.data(), kFields, MPI_DOUBLE,
                   (rank == 0) ? gbuf.data() : nullptr, kFields, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::string sz = fmt_size(bytes);
            std::printf("\n  [allgather %12s]  prefill_us:", sz.c_str());
            for (int r = 0; r < ws; ++r) {
                const double* rv = gbuf.data() + r * kFields;
                std::printf(" r%d=%.0f", r, rv[2 * measure]);
            }
            std::printf("\n");

            for (int r = 0; r < ws; ++r) {
                const double* rv = gbuf.data() + r * kFields;
                std::printf("    r%d ccl_ev: [", r);
                for (int i = 0; i < measure; ++i) {
                    if (i > 0) std::printf("  ");
                    std::printf("%.2f", rv[i]);
                }
                std::printf("]\n");
                std::printf("    r%d barrier: [", r);
                for (int i = 0; i < measure; ++i) {
                    if (i > 0) std::printf("  ");
                    std::printf("%.2f", rv[measure + i]);
                }
                std::printf("]\n");
            }

            // Method A: ccl_event
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
            // Method B: barrier
            double b_sum_avg = 0.0, b_sum_min = 0.0, b_sum_max = 0.0, b_sum_var = 0.0;
            for (int i = 0; i < measure; ++i) {
                double iter_min = gbuf[0 * kFields + measure + i];
                double iter_max = gbuf[0 * kFields + measure + i];
                double iter_sum = gbuf[0 * kFields + measure + i];
                for (int r = 1; r < ws; ++r) {
                    double val = gbuf[r * kFields + measure + i];
                    iter_min = std::min(iter_min, val);
                    iter_max = std::max(iter_max, val);
                    iter_sum += val;
                }
                b_sum_min += iter_min;
                b_sum_max += iter_max;
                b_sum_avg += iter_sum / ws;
                b_sum_var += (iter_max - iter_min);
            }

            table_rows.push_back({bytes,
                sum_avg / measure, sum_min / measure,
                sum_max / measure, sum_var / measure,
                b_sum_avg / measure, b_sum_min / measure,
                b_sum_max / measure, b_sum_var / measure});
            std::fflush(stdout);
        }
    }

    // ── Summary table ─────────────────────────────────────────────────────
    if (rank == 0) {
        std::printf("\n\n  -- allgather latency summary (method A: ccl_event) --\n\n");
        print_table_header();
        for (const auto& row : table_rows)
            print_table_row(row.bytes, row.avg_us, row.min_us,
                            row.max_us, row.var_us, ws);

        std::printf("\n\n  -- allgather latency summary (method B: barrier) --\n\n");
        print_table_header();
        for (const auto& row : table_rows)
            print_table_row(row.bytes, row.b_avg_us, row.b_min_us,
                            row.b_max_us, row.b_var_us, ws);

        std::printf("\n");
        std::fflush(stdout);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────
    MPI_Barrier(MPI_COMM_WORLD);
    q.wait();
    sycl::free(buf_chunk, q);
    sycl::free(buf_out,   q);
    sycl::free(prefill,   q);
    q.wait();
    MPI_Finalize();
    return 0;
}
