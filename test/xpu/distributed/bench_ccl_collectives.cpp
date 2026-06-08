/*
 * bench_ccl_collectives.cpp  (v3)
 *
 * Pure C++/SYCL/oneCCL latency benchmark for small messages (≤ 2 MiB):
 *   - allreduce     (SUM, bf16)
 *   - allgatherv    (bf16)
 *   - reduce_scatter(SUM, bf16)
 *
 * Timing (two independent measurements per CCL call):
 *   event(ms)  = SYCL device kernel time  (post-bookend.command_start −
 *                pre-bookend.command_end on the same in-order queue)
 *   call(ms)   = host wall-clock time     (CCL submit → ccl::event::wait())
 *
 * The post-bookend is submitted to the SYCL queue BEFORE ccl_event.wait(),
 * so event(ms) is unaffected by host polling latency.
 *
 * Output style mirrors bench_c10d_xccl.py (Python/torch path):
 *   one section per op, rows by size, algBW + busBW in GB/s (base-10),
 *   cross-rank averages in the table, per-rank detail as indented comment.
 *
 *   BusBW correction (nccl-tests):
 *     allreduce:     2*(n-1)/n × algBW
 *     allgather:     (n-1)/n   × algBW
 *     reduce_scatter:(n-1)/n   × algBW
 *
 * CLI:
 *   --min  N      log2 of minimum numel  (default 7  → 128 elems = 256 B)
 *   --max  N      log2 of maximum numel  (default 20 → 1M  elems = 2 MB)
 *   --step N      stride in log2 space   (default 1)
 *   --warmup N    warmup iterations      (default 20)
 *   --iters  N    timed  iterations      (default 100)
 *   --op   LIST   comma-separated: ar, ag, rs  (default ar,ag,rs)
 *
 * Build:
 *   icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -std=c++17 \
 *        -I${CCL_ROOT}/include -I${I_MPI_ROOT}/include \
 *        -L${CCL_ROOT}/lib -L${I_MPI_ROOT}/lib \
 *        -lccl -lmpi \
 *        bench_ccl_collectives.cpp -o bench_ccl_collectives
 *
 * Run examples:
 *   ZE_AFFINITY_MASK=0,1,2,3 CCL_ATL_TRANSPORT=mpi \
 *   mpirun -n 4 ./bench_ccl_collectives
 *   mpirun -n 4 ./bench_ccl_collectives --op ar
 *   mpirun -n 4 ./bench_ccl_collectives --op ag,rs --min 10 --max 20
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>
#include <oneapi/ccl.hpp>

using bf16 = sycl::ext::oneapi::bfloat16;

// ---------------------------------------------------------------------------
// Empty SYCL kernels used as timeline bookends for device-time measurement.
// Submitted to the same in-order queue as CCL ops:
//   pre  — submitted before the CCL call; marks start of device timeline
//   post — submitted after  the CCL call (before host wait); marks end
// With an in-order queue:  kern_time = post.command_start − pre.command_end
// ---------------------------------------------------------------------------
struct PreBookend  { void operator()() const noexcept {} };
struct PostBookend { void operator()() const noexcept {} };

// ---------------------------------------------------------------------------
// Op selection bitmask
// ---------------------------------------------------------------------------
enum OpBit : int { kAR = 1, kAG = 2, kRS = 4 };

struct OpDef { int bit; const char* name; };
constexpr OpDef kOpDefs[] = {
    {kAR, "allreduce"},
    {kAG, "allgather"},
    {kRS, "reduce_scatter"},
};

static int parse_ops(const char* s) {
    int ops = 0;
    std::string tok;
    std::istringstream ss(s);
    while (std::getline(ss, tok, ',')) {
        if      (tok == "ar") ops |= kAR;
        else if (tok == "ag") ops |= kAG;
        else if (tok == "rs") ops |= kRS;
    }
    return ops ? ops : (kAR | kAG | kRS);
}

// ---------------------------------------------------------------------------
// Output helpers  (mirrors bench_c10d_xccl.py conventions)
// ---------------------------------------------------------------------------

// Human-readable byte size
static std::string format_size(double bytes) {
    char buf[32];
    if      (bytes >= (double)(1LL << 30)) std::snprintf(buf, sizeof(buf), "%.2f GB", bytes / (double)(1LL << 30));
    else if (bytes >= (double)(1LL << 20)) std::snprintf(buf, sizeof(buf), "%.2f MB", bytes / (double)(1LL << 20));
    else if (bytes >= (double)(1LL << 10)) std::snprintf(buf, sizeof(buf), "%.2f KB", bytes / (double)(1LL << 10));
    else                                    std::snprintf(buf, sizeof(buf), "%.0f B",  bytes);
    return std::string(buf);
}

// BusBW correction factor — nccl-tests convention
static double busbw_factor(int op_bit, int ws) {
    double n = static_cast<double>(ws);
    if (op_bit == kAR) return 2.0 * (n - 1.0) / n;  // allreduce
    return (n - 1.0) / n;                             // allgather / reduce_scatter
}

// AlgBW in GB/s (base-10)
static double algbw_gbs(double bytes, double event_us) {
    return event_us > 0.0 ? bytes / (event_us * 1e-6) / 1e9 : 0.0;
}

// Print the column header for one op section
static void print_section_header() {
    std::printf("  %-22s %12s  %12s %12s  %14s %14s\n",
        "Op", "Size", "event(ms)", "call(ms)", "algBW(GB/s)", "busBW(GB/s)");
    std::printf("  %-22s %12s  %12s %12s  %14s %14s\n",
        "----------------------", "------------",
        "------------", "------------",
        "--------------", "--------------");
}

// Print one data row (cross-rank averages)
static void print_row(const char* op_name, double bytes,
                      double avg_event_us, double avg_call_us,
                      int op_bit, int ws) {
    double alg = algbw_gbs(bytes, avg_event_us);
    double bus = alg * busbw_factor(op_bit, ws);
    std::printf("  %-22s %12s  %12.5f %12.5f  %14.3f %14.3f\n",
        op_name,
        format_size(bytes).c_str(),
        avg_event_us / 1000.0,
        avg_call_us  / 1000.0,
        alg, bus);
}

// Per-rank detail line with op+size label:
//   "  [allreduce  2.00 KB]  r0=0.020/0.015  r1=...  (call/event ms)"
static void print_rank_detail(const char* op_name, const char* size_str,
                              const double* gbuf, int ws) {
    std::printf("  [%-14s %12s] ", op_name, size_str);
    for (int r = 0; r < ws; ++r)
        std::printf(" r%d=%.3f/%.3f", r,
            gbuf[r * 2 + 0] / 1000.0,
            gbuf[r * 2 + 1] / 1000.0);
    std::printf("  (call/event ms)\n");
}

// ---------------------------------------------------------------------------
// Timed run for one CCL op.
//
//   submit — callable () -> ccl::event
//            submits CCL work to the SYCL queue and returns WITHOUT blocking
//
// Per-iteration flow:
//   1. q.single_task<PreBookend>   → enqueued before CCL
//   2. submit()                    → CCL kernels enqueued after pre
//   3. q.single_task<PostBookend>  → enqueued after CCL (before host wait)
//   4. ccl_ev.wait()               → host waits for GPU completion
//   5. post_e.wait()               → ensures profiling timestamps are valid
//   6. kern_us = (post.command_start − pre.command_end) / 1000
//   7. host_us = wall-clock span of steps 2-4
//
// Returns {avg_host_us, avg_kern_us} over `iters` iterations.
// ---------------------------------------------------------------------------
using clk = std::chrono::high_resolution_clock;
static double to_us(clk::duration d) {
    return std::chrono::duration<double, std::micro>(d).count();
}

template <typename SubmitFn>
static std::pair<double, double>
timed_run(sycl::queue& q, int warmup, int iters, MPI_Comm comm, SubmitFn submit)
{
    for (int i = 0; i < warmup; ++i) submit().wait();
    MPI_Barrier(comm);

    double sum_call = 0.0, sum_event = 0.0;
    for (int i = 0; i < iters; ++i) {
        sycl::event pre = q.single_task<PreBookend>(PreBookend{});

        auto t0 = clk::now();
        ccl::event ccl_ev = submit();                                  // enqueue CCL (async)
        double call_us = to_us(clk::now() - t0);                       // host API dispatch time only

        sycl::event post  = q.single_task<PostBookend>(PostBookend{}); // enqueue post
        ccl_ev.wait();                                                 // block host (for event profiling)

        post.wait();  // ensure GPU done; profiling timestamps now valid

        uint64_t t_pre  = pre.get_profiling_info<
            sycl::info::event_profiling::command_end>();
        uint64_t t_post = post.get_profiling_info<
            sycl::info::event_profiling::command_start>();
        double event_us = (t_post >= t_pre)
                          ? static_cast<double>(t_post - t_pre) / 1000.0
                          : 0.0;

        sum_call  += call_us;
        sum_event += event_us;
    }
    return {sum_call / iters, sum_event / iters};  // {call_us, event_us}
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int min_log2 = 7, max_log2 = 20, warmup = 20, iters = 100, step = 1;
    int ops = kAR | kAG | kRS;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&](int d) { return (i + 1 < argc) ? std::atoi(argv[++i]) : d; };
        if      (a == "--min")                min_log2 = nxt(min_log2);
        else if (a == "--max")                max_log2 = nxt(max_log2);
        else if (a == "--warmup")             warmup   = nxt(warmup);
        else if (a == "--iters")              iters    = nxt(iters);
        else if (a == "--step")               step     = nxt(step);
        else if (a == "--op" && i+1 < argc)   ops      = parse_ops(argv[++i]);
    }

    // ---- MPI ----
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int rank = 0, ws = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ws);

    // ---- SYCL: one device per rank, profiling enabled ----
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if ((int)devs.size() < ws) {
        if (rank == 0)
            std::fprintf(stderr,
                "bench_ccl: need %d GPUs visible, found %zu\n", ws, devs.size());
        MPI_Finalize();
        return 1;
    }
    sycl::device  dev = devs[rank];
    sycl::context ctx(dev);
    sycl::queue   q(ctx, dev, {
        sycl::property::queue::in_order{},
        sycl::property::queue::enable_profiling{}
    });

    // ---- Allocate buffers (max size, reused across all sizes) ----
    //   buf_full  — allreduce src, reducescatter full-send, (allgather full-recv)
    //   buf_out   — allreduce dst, allgather full-recv
    //   buf_chunk — allgather per-rank send (numel/ws), reducescatter recv (numel/ws)
    const int64_t max_numel = (int64_t)1 << max_log2;
    bf16* buf_full  = sycl::malloc_device<bf16>(max_numel, q);
    bf16* buf_out   = sycl::malloc_device<bf16>(max_numel, q);
    bf16* buf_chunk = sycl::malloc_device<bf16>(max_numel, q);  // size ≥ max_numel/ws
    q.memset(buf_full,  0, max_numel * sizeof(bf16));
    q.memset(buf_out,   0, max_numel * sizeof(bf16));
    q.memset(buf_chunk, 0, max_numel * sizeof(bf16));
    q.wait();
    MPI_Barrier(MPI_COMM_WORLD);

    // ---- oneCCL setup ----
    ccl::init();
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type kvs_addr{};
    if (rank == 0) {
        kvs      = ccl::create_main_kvs();
        kvs_addr = kvs->get_address();
    }
    MPI_Bcast(kvs_addr.data(), kvs_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank != 0) kvs = ccl::create_kvs(kvs_addr);
    auto ccl_dev    = ccl::create_device(dev);
    auto ccl_ctx    = ccl::create_context(ctx);
    auto ccl_comm   = ccl::create_communicator(ws, rank, ccl_dev, ccl_ctx, kvs);
    auto ccl_stream = ccl::create_stream(q);

    // ---- Banner ----
    if (rank == 0) {
        const std::string sep(86, '=');
        std::printf("\n%s\n", sep.c_str());
        std::printf("  Collective Benchmark  (oneCCL native / SYCL + SYCL event profiling)\n");
        std::printf("  world_size=%d  dtype=bfloat16  iters=%d  warmup=%d\n", ws, iters, warmup);
        std::printf("  sizes: 2^%d .. 2^%d bf16 elements", min_log2, max_log2);
        std::printf("  (%.0f B .. %s)\n",
            static_cast<double>((int64_t)1 << min_log2) * 2.0,
            format_size(static_cast<double>((int64_t)1 << max_log2) * 2.0).c_str());
        std::printf("  ops:");
        for (const auto& od : kOpDefs)
            if (ops & od.bit) std::printf(" %s", od.name);
        std::printf("\n");
        std::printf("  event(ms) = SYCL device kernel time  "
                    "call(ms) = host wall-clock latency\n");
        std::printf("%s\n", sep.c_str());
        std::fflush(stdout);
    }

    // Gather buffer: 2 doubles per rank [call_us, event_us] for the current op
    std::vector<double> gbuf(rank == 0 ? ws * 2 : 1, 0.0);
    double my_g[2] = {};

    // Row accumulator for the clean per-op table printed after all sizes
    struct RowData { double bytes, avg_event_us, avg_call_us; };

    // ---- Outer loop: one section per op ----
    for (const auto& od : kOpDefs) {
        if (!(ops & od.bit)) continue;

        std::vector<RowData> table_rows;

        // ---- Phase 1: run all sizes, print per-rank details immediately ----
        if (rank == 0) {
            std::printf("\n  -- per-rank detail (call/event ms) --\n");
            std::fflush(stdout);
        }

        for (int lg = min_log2; lg <= max_log2; lg += step) {
            const int64_t numel = (int64_t)1 << lg;
            if (numel % ws != 0) continue;

            const int64_t chunk = numel / ws;
            const double  bytes = static_cast<double>(numel) * 2.0;
            const std::vector<size_t> ag_counts(ws, static_cast<size_t>(chunk));

            my_g[0] = my_g[1] = 0.0;

            if (od.bit == kAR) {
                auto [c, e] = timed_run(q, warmup, iters, MPI_COMM_WORLD,
                    [&]() -> ccl::event {
                        return ccl::allreduce(
                            buf_full, buf_out, static_cast<size_t>(numel),
                            ccl::datatype::bfloat16, ccl::reduction::sum,
                            ccl_comm, ccl_stream);
                    });
                my_g[0] = c;  my_g[1] = e;
            } else if (od.bit == kAG) {
                auto [c, e] = timed_run(q, warmup, iters, MPI_COMM_WORLD,
                    [&]() -> ccl::event {
                        return ccl::allgatherv(
                            buf_chunk, static_cast<size_t>(chunk),
                            buf_out, ag_counts,
                            ccl::datatype::bfloat16,
                            ccl_comm, ccl_stream);
                    });
                my_g[0] = c;  my_g[1] = e;
            } else {  // kRS
                auto [c, e] = timed_run(q, warmup, iters, MPI_COMM_WORLD,
                    [&]() -> ccl::event {
                        return ccl::reduce_scatter(
                            buf_full, buf_chunk, static_cast<size_t>(chunk),
                            ccl::datatype::bfloat16, ccl::reduction::sum,
                            ccl_comm, ccl_stream);
                    });
                my_g[0] = c;  my_g[1] = e;
            }

            // Gather per-rank timing to rank 0
            MPI_Gather(my_g, 2, MPI_DOUBLE,
                       (rank == 0) ? gbuf.data() : nullptr, 2, MPI_DOUBLE,
                       0, MPI_COMM_WORLD);

            if (rank == 0) {
                // Cross-rank averages
                double avg_call = 0.0, avg_event = 0.0;
                for (int r = 0; r < ws; ++r) {
                    avg_call  += gbuf[r * 2 + 0];
                    avg_event += gbuf[r * 2 + 1];
                }
                avg_call  /= ws;
                avg_event /= ws;

                // Immediately print per-rank detail
                print_rank_detail(od.name, format_size(bytes).c_str(),
                                  gbuf.data(), ws);
                std::fflush(stdout);

                // Accumulate for clean table
                table_rows.push_back({bytes, avg_event, avg_call});
            }
        }  // sizes

        // ---- Phase 2: print clean table for this op ----
        if (rank == 0) {
            std::printf("\n");
            print_section_header();
            for (const auto& row : table_rows)
                print_row(od.name, row.bytes, row.avg_event_us, row.avg_call_us,
                          od.bit, ws);
            std::printf("\n");
            std::fflush(stdout);
        }
    }  // ops

    // ---- Cleanup ----
    sycl::free(buf_full,  q);
    sycl::free(buf_out,   q);
    sycl::free(buf_chunk, q);
    MPI_Finalize();
    return 0;
}
