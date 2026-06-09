/*
 * bench_ccl_prefill.cpp
 *
 * GEMM-prefill fire-and-forget CCL latency benchmark (native C++/SYCL/oneCCL).
 * Mirrors bench_ccl_event.py.
 *
 * Measurement strategy
 * ─────────────────────
 * For each op × size:
 *   1. warmup iterations (fully synchronous, `ccl_ev.wait()` every iter)
 *   2. Submit a large "prefill" element-wise kernel to keep the GPU busy
 *      while the host dispatches the entire timing loop.
 *   3. Fire-and-forget loop (NO per-iter wait):
 *         for i in [0, loop):
 *             pre_evs[i]  = q.single_task<Pre>()
 *             ccl_evs[i]  = ccl_op()           ← no .wait()
 *             post_evs[i] = q.single_task<Post>()
 *   4. q.wait()  — single sync point for all GPU work
 *   5. Read SYCL event timestamps:
 *         per_iter_ms[i] = post_evs[i].command_start − pre_evs[i].command_end
 *         span_ms        = (post_evs[loop-1].command_start −
 *                           pre_evs[0].command_end) / loop
 *   6. MPI_Gather per-rank per-iter data to rank 0.
 *   7. Output:
 *        • per-rank per-iter detail (all iterations, all ranks)
 *        • clean summary table: min_ms (global min), span_ms, algBW, busBW
 *
 * Prefill kernel
 * ──────────────
 * A parallel_for over `prefill_n` bf16 elements, each doing `prefill_reps`
 * rounds of `v = v * 1.001 + 0.001`.  Tune so GPU time ≈ loop × dispatch_us.
 * Default: prefill_n = 64M, prefill_reps = 200  → ~0.5-1 ms on BMG.
 *
 * CLI
 * ───
 *   --min  N          log2 min numel (default 7  → 256 B)
 *   --max  N          log2 max numel (default 20 → 2 MB)
 *   --step N          stride in log2 (default 1)
 *   --warmup N        warmup iters   (default 20)
 *   --loop   N        timed iters    (default 50)
 *   --op   LIST       ar,ag,rs       (default all)
 *   --prefill-n N     prefill elements (default 67108864 = 64M)
 *   --prefill-reps N  compute reps per element (default 200)
 *
 * Build
 * ─────
 *   icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -std=c++17 \
 *        -I${CCL_ROOT}/include -I${I_MPI_ROOT}/include \
 *        -L${CCL_ROOT}/lib -L${I_MPI_ROOT}/lib \
 *        -lccl -lmpi \
 *        bench_ccl_prefill.cpp -o bench_ccl_prefill
 *
 * Run
 * ───
 *   ZE_AFFINITY_MASK=0,1,2,3 CCL_ATL_TRANSPORT=mpi \
 *   mpirun -n 4 ./bench_ccl_prefill
 *   mpirun -n 4 ./bench_ccl_prefill --op rs --min 7 --max 20 --loop 100
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>
#include <oneapi/ccl.hpp>
#include <mpi.h>

using bf16 = sycl::ext::oneapi::bfloat16;

// ---------------------------------------------------------------------------
// SYCL bookend kernel tags (unique names per binary)
// ---------------------------------------------------------------------------
struct PrefillPre  { void operator()() const {} };
struct PrefillPost { void operator()() const {} };

// ---------------------------------------------------------------------------
// Op selection
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

static double busbw_factor(int op_bit, int ws) {
    double n = static_cast<double>(ws);
    return (op_bit == kAR) ? 2.0*(n-1)/n : (n-1)/n;
}

static double algbw_gbs(double bytes, double us) {
    return us > 0.0 ? bytes / (us * 1e-6) / 1e9 : 0.0;
}

static void print_table_header() {
    std::printf("  %-22s %12s  %10s %10s  %14s %14s\n",
        "Op","Size","min_us","span_us","algBW(GB/s)","busBW(GB/s)");
    std::printf("  %-22s %12s  %10s %10s  %14s %14s\n",
        "----------------------","------------",
        "----------","----------",
        "--------------","--------------");
}

static void print_table_row(const char* op, double bytes,
                            double min_us, double span_us,
                            int op_bit, int ws) {
    double alg = algbw_gbs(bytes, min_us);
    double bus = alg * busbw_factor(op_bit, ws);
    std::printf("  %-22s %12s  %10.2f %10.2f  %14.3f %14.3f\n",
        op, fmt_size(bytes).c_str(), min_us, span_us, alg, bus);
}

// ---------------------------------------------------------------------------
// Bench result
// ---------------------------------------------------------------------------
struct BenchResult {
    std::vector<double> per_us;   // per_us[i] = event time for iteration i (µs)
    double span_us;               // (post[loop-1].start - pre[0].end) / loop (µs)
};

// ---------------------------------------------------------------------------
// bench_op: fire-and-forget loop with bookend events
// ---------------------------------------------------------------------------
template<typename SubmitFn>
static BenchResult bench_op(sycl::queue& q,
                             int warmup, int loop,
                             MPI_Comm comm,
                             bf16* prefill,
                             size_t prefill_n,
                             int prefill_reps,
                             SubmitFn submit) {
    // ── warmup (fully synchronous) ─────────────────────────────────────────
    for (int i = 0; i < warmup; ++i) {
        submit().wait();
    }
    q.wait();
    MPI_Barrier(comm);

    // ── allocate bookend event storage ────────────────────────────────────
    std::vector<sycl::event> pre_evs(loop), post_evs(loop);

    // ── prefill kernel: keeps GPU busy during host loop dispatch ──────────
    {
        auto* p = prefill;
        int   reps = prefill_reps;
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>{prefill_n}, [=](sycl::id<1> id) {
                bf16 v = p[id[0]];
                for (int r = 0; r < reps; ++r)
                    v = v * bf16(1.001f) + bf16(0.001f);
                p[id[0]] = v;
            });
        });
    }

    // ── fire-and-forget timed loop ─────────────────────────────────────────
    for (int i = 0; i < loop; ++i) {
        pre_evs[i]  = q.single_task<PrefillPre>(PrefillPre{});
        static_cast<void>(submit());   // enqueue CCL; discard event handle
        post_evs[i] = q.single_task<PrefillPost>(PrefillPost{});
    }

    // ── single synchronize ────────────────────────────────────────────────
    q.wait();

    // ── read timestamps (ns → µs) ──────────────────────────────────────────
    std::vector<double> per_us(loop);
    for (int i = 0; i < loop; ++i) {
        uint64_t t0 = pre_evs[i].get_profiling_info<
            sycl::info::event_profiling::command_end>();
        uint64_t t1 = post_evs[i].get_profiling_info<
            sycl::info::event_profiling::command_start>();
        per_us[i] = (t1 >= t0) ? static_cast<double>(t1 - t0) / 1e3 : 0.0;
    }

    // ── span (ns → µs): pipeline-throughput latency ────────────────────────
    // span_us = (GPU time from start of iter-0 to end of iter-(loop-1)) / loop
    // = average interval when all ops are queued back-to-back with no host gap
    uint64_t t_first = pre_evs[0].get_profiling_info<
        sycl::info::event_profiling::command_end>();
    uint64_t t_last  = post_evs[loop-1].get_profiling_info<
        sycl::info::event_profiling::command_start>();
    double span_us = (t_last >= t_first)
                     ? static_cast<double>(t_last - t_first) / 1e3 / loop
                     : 0.0;

    return {per_us, span_us};
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int min_log2    = 7;
    int max_log2    = 20;
    int step        = 1;
    int warmup      = 20;
    int loop        = 50;
    int ops         = kAR | kAG | kRS;
    size_t prefill_n    = 64ULL * 1024 * 1024;  // 64M bf16 elements
    int    prefill_reps = 200;

    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i],"--min")         && i+1<argc) min_log2    = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--max")         && i+1<argc) max_log2    = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--step")        && i+1<argc) step        = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--warmup")      && i+1<argc) warmup      = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--loop")        && i+1<argc) loop        = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--op")          && i+1<argc) ops         = parse_ops(argv[++i]);
        else if (!std::strcmp(argv[i],"--prefill-n")   && i+1<argc) prefill_n   = static_cast<size_t>(std::atoll(argv[++i]));
        else if (!std::strcmp(argv[i],"--prefill-reps")&& i+1<argc) prefill_reps= std::atoi(argv[++i]);
    }

    // CCL's fire-and-forget mode uses background threads that call MPI
    // concurrently → MPI_THREAD_MULTIPLE is required.
    int mpi_provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_provided);
    int rank, ws;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ws);

    // Select one GPU per rank
    auto devs = sycl::platform{sycl::gpu_selector_v}.get_devices();
    if ((int)devs.size() < ws) {
        if (rank == 0)
            std::fprintf(stderr, "bench_ccl_prefill: need %d GPUs visible, found %zu\n",
                         ws, devs.size());
        MPI_Finalize();
        return 1;
    }
    sycl::device  dev = devs[rank];
    sycl::context ctx(dev);

    // SYCL queue with profiling enabled (explicit device + context)
    sycl::queue q{ctx, dev,
                  {sycl::property::queue::in_order{},
                   sycl::property::queue::enable_profiling{}}};

    // oneCCL init
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

    // Shared buffers (max size: numel = 2^max_log2)
    const int64_t max_numel = (int64_t)1 << max_log2;
    bf16* buf_full  = sycl::malloc_device<bf16>(max_numel,          q);
    bf16* buf_chunk = sycl::malloc_device<bf16>(max_numel / ws,     q);
    bf16* buf_out   = sycl::malloc_device<bf16>(max_numel,          q);
    bf16* prefill   = sycl::malloc_device<bf16>(prefill_n,          q);
    q.memset(buf_full,  0, max_numel * sizeof(bf16));
    q.memset(buf_chunk, 0, (max_numel/ws) * sizeof(bf16));
    q.memset(buf_out,   0, max_numel * sizeof(bf16));
    q.memset(prefill,   0, prefill_n * sizeof(bf16));
    q.wait();

    // Gather buffers: [loop per-iter ms + 1 span] per rank
    const int kFields = loop + 1;
    std::vector<double> gbuf(rank == 0 ? ws * kFields : 1, 0.0);
    std::vector<double> my_g(kFields, 0.0);

    // ── Banner ─────────────────────────────────────────────────────────────
    if (rank == 0) {
        const std::string sep(90, '=');
        std::printf("\n%s\n", sep.c_str());
        std::printf("  GEMM-Prefill Fire-and-Forget CCL Benchmark  "
                    "(oneCCL native / SYCL event profiling)\n");
        std::printf("  world_size=%d  dtype=bfloat16  warmup=%d  loop=%d\n",
                    ws, warmup, loop);
        std::printf("  prefill: %zu M bf16 elements × %d reps/elem\n",
                    prefill_n >> 20, prefill_reps);
        std::printf("  sizes: 2^%d .. 2^%d bf16 elements  (%s .. %s)\n",
                    min_log2, max_log2,
                    fmt_size((double)((int64_t)1<<min_log2)*2).c_str(),
                    fmt_size((double)((int64_t)1<<max_log2)*2).c_str());
        std::printf("  ops:");
        for (const auto& od : kOpDefs) if (ops & od.bit) std::printf(" %s", od.name);
        std::printf("\n");
        std::printf("  per_iter_us = SYCL event time per iteration (µs)  "
                    "span_us = (GPU total span) / loop (µs, pipeline throughput)\n");
        std::printf("%s\n", sep.c_str());
        std::fflush(stdout);
    }

    // ── Outer loop: one section per op ─────────────────────────────────────
    for (const auto& od : kOpDefs) {
        if (!(ops & od.bit)) continue;

        // Table rows accumulated for the clean summary at the end
        struct RowData { double bytes; std::string size_str;
                         double global_min; double avg_span; };
        std::vector<RowData> table_rows;

        if (rank == 0) {
            std::printf("\n\n%s\n", std::string(90, '-').c_str());
            std::printf("  Op: %s\n", od.name);
            std::printf("%s\n", std::string(90, '-').c_str());
            std::printf("\n  -- per-rank per-iter event times (µs) --\n");
            std::fflush(stdout);
        }

        // ── Inner loop: sizes ──────────────────────────────────────────────
        for (int lg = min_log2; lg <= max_log2; lg += step) {
            const int64_t numel = (int64_t)1 << lg;
            if (numel % ws != 0) continue;

            const int64_t chunk = numel / ws;
            const double  bytes = static_cast<double>(numel) * 2.0;
            const std::vector<size_t> ag_counts(ws, static_cast<size_t>(chunk));

            BenchResult res;
            if (od.bit == kAR) {
                res = bench_op(q, warmup, loop, MPI_COMM_WORLD,
                               prefill, prefill_n, prefill_reps,
                    [&]() -> ccl::event {
                        return ccl::allreduce(
                            buf_full, buf_out, static_cast<size_t>(numel),
                            ccl::datatype::bfloat16, ccl::reduction::sum,
                            ccl_comm, ccl_stream);
                    });
            } else if (od.bit == kAG) {
                res = bench_op(q, warmup, loop, MPI_COMM_WORLD,
                               prefill, prefill_n, prefill_reps,
                    [&]() -> ccl::event {
                        return ccl::allgatherv(
                            buf_chunk, static_cast<size_t>(chunk),
                            buf_out, ag_counts,
                            ccl::datatype::bfloat16,
                            ccl_comm, ccl_stream);
                    });
            } else {  // kRS
                res = bench_op(q, warmup, loop, MPI_COMM_WORLD,
                               prefill, prefill_n, prefill_reps,
                    [&]() -> ccl::event {
                        return ccl::reduce_scatter(
                            buf_full, buf_chunk, static_cast<size_t>(chunk),
                            ccl::datatype::bfloat16, ccl::reduction::sum,
                            ccl_comm, ccl_stream);
                    });
            }

            // Pack: [per_us[0..loop-1], span_us]
            for (int i = 0; i < loop; ++i) my_g[i] = res.per_us[i];
            my_g[loop] = res.span_us;

            MPI_Gather(my_g.data(), kFields, MPI_DOUBLE,
                       (rank == 0) ? gbuf.data() : nullptr, kFields, MPI_DOUBLE,
                       0, MPI_COMM_WORLD);

            if (rank == 0) {
                std::string sz = fmt_size(bytes);
                std::printf("\n  [%-14s %12s]\n", od.name, sz.c_str());

                double global_min = 1e18;
                double sum_span   = 0.0;
                for (int r = 0; r < ws; ++r) {
                    const double* rv = gbuf.data() + r * kFields;
                    double r_min  = rv[0];
                    double r_span = rv[loop];
                    std::printf("    r%d: [", r);
                    for (int i = 0; i < loop; ++i) {
                        std::printf("%.2f", rv[i]);
                        if (i < loop-1) std::printf("  ");
                        r_min = std::min(r_min, rv[i]);
                    }
                    std::printf("]  min=%.2f  span=%.2f  (µs)\n", r_min, r_span);
                    global_min = std::min(global_min, r_min);
                    sum_span  += r_span;
                }
                table_rows.push_back({bytes, sz, global_min, sum_span / ws});
                std::fflush(stdout);
            }
        }  // sizes

        // ── Summary table ──────────────────────────────────────────────────
        if (rank == 0) {
            std::printf("\n  -- %s  summary table --\n\n", od.name);
            print_table_header();
            for (const auto& row : table_rows)
                print_table_row(od.name, row.bytes,
                                row.global_min, row.avg_span,
                                od.bit, ws);
            std::printf("\n");
            std::fflush(stdout);
        }
    }  // ops

    // ── Cleanup: free device memory before CCL objects go out of scope ──────
    MPI_Barrier(MPI_COMM_WORLD);
    q.wait();
    sycl::free(buf_full,  q);
    sycl::free(buf_chunk, q);
    sycl::free(buf_out,   q);
    sycl::free(prefill,   q);
    q.wait();
    // ccl_stream / ccl_comm / kvs destructors run here (before MPI_Finalize)
    MPI_Finalize();
    return 0;
}
