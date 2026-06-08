/*
 * bench_ccl_collectives.cpp
 *
 * Pure C++/SYCL/oneCCL latency benchmark for small messages (≤ 2 MiB):
 *   - allreduce     (SUM, bf16) — each rank contributes/receives `numel` elements
 *   - allgather     (bf16)      — each rank sends `numel/ws`, receives `numel` total
 *   - reducescatter (SUM, bf16) — each rank sends `numel` total, receives `numel/ws`
 *
 * The parameter `numel` sweeps powers of 2 from --min to --max (log2 of
 * bf16 element count). Default --max 20 → numel=1 048 576 bf16 = 2 MiB.
 *
 * Bandwidth column = (numel * sizeof(bf16)) / latency  (simple goodput;
 * all three collectives use the same denominator for easy comparison).
 *
 * Build (see build_and_run_ccl.sh):
 *   icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -std=c++17 \
 *        -I${CCL_ROOT}/include -I${I_MPI_ROOT}/include                       \
 *        -L${CCL_ROOT}/lib -L${I_MPI_ROOT}/lib                               \
 *        -lccl -lmpi                                                          \
 *        bench_ccl_collectives.cpp -o bench_ccl_collectives
 *
 * Run (4 ranks, cards 0-3):
 *   ZE_AFFINITY_MASK=0,1,2,3 CCL_ATL_TRANSPORT=mpi \
 *   mpirun -n 4 ./bench_ccl_collectives --min 7 --max 20 --warmup 20 --iters 100
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <string>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>
#include <vector>
#include <oneapi/ccl.hpp>

using bf16 = sycl::ext::oneapi::bfloat16;
constexpr double kBf16Bytes = 2.0;

int main(int argc, char** argv) {
    // ---- parse CLI ----
    int min_log2 = 7, max_log2 = 20, warmup = 20, iters = 100, step = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&](int d) { return (i + 1 < argc) ? std::atoi(argv[++i]) : d; };
        if      (a == "--min")    min_log2 = nxt(min_log2);
        else if (a == "--max")    max_log2 = nxt(max_log2);
        else if (a == "--warmup") warmup   = nxt(warmup);
        else if (a == "--iters")  iters    = nxt(iters);
        else if (a == "--step")   step     = nxt(step);
    }

    // ---- MPI init ----
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int rank = 0, ws = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ws);

    // ---- SYCL device: one device per rank ----
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if ((int)devs.size() < ws) {
        if (rank == 0)
            std::fprintf(stderr,
                "bench_ccl_collectives: need %d GPU devices, found %zu\n",
                ws, devs.size());
        MPI_Finalize();
        return 1;
    }
    sycl::device  dev = devs[rank];
    sycl::context ctx(dev);
    sycl::queue   q(ctx, dev, sycl::property_list{sycl::property::queue::in_order{}});

    // ---- allocate max-size buffers once ----
    //
    //   buf_full : max_numel elements
    //              → allreduce src/dst, allgather full recv, reducescatter full send
    //   buf_out  : max_numel elements
    //              → allreduce dst (separate from src), allgather recv (can alias buf_full)
    //   buf_chunk: max_numel elements (overallocated; effective size = max_numel/ws)
    //              → allgather per-rank send chunk, reducescatter per-rank recv chunk
    //
    const int64_t max_numel = (int64_t)1 << max_log2;

    bf16* buf_full  = sycl::malloc_device<bf16>(max_numel, q);
    bf16* buf_out   = sycl::malloc_device<bf16>(max_numel, q);
    bf16* buf_chunk = sycl::malloc_device<bf16>(max_numel, q); // size >= max_numel/ws

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

    // ---- timing helpers ----
    using clk = std::chrono::high_resolution_clock;
    auto us_of = [](clk::duration d) {
        return std::chrono::duration<double, std::micro>(d).count();
    };
    // Simple goodput: bytes / time  (all three ops use same denominator)
    auto gib_s = [](double bytes, double t_us) -> double {
        return bytes / (t_us * 1e-6) / static_cast<double>(1ULL << 30);
    };

    if (rank == 0) {
        // Column key:
        //   bytes  = numel * 2  (bf16 per-op data size)
        //   ar_*   = allreduce   (each rank: numel in, numel out)
        //   ag_*   = allgatherv  (each rank: numel/ws send, numel recv)
        //   rs_*   = reduce_scatter (each rank: numel send, numel/ws recv)
        //   BW     = bytes / latency  (GiB/s goodput, same denominator for all)
        std::printf("%-12s %-10s  %12s %10s  %12s %10s  %12s %10s\n",
            "bytes", "numel(bf16)",
            "ar_us",  "ar_GiB/s",
            "ag_us",  "ag_GiB/s",
            "rs_us",  "rs_GiB/s");
        std::printf("%-12s %-10s  %12s %10s  %12s %10s  %12s %10s\n",
            "------------", "----------",
            "------------", "----------",
            "------------", "----------",
            "------------", "----------");
        std::fflush(stdout);
    }

    for (int lg = min_log2; lg <= max_log2; lg += step) {
        const int64_t numel = (int64_t)1 << lg;

        // allgather/reducescatter require numel divisible by ws
        if (numel % ws != 0) continue;

        const int64_t chunk = numel / ws;
        const std::vector<size_t> ag_recv_counts(ws, static_cast<size_t>(chunk));

        // ------------------------------------------------------------------
        // allreduce: buf_full[0..numel) → buf_out[0..numel)
        // ------------------------------------------------------------------
        auto run_ar = [&]() {
            ccl::allreduce(
                buf_full, buf_out, static_cast<size_t>(numel),
                ccl::datatype::bfloat16, ccl::reduction::sum,
                ccl_comm, ccl_stream).wait();
        };
        for (int i = 0; i < warmup; ++i) run_ar();
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = clk::now();
        for (int i = 0; i < iters; ++i) run_ar();
        double ar_us = us_of(clk::now() - t0) / iters;

        // ------------------------------------------------------------------
        // allgatherv: buf_chunk[0..chunk) → buf_out[0..numel)
        //   each rank sends `chunk` elements; recv_counts[r] = chunk for all r
        // ------------------------------------------------------------------
        auto run_ag = [&]() {
            ccl::allgatherv(
                buf_chunk, static_cast<size_t>(chunk),
                buf_out, ag_recv_counts,
                ccl::datatype::bfloat16,
                ccl_comm, ccl_stream).wait();
        };
        for (int i = 0; i < warmup; ++i) run_ag();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = clk::now();
        for (int i = 0; i < iters; ++i) run_ag();
        double ag_us = us_of(clk::now() - t0) / iters;

        // ------------------------------------------------------------------
        // reduce_scatter: buf_full[0..numel) → buf_chunk[0..chunk)
        //   total send = numel (all ranks contribute); each rank receives chunk
        // ------------------------------------------------------------------
        auto run_rs = [&]() {
            ccl::reduce_scatter(
                buf_full, buf_chunk, static_cast<size_t>(chunk),
                ccl::datatype::bfloat16, ccl::reduction::sum,
                ccl_comm, ccl_stream).wait();
        };
        for (int i = 0; i < warmup; ++i) run_rs();
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = clk::now();
        for (int i = 0; i < iters; ++i) run_rs();
        double rs_us = us_of(clk::now() - t0) / iters;

        // ------------------------------------------------------------------
        // Print: only rank 0 (all ranks have identical timing since
        //        CCL blocks until all ranks complete)
        // ------------------------------------------------------------------
        if (rank == 0) {
            double bytes = static_cast<double>(numel) * kBf16Bytes;
            std::printf(
                "%-12lld %-10lld  %12.2f %10.3f  %12.2f %10.3f  %12.2f %10.3f\n",
                static_cast<long long>(numel) * 2LL,
                static_cast<long long>(numel),
                ar_us, gib_s(bytes, ar_us),
                ag_us, gib_s(bytes, ag_us),
                rs_us, gib_s(bytes, rs_us));
            std::fflush(stdout);
        }
    }

    // ---- cleanup ----
    sycl::free(buf_full,  q);
    sycl::free(buf_out,   q);
    sycl::free(buf_chunk, q);
    MPI_Finalize();
    return 0;
}
