/*
 * bench_oneshot_vs_ccl.cpp
 *
 * Pure C++/SYCL/oneCCL benchmark (no torch) for:
 *   (a) our one_shot all-reduce SUM kernel — fused (in-kernel signal barrier)
 *   (b) our one_shot all-reduce SUM kernel — non-fused (reduce only, host-side barrier)
 *   (c) oneCCL allreduce (reference)
 *
 * All three variants run through the same MPI-launched multi-process pipeline
 * used by sycl-tla's standalone allreduce demo:
 *   mpirun -n WS ./bench_oneshot_vs_ccl --min 10 --max 22
 *
 * Measurement: wall-clock per iteration, time from q.submit() to q.wait(),
 * with an MPI_Barrier before each timed region so all ranks start together.
 *
 *   icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 \
 *        -I/opt/intel/oneapi/2025.3/include \
 *        -I/root/hanchao/sycl-tla/examples/00_bmg_gemm \
 *        -L/opt/intel/oneapi/2025.3/lib -lccl \
 *        -I$I_MPI_ROOT/include -L$I_MPI_ROOT/lib -lmpi \
 *        -L/opt/intel/oneapi/2025.3/lib -lze_loader \
 *        bench_oneshot_vs_ccl.cpp -o bench_oneshot_vs_ccl
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>
#include <vector>

#include <level_zero/ze_api.h>
#include <oneapi/ccl.hpp>

// symm.hpp (from sycl-tla) provides exchange_ipc_ptrs / close_ipc_ptrs
// free functions (and an unused SymmMemory class we do not instantiate).
#include "symm.hpp"

using bf16 = sycl::ext::oneapi::bfloat16;

// ---------- constants (match torch-xpu-ops XPUSymmetricMemoryOps.cpp) ----------
constexpr int kOneShotMaxNumGroups = 24;
constexpr int kOneShotMaxNumThreads = 512;
constexpr int kFusedSignalBaseU32 = 512;
constexpr int kSignalPadU32Slots = 4096; // big enough for pre+post
constexpr int kVecBytes = 16;

template <int N>
struct alignas(kVecBytes) VecBf16 {
  bf16 data[N];
};

// signal helpers: store_release/load_acquire come from symm.hpp.
inline void put_signal_dev(uint32_t* addr) {
  while (load_acquire(addr) != 0)
    ;
  store_release(addr, 1);
}
inline void wait_signal_dev(uint32_t* addr) {
  while (load_acquire(addr) != 1)
    ;
  store_release(addr, 0);
}

// ---------- reduce kernels (bf16, ws=4 only — matches §12.6 bench) ----------
constexpr int kWorldSize = 4;
constexpr int kN = kVecBytes / (int)sizeof(bf16); // 8

struct OneShotKernel {
  bf16** peer_ptrs;     // device USM, 4 entries
  bf16* output_ptr;     // local
  int64_t numel;
  int my_rank;
  void operator()(sycl::nd_item<1> it) const {
    const int64_t tid = (int64_t)it.get_global_linear_id();
    const int64_t stride = (int64_t)it.get_global_range(0);
    const int64_t vec_total = numel / kN;
    using V = VecBf16<kN>;
    for (int64_t v = tid; v < vec_total; v += stride) {
      const int64_t e = v * kN;
      V acc = *reinterpret_cast<const V*>(peer_ptrs[my_rank] + e);
#pragma unroll
      for (int step = 1; step < kWorldSize; ++step) {
        const int p = (my_rank + step) % kWorldSize;
        V rhs = *reinterpret_cast<const V*>(peer_ptrs[p] + e);
#pragma unroll
        for (int i = 0; i < kN; ++i) {
          acc.data[i] = (bf16)((float)acc.data[i] + (float)rhs.data[i]);
        }
      }
      *reinterpret_cast<V*>(output_ptr + e) = acc;
    }
  }
};

// Standalone signal-pad barrier kernel, matches torch-xpu-ops
// Signal.cpp::barrierKernel. Slot layout per rank: signal_pads[r] + ws*channel + src.
// channel=0 is reserved for this standalone barrier; fused kernel uses
// kFusedSignalBaseU32 offset, so they never collide.
struct BarrierKernel {
  uint32_t** signal_pads;
  int channel;
  int my_rank;
  void operator()(sycl::nd_item<1> it) const {
    const auto tid = it.get_local_id(0);
    if (tid < (size_t)kWorldSize) {
      int target = (int)tid;
      if (target == my_rank) return;
      put_signal_dev(signal_pads[target] + kWorldSize * channel + my_rank);
      wait_signal_dev(signal_pads[my_rank] + kWorldSize * channel + target);
    }
  }
};

struct FusedOneShotKernel {
  bf16** peer_ptrs;
  bf16* output_ptr;
  uint32_t** signal_pads; // device USM, 4 entries
  int64_t numel;
  int my_rank;

  static inline uint32_t* slot_of(
      uint32_t** pads, int owner, int region, int group_id, int src_rank) {
    const int64_t region_off = (int64_t)region * kOneShotMaxNumGroups * kWorldSize;
    return pads[owner] + kFusedSignalBaseU32 + region_off +
        (int64_t)group_id * kWorldSize + src_rank;
  }
  inline void wg_barrier(sycl::nd_item<1> it, int region) const {
    const auto lid = it.get_local_id(0);
    const auto gid = it.get_group(0);
    if (lid < (size_t)kWorldSize) {
      int peer = (int)lid;
      if (peer != my_rank) {
        uint32_t* put = slot_of(signal_pads, peer, region, gid, my_rank);
        uint32_t* wait = slot_of(signal_pads, my_rank, region, gid, peer);
        put_signal_dev(put);
        wait_signal_dev(wait);
      }
    }
    it.barrier(sycl::access::fence_space::local_space);
  }
  void operator()(sycl::nd_item<1> it) const {
    wg_barrier(it, /*region=*/0);

    const int64_t tid = (int64_t)it.get_global_linear_id();
    const int64_t stride = (int64_t)it.get_global_range(0);
    const int64_t vec_total = numel / kN;
    using V = VecBf16<kN>;
    for (int64_t v = tid; v < vec_total; v += stride) {
      const int64_t e = v * kN;
      V acc = *reinterpret_cast<const V*>(peer_ptrs[my_rank] + e);
#pragma unroll
      for (int step = 1; step < kWorldSize; ++step) {
        const int p = (my_rank + step) % kWorldSize;
        V rhs = *reinterpret_cast<const V*>(peer_ptrs[p] + e);
#pragma unroll
        for (int i = 0; i < kN; ++i) {
          acc.data[i] = (bf16)((float)acc.data[i] + (float)rhs.data[i]);
        }
      }
      *reinterpret_cast<V*>(output_ptr + e) = acc;
    }

    // post barrier: ensure all WG threads finished reads before unblocking peers.
    it.barrier(sycl::access::fence_space::local_space);
    wg_barrier(it, /*region=*/1);
  }
};

static inline void init_launch_cfg(
    int64_t numel, int64_t& groups, int64_t& threads) {
  int64_t total_vec = (numel + kN - 1) / kN;
  if (total_vec <= kOneShotMaxNumThreads) {
    groups = 1;
    threads = std::max<int64_t>(32, (total_vec + 31) / 32 * 32);
  } else {
    groups = std::min<int64_t>(
        (total_vec + kOneShotMaxNumThreads - 1) / kOneShotMaxNumThreads,
        (int64_t)kOneShotMaxNumGroups);
    threads = kOneShotMaxNumThreads;
  }
}

// ---------- main ----------
int main(int argc, char** argv) {
  int min_log2 = 10, max_log2 = 22, warmup = 10, iters = 50;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nxt = [&](int d) { return (i + 1 < argc) ? std::atoi(argv[++i]) : d; };
    if (a == "--min") min_log2 = nxt(min_log2);
    else if (a == "--max") max_log2 = nxt(max_log2);
    else if (a == "--warmup") warmup = nxt(warmup);
    else if (a == "--iters") iters = nxt(iters);
  }

  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  int rank = 0, ws = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ws);
  if (ws != kWorldSize) {
    if (rank == 0)
      std::fprintf(stderr, "bench requires world_size=%d (got %d)\n", kWorldSize, ws);
    MPI_Finalize();
    return 1;
  }

  // Select device: one per rank, constrained by ZE_AFFINITY_MASK.
  auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
  if ((int)devs.size() < ws) {
    if (rank == 0)
      std::fprintf(stderr, "need %d GPU devices visible, found %zu\n", ws, devs.size());
    MPI_Finalize();
    return 1;
  }
  sycl::device dev = devs[rank];
  sycl::context ctx(dev);
  sycl::queue q(ctx, dev,
      sycl::property_list{sycl::property::queue::in_order{}});

  // Allocate max-size buffers once.
  const int64_t max_numel = (int64_t)1 << max_log2;
  bf16* in = sycl::malloc_device<bf16>(max_numel, q);
  bf16* out = sycl::malloc_device<bf16>(max_numel, q);
  uint32_t* sig = sycl::malloc_device<uint32_t>(kSignalPadU32Slots, q);
  q.memset(sig, 0, kSignalPadU32Slots * sizeof(uint32_t)).wait();

  // Fill input with rank+1 so allreduce sum = 1+2+3+4 = 10.
  q.submit([&](sycl::handler& h) {
    bf16 val = (bf16)(float)(rank + 1);
    bf16* p = in;
    h.parallel_for(sycl::range<1>(max_numel), [=](sycl::id<1> i) { p[i] = val; });
  }).wait();
  MPI_Barrier(MPI_COMM_WORLD);

  // L0 IPC exchange for in buffer and signal pad.
  std::vector<void*> opened_in, opened_sig;
  auto peer_in = exchange_ipc_ptrs(in, rank, ws, q, opened_in);
  auto peer_sig = exchange_ipc_ptrs(sig, rank, ws, q, opened_sig);

  // Make peer memory resident on local device.
  auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
  auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
  for (int r = 0; r < ws; ++r)
    if (r != rank) {
      zeContextMakeMemoryResident(ze_ctx, ze_dev, peer_in[r], max_numel * sizeof(bf16));
      zeContextMakeMemoryResident(ze_ctx, ze_dev, peer_sig[r], kSignalPadU32Slots * sizeof(uint32_t));
    }

  // Upload peer pointer tables to device.
  bf16** d_peer_in = sycl::malloc_device<bf16*>(ws, q);
  uint32_t** d_peer_sig = sycl::malloc_device<uint32_t*>(ws, q);
  {
    std::vector<bf16*> h_in(ws);
    std::vector<uint32_t*> h_sig(ws);
    for (int r = 0; r < ws; ++r) {
      h_in[r] = reinterpret_cast<bf16*>(peer_in[r]);
      h_sig[r] = reinterpret_cast<uint32_t*>(peer_sig[r]);
    }
    q.memcpy(d_peer_in, h_in.data(), ws * sizeof(bf16*));
    q.memcpy(d_peer_sig, h_sig.data(), ws * sizeof(uint32_t*));
    q.wait();
  }

  // ---------- oneCCL setup ----------
  ccl::init();
  ccl::shared_ptr_class<ccl::kvs> kvs;
  ccl::kvs::address_type kvs_addr{};
  if (rank == 0) {
    kvs = ccl::create_main_kvs();
    kvs_addr = kvs->get_address();
  }
  MPI_Bcast(kvs_addr.data(), kvs_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
  if (rank != 0) kvs = ccl::create_kvs(kvs_addr);
  auto ccl_dev = ccl::create_device(dev);
  auto ccl_ctx = ccl::create_context(ctx);
  auto ccl_comm = ccl::create_communicator(ws, rank, ccl_dev, ccl_ctx, kvs);
  auto ccl_stream = ccl::create_stream(q);

  // ---------- bench loop ----------
  using clk = std::chrono::high_resolution_clock;
  auto us = [](clk::duration d) {
    return std::chrono::duration<double, std::micro>(d).count();
  };

  if (rank == 0) {
    std::printf(
        "%-8s %-10s %12s %12s %12s %12s %12s %12s\n",
        "bytes", "numel", "fused_us", "nofuse_us", "ccl_us",
        "fused_BW", "nofuse_BW", "ccl_BW");
  }

  for (int lg = min_log2; lg <= max_log2; lg += 2) {
    int64_t numel = (int64_t)1 << lg;
    if (numel % kN != 0) continue;

    int64_t groups = 0, threads = 0;
    init_launch_cfg(numel, groups, threads);

    auto run_fused = [&]() {
      FusedOneShotKernel ker{d_peer_in, out, d_peer_sig, numel, rank};
      q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(groups * threads, threads), ker);
      });
    };
    auto run_nofuse = [&]() {
      // Mirror torch USE_SIGNAL_BARRIER=1 one_shot path:
      //   barrier(ch=0)  ->  reduce kernel  ->  barrier(ch=1)
      // All three are in-order ops on the same SYCL queue, no MPI involvement.
      BarrierKernel b0{d_peer_sig, /*channel=*/0, rank};
      BarrierKernel b1{d_peer_sig, /*channel=*/1, rank};
      OneShotKernel ker{d_peer_in, out, numel, rank};
      q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(std::max(32, kWorldSize),
                                         std::max(32, kWorldSize)), b0);
      });
      q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(groups * threads, threads), ker);
      });
      q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(std::max(32, kWorldSize),
                                         std::max(32, kWorldSize)), b1);
      });
    };
    auto run_ccl = [&]() {
      ccl::allreduce(
          in, out, (size_t)numel, ccl::datatype::bfloat16,
          ccl::reduction::sum, ccl_comm, ccl_stream).wait();
    };

    // Warmup
    for (int i = 0; i < warmup; ++i) { run_fused(); q.wait(); }
    MPI_Barrier(MPI_COMM_WORLD);
    // Fused
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) { run_fused(); q.wait(); }
    auto fused_us = us(clk::now() - t0) / iters;

    for (int i = 0; i < warmup; ++i) { run_nofuse(); q.wait(); }
    MPI_Barrier(MPI_COMM_WORLD);
    // Non-fused: signal-pad barrier kernel on stream (same as torch
    // USE_SIGNAL_BARRIER=1). MPI is only the launcher, not used per-iter.
    t0 = clk::now();
    for (int i = 0; i < iters; ++i) {
      run_nofuse();
      q.wait();
    }
    auto nofuse_us = us(clk::now() - t0) / iters;

    for (int i = 0; i < warmup; ++i) run_ccl();
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = clk::now();
    for (int i = 0; i < iters; ++i) run_ccl();
    auto ccl_us = us(clk::now() - t0) / iters;

    if (rank == 0) {
      double bytes = (double)numel * 2.0;
      auto bw = [&](double t_us) { return bytes / (t_us * 1e-6) / (1ULL << 30); };
      std::printf(
          "%-8lld %-10lld %12.2f %12.2f %12.2f %12.3f %12.3f %12.3f\n",
          (long long)(numel * 2), (long long)numel,
          fused_us, nofuse_us, ccl_us,
          bw(fused_us), bw(nofuse_us), bw(ccl_us));
      std::fflush(stdout);
    }
  }

  // Cleanup
  close_ipc_ptrs(q, opened_in);
  close_ipc_ptrs(q, opened_sig);
  sycl::free(in, q);
  sycl::free(out, q);
  sycl::free(sig, q);
  sycl::free(d_peer_in, q);
  sycl::free(d_peer_sig, q);

  MPI_Finalize();
  return 0;
}
