// Standalone PERFORMANCE reproducer for the cross-rank peer-write store in
// RingReduceScatterUnpermute.cpp.
//
// It isolates the ONE store that the shipped fix changed: the 16-byte write that
// pushes a reduced partial into the RIGHT peer's acc buffer (which the peer then
// remote-reads on the next ring step). It runs the exact reduce-scatter ring
// dataflow across N XPU devices (single process, one Level-Zero context, P2P USM)
// and, for each store cache-control variant, reports effective cross-GPU store
// bandwidth (GB/s) + per-iter latency, plus a correctness check.
//
// SCOPE / WHAT THIS DOES AND DOES NOT SHOW:
//   * PERFORMANCE (primary): faithfully reproduces the store-variant tradeoff.
//     On fabric-bound sizes (e.g. --tokens 2048) the LSC write-back store
//     (WB_L3WB, the buggy default) and streaming store (S_L3WB) are the fastest;
//     the plain coherent store (PLAIN, the shipped fix) is ~13-26% slower. This
//     lets you pick the fastest store that is ALSO coherent on the target fabric
//     (candidates that keep L3 write-combining while bypassing the write-back L1:
//     UC_L3WB / WT_L3WB / S_L3WB).
//   * CORRECTNESS: this single-process harness uses one Level-Zero context with
//     P2P USM, which is MORE strongly coherent than the real deployment's
//     multi-process Level-Zero IPC mapping (the perf2 test runs one process per
//     rank). As a result WB does NOT go stale here. The actual staleness
//     regression is reproduced/guarded by the multi-process test
//     test_symm_buffer_unpermute_capacity_regression_dist.py. Use THIS tool to
//     rank the variants by bandwidth; use that UT to confirm the chosen variant
//     is coherent under the real IPC mapping.
//
// Build:
//   icpx -fsycl -std=c++17 -O2 RingReduceScatterUnpermute_store_repro.cpp \
//        -o ring_rsu_store_repro
// Run:
//   ./ring_rsu_store_repro --world 8 --tokens 2048 --iters 200   # fabric-bound
//   ./ring_rsu_store_repro --world 4 --tokens 64 --max-tokens 2048
//
// Store variants map to Intel LSC store cache-control (STCC) encodings:
//   PLAIN     : plain coherent store  (`*dst = v`)      <- the shipped fix
//   UC_UC     : LSC L1UC_L3UC   (1)   uncached
//   UC_L3WB   : LSC L1UC_L3WB   (2)   L1 bypass, L3 write-back (keeps L3 combine)
//   WT_L3UC   : LSC L1WT_L3UC   (3)
//   WT_L3WB   : LSC L1WT_L3WB   (4)   write-through L1, L3 write-back
//   S_L3WB    : LSC L1S_L3WB    (6)   streaming L1, L3 write-back
//   WB_L3WB   : LSC L1WB_L3WB   (7)   write-back (the buggy default) <- fastest

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// LSC store builtin (device only). vec4 of uint32 == 16 bytes, same as kernel.
// ---------------------------------------------------------------------------
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
typedef uint32_t repro_u4 __attribute__((ext_vector_type(4)));
enum ReproStcc {
  STCC_UC_UC = 1,
  STCC_UC_L3WB = 2,
  STCC_WT_L3UC = 3,
  STCC_WT_L3WB = 4,
  STCC_S_L3WB = 6,
  STCC_WB_L3WB = 7,
};
SYCL_EXTERNAL extern "C" void __builtin_IB_lsc_store_global_uint4(
    __attribute__((opencl_global)) repro_u4* base,
    int off,
    repro_u4 val,
    enum ReproStcc cc);
#endif

// Store-variant selector (compile-time so the LSC immediate is a constant).
enum Variant {
  V_PLAIN = 0,
  V_UC_UC = 1,
  V_UC_L3WB = 2,
  V_WT_L3UC = 3,
  V_WT_L3WB = 4,
  V_S_L3WB = 6,
  V_WB_L3WB = 7,
};

template <int V>
inline void store16(uint32_t* dst, const uint32_t* src) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
  repro_u4 v = *reinterpret_cast<const repro_u4*>(src);
  if constexpr (V == V_PLAIN) {
    *reinterpret_cast<repro_u4*>(dst) = v;
  } else {
    __builtin_IB_lsc_store_global_uint4(
        (__attribute__((opencl_global)) repro_u4*)(dst),
        0,
        v,
        static_cast<ReproStcc>(V));
  }
#else
  for (int i = 0; i < 4; ++i) dst[i] = src[i];
#endif
}

inline void store_release_sys(uint32_t* addr, uint32_t val) {
  *addr = val;
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
}

inline void wait_eq_sys(volatile uint32_t* addr, uint32_t val) {
  for (;;) {
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    if (*const_cast<uint32_t*>(addr) == val) break;
  }
}

// Deterministic per-rank contribution for block `b`, element `i`. Chosen so the
// exact reduced result is trivially computable and any stale read corrupts it.
inline uint32_t contrib(int rank, int64_t /*b*/, int64_t i) {
  return static_cast<uint32_t>((rank + 1) * 1000003u + static_cast<uint32_t>(i));
}

// ---------------------------------------------------------------------------
// Ring reduce-scatter kernel, mirroring RingReduceScatterUnpermuteSingleKernel:
// each rank folds its contribution into the incoming partial and pushes the
// result into the RIGHT peer's acc using the store variant under test.
// acc layout: [world_size blocks][live_tokens][hidden], compact by LIVE tokens
// (block stride = live_tokens*hidden), while the buffer is ALLOCATED for
// max_tokens (capacity) -- reproducing the below-capacity trigger.
// ---------------------------------------------------------------------------
template <int V>
struct RingRsuStoreKernel {
  const int64_t* rank_accs;   // uint32_t* per rank, cast to int64
  const int64_t* rank_flags;  // uint32_t* per rank, cast to int64
  uint32_t* output;           // [live_tokens*hidden] local final block
  int64_t hidden;
  int64_t live_tokens;
  int64_t tokens_per_wg;
  int32_t rank;
  int32_t world_size;
  int32_t right;
  int32_t num_wg;
  uint32_t tag;

  inline void fold_block(
      int32_t block,
      const uint32_t* acc_base,  // nullptr for phase 0 (seed)
      uint32_t* dst_base,
      int64_t token_base,
      int64_t token_cnt,
      int32_t lid,
      int32_t lsize) const {
    const int64_t hidden_vecs = hidden / 4;
    for (int64_t lt = 0; lt < token_cnt; ++lt) {
      const int64_t local_t = token_base + lt;
      const int64_t row_off =
          (static_cast<int64_t>(block) * live_tokens + local_t) * hidden;
      const uint32_t* acc_row = (acc_base != nullptr) ? acc_base + row_off : nullptr;
      uint32_t* dst_row = dst_base + row_off;
      for (int64_t vh = lid; vh < hidden_vecs; vh += lsize) {
        const int64_t h = vh * 4;
        uint32_t v[4];
        for (int i = 0; i < 4; ++i) {
          uint32_t a = (acc_row != nullptr) ? acc_row[h + i] : 0u;
          v[i] = a + contrib(rank, block, h + i);
        }
        store16<V>(dst_row + h, v);
      }
    }
  }

  inline void fold_final(
      int32_t block,
      const uint32_t* acc_base,
      int64_t token_base,
      int64_t token_cnt,
      int32_t lid,
      int32_t lsize) const {
    const int64_t hidden_vecs = hidden / 4;
    for (int64_t lt = 0; lt < token_cnt; ++lt) {
      const int64_t local_t = token_base + lt;
      const int64_t acc_off =
          (static_cast<int64_t>(block) * live_tokens + local_t) * hidden;
      uint32_t* dst_row = output + local_t * hidden;  // compact local output
      for (int64_t vh = lid; vh < hidden_vecs; vh += lsize) {
        const int64_t h = vh * 4;
        for (int i = 0; i < 4; ++i)
          dst_row[h + i] = acc_base[acc_off + h + i] + contrib(rank, block, h + i);
      }
    }
  }

  void operator()(sycl::nd_item<1> item) const {
    const int32_t wg = static_cast<int32_t>(item.get_group(0));
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lsize = static_cast<int32_t>(item.get_local_range(0));

    const int64_t token_base = static_cast<int64_t>(wg) * tokens_per_wg;
    int64_t token_cnt = live_tokens - token_base;
    if (token_cnt > tokens_per_wg) token_cnt = tokens_per_wg;
    if (token_cnt < 0) token_cnt = 0;

    uint32_t* my_pad = reinterpret_cast<uint32_t*>(rank_flags[rank]);
    uint32_t* right_pad = reinterpret_cast<uint32_t*>(rank_flags[right]);
    uint32_t* my_acc = reinterpret_cast<uint32_t*>(rank_accs[rank]);
    uint32_t* right_acc = reinterpret_cast<uint32_t*>(rank_accs[right]);

    // Phase 0: seed block b0 into the right peer's acc.
    {
      const int32_t b0 = (rank - 1 + world_size) % world_size;
      fold_block(b0, nullptr, right_acc, token_base, token_cnt, lid, lsize);
      sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
      item.barrier(sycl::access::fence_space::local_space);
      if (lid == 0) store_release_sys(right_pad + (0 * num_wg + wg), tag);
    }

    // Steps 1..ws-1: fold our contribution into the incoming partial.
    for (int32_t t = 1; t < world_size; ++t) {
      if (lid == 0) wait_eq_sys(my_pad + ((t - 1) * num_wg + wg), tag);
      item.barrier(sycl::access::fence_space::local_space);
      sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);

      const int32_t b_t = (rank - 1 - t + 2 * world_size) % world_size;
      if (t < world_size - 1) {
        fold_block(b_t, my_acc, right_acc, token_base, token_cnt, lid, lsize);
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        item.barrier(sycl::access::fence_space::local_space);
        if (lid == 0) store_release_sys(right_pad + (t * num_wg + wg), tag);
      } else {
        fold_final(b_t, my_acc, token_base, token_cnt, lid, lsize);
      }
    }
  }
};

static void compute_launch(
    int64_t live_tokens, int64_t hidden, int64_t threads, int vec,
    int32_t& num_wg, int64_t& tokens_per_wg) {
  const int64_t chunk = live_tokens * hidden;
  const int64_t per_wg = threads * vec;
  int64_t nwg = (chunk + per_wg - 1) / per_wg;
  if (nwg < 1) nwg = 1;
  if (nwg > 64) nwg = 64;  // RING_MAX_WG in the real kernel
  int64_t tpw = (live_tokens + nwg - 1) / nwg;
  if (tpw < 1) tpw = 1;
  nwg = (live_tokens + tpw - 1) / tpw;
  if (nwg < 1) nwg = 1;
  num_wg = static_cast<int32_t>(nwg);
  tokens_per_wg = tpw;
}

struct RankCtx {
  sycl::queue q;
  uint32_t* acc;     // capacity: max_tokens*hidden*world
  uint32_t* flag;    // world*num_wg
  uint32_t* output;  // live_tokens*hidden
  int64_t* accs_tbl; // device copy of all ranks' acc ptrs
  int64_t* flags_tbl;
};

template <int V>
static double launch_all(std::vector<RankCtx>& ranks, int world, int64_t hidden,
                         int64_t live, int64_t tpw, int num_wg, uint32_t tag,
                         int threads) {
  for (int r = 0; r < world; ++r) {
    auto& rc = ranks[r];
    RingRsuStoreKernel<V> k{rc.accs_tbl, rc.flags_tbl, rc.output, hidden, live,
                            tpw, r, world, (r + 1) % world, num_wg, tag};
    rc.q.submit([&](sycl::handler& h) {
      h.parallel_for(
          sycl::nd_range<1>(sycl::range<1>((size_t)num_wg * threads),
                            sycl::range<1>(threads)),
          k);
    });
  }
  for (int r = 0; r < world; ++r) ranks[r].q.wait();
  return 0.0;
}

using LaunchFn = void (*)(std::vector<RankCtx>&, int, int64_t, int64_t, int64_t,
                          int, uint32_t, int);

template <int V>
static void launch_dispatch(std::vector<RankCtx>& ranks, int world, int64_t hidden,
                            int64_t live, int64_t tpw, int num_wg, uint32_t tag,
                            int threads) {
  launch_all<V>(ranks, world, hidden, live, tpw, num_wg, tag, threads);
}

struct VariantEntry { const char* name; LaunchFn fn; };

int main(int argc, char** argv) {
  int world = -1;
  int64_t hidden = 2048;
  int64_t live = 64;       // live tokens per rank
  int64_t max_tokens = 2048;  // buffer capacity per rank
  int iters = 200;
  int warmup = 50;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::stoll(argv[++i]); };
    if (a == "--world") world = (int)next();
    else if (a == "--hidden") hidden = next();
    else if (a == "--tokens") live = next();
    else if (a == "--max-tokens") max_tokens = next();
    else if (a == "--iters") iters = (int)next();
    else if (a == "--warmup") warmup = (int)next();
    else { printf("unknown arg: %s\n", a.c_str()); return 1; }
  }
  if (hidden % 4 != 0) { printf("hidden must be %% 4\n"); return 1; }
  if (live > max_tokens) { printf("--tokens must be <= --max-tokens\n"); return 1; }

  std::vector<sycl::device> devs;
  for (auto& p : sycl::platform::get_platforms())
    if (p.get_backend() == sycl::backend::ext_oneapi_level_zero)
      for (auto& d : p.get_devices())
        if (d.is_gpu()) devs.push_back(d);
  if (world <= 0) world = (int)devs.size();
  if (world < 2 || (int)devs.size() < world) {
    printf("need >= %d L0 GPUs (have %zu)\n", world < 2 ? 2 : world, devs.size());
    return 1;
  }
  devs.resize(world);
  sycl::context ctx(devs);

  const int threads = 256;
  int32_t num_wg = 1; int64_t tpw = live;
  compute_launch(live, hidden, threads, 4, num_wg, tpw);

  const int64_t cap_elems = max_tokens * hidden * world;   // acc capacity
  const int64_t flag_elems = (int64_t)world * num_wg;
  const int64_t out_elems = live * hidden;

  std::vector<RankCtx> ranks(world);
  std::vector<int64_t> acc_ptrs(world), flag_ptrs(world);
  for (int r = 0; r < world; ++r) {
    ranks[r].q = sycl::queue(ctx, devs[r]);
    ranks[r].acc = sycl::malloc_device<uint32_t>(cap_elems, devs[r], ctx);
    ranks[r].flag = sycl::malloc_device<uint32_t>(flag_elems, devs[r], ctx);
    ranks[r].output = sycl::malloc_device<uint32_t>(out_elems, devs[r], ctx);
    acc_ptrs[r] = reinterpret_cast<int64_t>(ranks[r].acc);
    flag_ptrs[r] = reinterpret_cast<int64_t>(ranks[r].flag);
  }
  for (int r = 0; r < world; ++r) {
    ranks[r].q.memset(ranks[r].acc, 0, cap_elems * sizeof(uint32_t));
    ranks[r].q.memset(ranks[r].flag, 0, flag_elems * sizeof(uint32_t));
    ranks[r].accs_tbl = sycl::malloc_device<int64_t>(world, devs[r], ctx);
    ranks[r].flags_tbl = sycl::malloc_device<int64_t>(world, devs[r], ctx);
  }
  for (int r = 0; r < world; ++r) {
    ranks[r].q.memcpy(ranks[r].accs_tbl, acc_ptrs.data(), world * sizeof(int64_t));
    ranks[r].q.memcpy(ranks[r].flags_tbl, flag_ptrs.data(), world * sizeof(int64_t));
  }
  for (int r = 0; r < world; ++r) ranks[r].q.wait();

  // Expected exact reduced value for output[local_t][i]:
  //   sum_{r=0}^{world-1} contrib(r, block, i) with wraparound (uint32).
  auto expected = [&](int64_t i) -> uint32_t {
    uint32_t s = 0;
    for (int r = 0; r < world; ++r) s += (uint32_t)((r + 1) * 1000003u + (uint32_t)i);
    return s;
  };

  VariantEntry variants[] = {
      {"PLAIN   ", &launch_dispatch<V_PLAIN>},
      {"UC_UC   ", &launch_dispatch<V_UC_UC>},
      {"UC_L3WB ", &launch_dispatch<V_UC_L3WB>},
      {"WT_L3UC ", &launch_dispatch<V_WT_L3UC>},
      {"WT_L3WB ", &launch_dispatch<V_WT_L3WB>},
      {"S_L3WB  ", &launch_dispatch<V_S_L3WB>},
      {"WB_L3WB ", &launch_dispatch<V_WB_L3WB>},
  };

  printf("world=%d hidden=%ld live_tokens=%ld cap_tokens=%ld (below_cap=%s) "
         "num_wg=%d iters=%d\n",
         world, hidden, live, max_tokens,
         (live < max_tokens ? "yes" : "no"), num_wg, iters);
  // Cross-device bytes pushed per iteration (peer stores only): each rank does
  // (world-1) block-pushes of live*hidden*4 bytes.
  const double bytes_per_iter =
      (double)world * (world - 1) * live * hidden * sizeof(uint32_t);
  printf("%-9s  %-8s  %-10s  %-12s  %s\n", "variant", "correct", "max_diff",
         "ms/iter", "eff_GB/s(xdev)");

  uint32_t tag = 0;
  std::vector<uint32_t> host_out(out_elems);
  for (auto& ve : variants) {
    // warmup
    for (int it = 0; it < warmup; ++it) { ++tag; ve.fn(ranks, world, hidden, live, tpw, num_wg, tag, threads); }
    // timed
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) { ++tag; ve.fn(ranks, world, hidden, live, tpw, num_wg, tag, threads); }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    // correctness: check rank 0's output block against exact sum
    ranks[0].q.memcpy(host_out.data(), ranks[0].output, out_elems * sizeof(uint32_t)).wait();
    uint64_t bad = 0; uint32_t maxd = 0;
    for (int64_t t = 0; t < live; ++t)
      for (int64_t i = 0; i < hidden; ++i) {
        uint32_t got = host_out[t * hidden + i];
        uint32_t exp = expected(i);
        if (got != exp) { bad++; uint32_t d = got > exp ? got - exp : exp - got; if (d > maxd) maxd = d; }
      }
    double gbps = bytes_per_iter / (ms * 1e-3) / 1e9;
    printf("%-9s  %-8s  %-10u  %-12.4f  %.1f\n",
           ve.name, bad == 0 ? "OK" : "STALE", maxd, ms, gbps);
  }

  printf("\nNote: single-process P2P is more coherent than the real multi-process\n"
         "Level-Zero IPC mapping, so WB does not go STALE here. Rank variants by\n"
         "bandwidth above, then validate coherence under IPC with the multi-process\n"
         "test_symm_buffer_unpermute_capacity_regression_dist.py.\n");

  for (int r = 0; r < world; ++r) {
    sycl::free(ranks[r].acc, ctx);
    sycl::free(ranks[r].flag, ctx);
    sycl::free(ranks[r].output, ctx);
    sycl::free(ranks[r].accs_tbl, ctx);
    sycl::free(ranks[r].flags_tbl, ctx);
  }
  return 0;
}
