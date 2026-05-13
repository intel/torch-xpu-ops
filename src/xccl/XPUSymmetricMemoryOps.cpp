/*
 * XPU SymmetricMemory custom Ops
 *
 * Implements `symm_mem::one_shot_all_reduce` and `symm_mem::two_shot_all_reduce_`
 * on XPU, mirroring the CUDA implementation in
 * torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryOps.cu.
 *
 * Algorithm references:
 *   - Upstream CUDA: CUDASymmetricMemoryOps.cu (one_shot/two_shot kernels)
 *   - BMG SYCL     : sycl-tla/examples/00_bmg_gemm/gemm_allreduce_kernel.hpp
 *                    (allreduce_device<NUM_PER_TH>)
 *
 * Currently supports: dtype ∈ {float32, float16, bfloat16}, op="sum",
 * world_size ∈ {2,4,8}.
 */

#include <ATen/ATen.h>
#include <ATen/ceil_div.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <comm/SYCLHelpers.h>
#include <torch/library.h>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <xccl/Signal.hpp>

namespace c10d {
namespace symmetric_memory {

// Anchor symbol referenced by XPUSymmetricMemoryUtils.cpp to prevent
// --as-needed from dropping this SYCL .so as a DT_NEEDED of libtorch_xpu.so.
int xpu_symm_mem_ops_force_link = 0;

// ============================================================================
// Dispatch macros (mirror of CUDA side)
// ============================================================================
#define XPU_INT_SWITCH_CASE(name, val, ...) \
  case val: {                               \
    constexpr int name = val;               \
    __VA_ARGS__();                          \
    break;                                  \
  }

#define XPU_DISPATCH_WORLD_SIZES(world_size, ...)           \
  switch (world_size) {                                     \
    XPU_INT_SWITCH_CASE(k_world_size, 8, __VA_ARGS__);      \
    XPU_INT_SWITCH_CASE(k_world_size, 4, __VA_ARGS__);      \
    XPU_INT_SWITCH_CASE(k_world_size, 2, __VA_ARGS__);      \
    default: {                                              \
      TORCH_CHECK(                                          \
          false,                                            \
          "Not implemented for world_size=",                \
          world_size);                                      \
    }                                                       \
  }

// ============================================================================
// Launch configuration helpers
// ============================================================================
constexpr int kOneShotMaxNumGroups = 24;
constexpr int kOneShotMaxNumThreads = 512;
constexpr int kTwoShotMaxNumGroups = 24;
constexpr int kTwoShotMaxNumThreads = 512;
// 128-bit vectorized load: 16 bytes per thread per iteration.
constexpr int kVecBytes = 16;

template <typename T>
constexpr int elems_per_vec() {
  return kVecBytes / static_cast<int>(sizeof(T));
}

// Opaque 16-byte aligned vector holder so we can do 128-bit loads/stores
// for any scalar type (float / half / bfloat16) without relying on
// sycl::vec<bfloat16,N> arithmetic operators being defined.
template <typename T, int N>
struct alignas(kVecBytes) VecT {
  T data[N];
};

static inline void init_launch_cfg_1d(
    int64_t numel,
    int64_t elems_per_thread,
    int max_groups,
    int max_threads,
    int64_t& num_groups,
    int64_t& num_threads) {
  int64_t total_vec = at::ceil_div(numel, elems_per_thread);
  if (total_vec <= max_threads) {
    num_groups = 1;
    num_threads = std::max<int64_t>(32, (total_vec + 31) / 32 * 32);
  } else {
    num_groups = std::min<int64_t>(
        at::ceil_div(total_vec, (int64_t)max_threads), (int64_t)max_groups);
    num_threads = max_threads;
  }
}

// ============================================================================
// Kernel functors (out of anonymous namespace for SYCL AOT)
// ============================================================================

// ============================================================================
// Fused one-shot all-reduce: inlined per-workgroup signal-pad barrier.
// ============================================================================
// Signal-pad layout for this kernel (per rank, in uint32 slots):
//   [kFusedSignalBaseU32 ..                                ) pre-barrier
//   [kFusedSignalBaseU32 + kOneShotMaxNumGroups*kWorldSize ) post-barrier
// Slot [base+region_off + group_id*ws + src_rank] is written by src_rank
// (put_signal into peer's pad) and cleared by the owner rank (wait_signal).
// After an exchange round all slots return to 0, so the region can be reused
// across consecutive fused calls without an explicit reset.
//
// kFusedSignalBaseU32 is placed well beyond the `channel 0` region used by
// symm_mem->barrier(channel=0) so the two paths don't collide if mixed.
constexpr int kFusedSignalBaseU32 = 512;

template <typename scalar_t, int kWorldSize>
struct FusedOneShotAllReduceSumKernel {
  static constexpr int kN = elems_per_vec<scalar_t>();
  using Vec = VecT<scalar_t, kN>;

  scalar_t** peer_ptrs;
  scalar_t* output_ptr;
  uint32_t** signal_pads;
  int64_t input_offset;
  int64_t numel;
  int my_rank;

  static inline uint32_t* slot_of(
      uint32_t** signal_pads,
      int owner_rank,
      int region,
      int group_id,
      int src_rank) {
    const int64_t region_off =
        (int64_t)region * kOneShotMaxNumGroups * kWorldSize;
    return signal_pads[owner_rank] + kFusedSignalBaseU32 + region_off +
        (int64_t)group_id * kWorldSize + src_rank;
  }

  inline void wg_barrier_pre(sycl::nd_item<1> item) const {
    const auto lid = item.get_local_id(0);
    const auto group_id = item.get_group(0);
    if (lid < (size_t)kWorldSize) {
      int peer = static_cast<int>(lid);
      if (peer != my_rank) {
        uint32_t* put_addr = slot_of(
            signal_pads, peer, /*region=*/0, group_id, my_rank);
        uint32_t* wait_addr = slot_of(
            signal_pads, my_rank, /*region=*/0, group_id, peer);
        ::c10d::symmetric_memory::put_signal<
            std::memory_order_release>(put_addr);
        ::c10d::symmetric_memory::wait_signal<
            std::memory_order_acquire>(wait_addr);
      }
    }
    // Gate the non-barrier threads in this WG on the signal exchange above.
    // Local-scope fence is sufficient: the put/wait already issue
    // system-scope atomic_fence(release/acquire) internally, so cross-device
    // memory ordering is already guaranteed.
    item.barrier(sycl::access::fence_space::local_space);
  }

  inline void wg_barrier_post(sycl::nd_item<1> item) const {
    // First ensure all threads in this WG have finished their reads from
    // peer buffers before we signal peers that their buffers are free.
    item.barrier(sycl::access::fence_space::local_space);

    const auto lid = item.get_local_id(0);
    const auto group_id = item.get_group(0);
    if (lid < (size_t)kWorldSize) {
      int peer = static_cast<int>(lid);
      if (peer != my_rank) {
        uint32_t* put_addr = slot_of(
            signal_pads, peer, /*region=*/1, group_id, my_rank);
        uint32_t* wait_addr = slot_of(
            signal_pads, my_rank, /*region=*/1, group_id, peer);
        ::c10d::symmetric_memory::put_signal<
            std::memory_order_release>(put_addr);
        ::c10d::symmetric_memory::wait_signal<
            std::memory_order_acquire>(wait_addr);
      }
    }
    // No trailing item.barrier: nothing in this WG runs after the post
    // barrier; the kernel exits immediately and the XPU stream provides
    // queue-level ordering for the caller.
  }

  void operator()(sycl::nd_item<1> item) const {
    // pre-barrier: all peers have their buffers filled.
    wg_barrier_pre(item);

    const int64_t tid = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t stride = static_cast<int64_t>(item.get_global_range(0));
    const int64_t vec_total = numel / kN;

    // Rank rotation: each rank starts reading from its own (local) buffer
    // and fans out to peers in (my_rank+step) % kWorldSize order so at any
    // instant the kWorldSize ranks target distinct peers, avoiding an
    // all-to-one fabric hot-spot on PCIe-P2P.
    for (int64_t v = tid; v < vec_total; v += stride) {
      const int64_t elem_idx = v * kN + input_offset;
      Vec acc = *reinterpret_cast<const Vec*>(peer_ptrs[my_rank] + elem_idx);
#pragma unroll
      for (int step = 1; step < kWorldSize; ++step) {
        const int p = (my_rank + step) % kWorldSize;
        Vec rhs = *reinterpret_cast<const Vec*>(peer_ptrs[p] + elem_idx);
#pragma unroll
        for (int i = 0; i < kN; ++i) {
          acc.data[i] = static_cast<scalar_t>(
              static_cast<float>(acc.data[i]) +
              static_cast<float>(rhs.data[i]));
        }
      }
      *reinterpret_cast<Vec*>(output_ptr + v * kN) = acc;
    }
    if (tid == 0) {
      for (int64_t i = vec_total * kN; i < numel; ++i) {
        float a = static_cast<float>(peer_ptrs[my_rank][i + input_offset]);
#pragma unroll
        for (int step = 1; step < kWorldSize; ++step) {
          const int p = (my_rank + step) % kWorldSize;
          a += static_cast<float>(peer_ptrs[p][i + input_offset]);
        }
        output_ptr[i] = static_cast<scalar_t>(a);
      }
    }

    // post-barrier: prevent peers from overwriting their buffers before we
    // have finished reading them.
    wg_barrier_post(item);
  }
};


// ============================================================================
// Fused two-shot all-reduce: RS + AG in a single kernel with three inlined
// per-WG signal-pad barriers (pre / RS->AG mid / post).
// ============================================================================
// Key invariant: all ranks launch with identical (num_groups, num_threads),
// so the (group_id, lid) -> v mapping is identical on every rank. A per-WG
// signal exchange therefore guarantees that abs_elem written by RS on the
// owner rank is visible to the same WG's AG read on every rank.
//
// Signal-pad layout for this kernel (per rank, in uint32 slots):
//   [base + 0*Region ..) pre
//   [base + 1*Region ..) mid
//   [base + 2*Region ..) post
// where Region = kTwoShotMaxNumGroups * kWorldSize.
// kFusedTwoShotSignalBaseU32 sits beyond the fused one_shot region
// [512, 512+2*24*8) = [512, 896), so the two paths can coexist.
constexpr int kFusedTwoShotSignalBaseU32 = 1024;

template <typename scalar_t, int kWorldSize>
struct FusedTwoShotAllReduceSumKernel {
  static constexpr int kN = elems_per_vec<scalar_t>();
  using Vec = VecT<scalar_t, kN>;

  scalar_t** peer_ptrs;
  scalar_t* self_ptr;
  uint32_t** signal_pads;
  int64_t shard_numel;
  int64_t total_numel;
  int my_rank;

  static inline uint32_t* slot_of(
      uint32_t** signal_pads,
      int owner_rank,
      int region,
      int group_id,
      int src_rank) {
    const int64_t region_off =
        (int64_t)region * kTwoShotMaxNumGroups * kWorldSize;
    return signal_pads[owner_rank] + kFusedTwoShotSignalBaseU32 + region_off +
        (int64_t)group_id * kWorldSize + src_rank;
  }

  inline void wg_barrier(sycl::nd_item<1> item, int region) const {
    // Local fence: all threads in this WG must finish their device-memory
    // writes before any thread in this WG signals peers.
    item.barrier(sycl::access::fence_space::local_space);

    const auto lid = item.get_local_id(0);
    const auto group_id = item.get_group(0);
    if (lid < (size_t)kWorldSize) {
      int peer = static_cast<int>(lid);
      if (peer != my_rank) {
        uint32_t* put_addr = slot_of(
            signal_pads, peer, region, group_id, my_rank);
        uint32_t* wait_addr = slot_of(
            signal_pads, my_rank, region, group_id, peer);
        ::c10d::symmetric_memory::put_signal<
            std::memory_order_release>(put_addr);
        ::c10d::symmetric_memory::wait_signal<
            std::memory_order_acquire>(wait_addr);
      }
    }
    // Gate non-barrier threads on the signal exchange. Local fence is OK
    // because put/wait already issue system-scope atomic_fence internally.
    item.barrier(sycl::access::fence_space::local_space);
  }

  void operator()(sycl::nd_item<1> item) const {
    const int64_t tid = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t stride = static_cast<int64_t>(item.get_global_range(0));
    const int64_t vec_total = total_numel / kN;

    // pre-barrier: all peers have finished filling their input symm buffers.
    wg_barrier(item, /*region=*/0);

    // RS phase: only the owner of each abs_elem reduces across peers and
    // writes to its own buffer (self).
    for (int64_t v = tid; v < vec_total; v += stride) {
      const int64_t abs_elem = v * kN;
      int owner = static_cast<int>(abs_elem / shard_numel);
      if (owner >= kWorldSize) owner = kWorldSize - 1;
      if (owner == my_rank) {
        Vec acc = *reinterpret_cast<const Vec*>(peer_ptrs[0] + abs_elem);
#pragma unroll
        for (int p = 1; p < kWorldSize; ++p) {
          Vec rhs = *reinterpret_cast<const Vec*>(peer_ptrs[p] + abs_elem);
#pragma unroll
          for (int i = 0; i < kN; ++i) {
            acc.data[i] = static_cast<scalar_t>(
                static_cast<float>(acc.data[i]) +
                static_cast<float>(rhs.data[i]));
          }
        }
        *reinterpret_cast<Vec*>(self_ptr + abs_elem) = acc;
      }
    }

    // mid-barrier: RS writes on every owner are visible to AG readers.
    wg_barrier(item, /*region=*/1);

    // AG phase: read reduced shard from peer[owner] for elements not owned
    // by this rank.
    for (int64_t v = tid; v < vec_total; v += stride) {
      const int64_t abs_elem = v * kN;
      int owner = static_cast<int>(abs_elem / shard_numel);
      if (owner >= kWorldSize) owner = kWorldSize - 1;
      if (owner != my_rank) {
        Vec val = *reinterpret_cast<const Vec*>(peer_ptrs[owner] + abs_elem);
        *reinterpret_cast<Vec*>(self_ptr + abs_elem) = val;
      }
    }

    // post-barrier: prevent peers from overwriting their buffers before we
    // have finished reading them.
    wg_barrier(item, /*region=*/2);
  }
};

// ============================================================================
// Fused single-kernel ring all-reduce (Rabenseifner: N-1 RS + N-1 AG steps)
// ============================================================================
// Algorithm (pull-form): each rank has N shards (S/N elements each). Step k
// of phase RS: rank r reads left's shard (r-k-1)%N, accumulates into own.
// Step k of phase AG: rank r reads (=copies) left's shard (r-k)%N. After
// 2*(N-1) steps every rank has the fully reduced data in all N shards.
//
// Cross-rank sync: per-(step, wg_id) signal slot. WG_i posts to right's
// pad slot[(step, i)] after writing slice i in step k. Next-step start
// waits on self pad slot[(step-1, i)] (set by left). Last step's forward
// post is skipped (no peer reads it). End-of-iter back-sync: every WG
// posts to left's back slot, waits self back slot.
//
// Signal-pad layout (per rank, uint32 slots beyond kFusedTwoShotSignalBaseU32
// region used by fused_two_shot). Each ring step is split into kRingPipeHalves
// halves so step k+1 half h can fire while step k half (1-h) is still in
// flight (double-buffered pipeline).
//   forward[k, h, wg] = kFusedRingSignalBaseU32
//                       + (k * kRingPipeHalves + h) * num_wgs + wg
//   back[wg]          = kFusedRingSignalBaseU32
//                       + 2*(N-1)*kRingPipeHalves*num_wgs + wg
constexpr int kRingMaxNumGroups = 24;
constexpr int kRingMaxNumThreads = 512;
constexpr int kRingPipeHalves = 2;
// kFusedTwoShotSignalBaseU32 = 1024 + 3*kTwoShotMaxNumGroups*kWorldSize
// (worst case ws=8: 1024 + 3*24*8 = 1600). Place ring base at 1600.
// Worst-case fwd footprint: 2*(N-1)*kRingPipeHalves*kRingMaxNumGroups
// = 14*2*24 = 672 (ws=8). +24 back = 696. 1600 + 696 = 2296 < 2304 pad.
constexpr int kFusedRingSignalBaseU32 = 1600;

template <typename scalar_t, int kWorldSize>
struct FusedRingAllReduceSumKernel {
  static constexpr int kN = elems_per_vec<scalar_t>();
  using Vec = VecT<scalar_t, kN>;

  scalar_t** peer_ptrs;
  uint32_t** signal_pads;
  int64_t shard_numel; // = total_numel / kWorldSize (vec-aligned)
  int64_t total_numel; // original size; tail past N*shard_numel ignored
  int my_rank;
  int left_rank;
  int num_wgs; // == grid groups, <= kRingMaxNumGroups

  static inline uint32_t* fwd_slot(
      uint32_t** pads, int owner, int step, int half, int wg, int num_wgs_) {
    return pads[owner] + kFusedRingSignalBaseU32 +
        ((int64_t)step * kRingPipeHalves + half) * num_wgs_ + wg;
  }
  static inline uint32_t* back_slot(
      uint32_t** pads, int owner, int n_steps, int wg, int num_wgs_) {
    return pads[owner] + kFusedRingSignalBaseU32 +
        (int64_t)n_steps * kRingPipeHalves * num_wgs_ + wg;
  }

  void operator()(sycl::nd_item<1> item) const {
    const int wg_id = static_cast<int>(item.get_group(0));
    const auto lid = item.get_local_id(0);
    const auto local_size = static_cast<int64_t>(item.get_local_range(0));
    constexpr int n_steps = 2 * (kWorldSize - 1);

    const int64_t vec_total = shard_numel / kN;
    const int64_t vec_per_wg = (vec_total + num_wgs - 1) / num_wgs;
    const int64_t lo_vec = static_cast<int64_t>(wg_id) * vec_per_wg;
    const int64_t hi_vec =
        sycl::min(static_cast<int64_t>(wg_id + 1) * vec_per_wg, vec_total);
    const int64_t span_vec = hi_vec - lo_vec;

    for (int k = 0; k < n_steps; ++k) {
      const bool is_rs = (k < kWorldSize - 1);
      const int phase_k = is_rs ? k : (k - (kWorldSize - 1));
      const int recv_idx = is_rs
          ? (my_rank - phase_k - 1 + kWorldSize) % kWorldSize
          : (my_rank - phase_k + kWorldSize) % kWorldSize;
      const int64_t shard_off = (int64_t)recv_idx * shard_numel;

      scalar_t* my = peer_ptrs[my_rank] + shard_off;
      const scalar_t* left = peer_ptrs[left_rank] + shard_off;
      const bool shard_in_range = (shard_off + shard_numel) <= total_numel;
      const int right_rank = (my_rank + 1) % kWorldSize;

      // Process halves; per-half flag enables overlap of step (k+1, h) PCIe
      // read with peer's step (k, h+1) in flight (double-buffered pipeline).
      for (int h = 0; h < kRingPipeHalves; ++h) {
        const int64_t v_lo = lo_vec + (span_vec * h) / kRingPipeHalves;
        const int64_t v_hi = lo_vec + (span_vec * (h + 1)) / kRingPipeHalves;

        // Wait for left's previous-step same-half put on our slot.
        if (k > 0 && lid == 0) {
          ::c10d::symmetric_memory::wait_signal<std::memory_order_acquire>(
              fwd_slot(signal_pads, my_rank, k - 1, h, wg_id, num_wgs));
        }
        item.barrier(sycl::access::fence_space::local_space);

        if (shard_in_range && v_lo < v_hi) {
          for (int64_t v = v_lo + (int64_t)lid; v < v_hi; v += local_size) {
            const int64_t e = v * kN;
            const Vec rhs = *reinterpret_cast<const Vec*>(left + e);
            if (is_rs) {
              Vec lhs = *reinterpret_cast<const Vec*>(my + e);
#pragma unroll
              for (int i = 0; i < kN; ++i) {
                lhs.data[i] = static_cast<scalar_t>(
                    static_cast<float>(lhs.data[i]) +
                    static_cast<float>(rhs.data[i]));
              }
              *reinterpret_cast<Vec*>(my + e) = lhs;
            } else {
              *reinterpret_cast<Vec*>(my + e) = rhs;
            }
          }
        }

        item.barrier(sycl::access::fence_space::local_space);

        // Post to right's pad (skip last step: no peer reads it).
        if (k < n_steps - 1 && lid == 0) {
          ::c10d::symmetric_memory::put_signal_release_only<std::memory_order_release>(
              fwd_slot(signal_pads, right_rank, k, h, wg_id, num_wgs));
        }
      }
    }

    // End-of-iter backward sync: post to left, wait self.
    if (lid == 0) {
      ::c10d::symmetric_memory::put_signal_release_only<std::memory_order_release>(
          back_slot(signal_pads, left_rank, n_steps, wg_id, num_wgs));
      ::c10d::symmetric_memory::wait_signal<std::memory_order_acquire>(
          back_slot(signal_pads, my_rank, n_steps, wg_id, num_wgs));
    }
  }
};

// ============================================================================
// Fused 2D hierarchical ring all-reduce (ws=8 only): split 8 ranks into two
// groups of 4. Three phases:
//   A. Intra-group ring reduce-scatter (3 steps over kGS=4 ranks)
//      After: rank (g, p) holds shard p of shard_numel reduced over group g.
//   B. Cross-group exchange + sum (1 bilateral step, partner = my_rank XOR 4)
//      After: rank (g, p) holds shard p with full 8-rank sum.
//   C. Intra-group ring all-gather (3 steps over kGS=4 ranks)
//      After: every rank in each group has full output.
// Total step count = 7 vs flat ring 14: half the per-step barrier overhead.
// Per-rank PCIe traffic = 1.75*S (same bytes as flat ring; gain is purely
// from the step-count reduction).
//
// Signal-pad layout (per rank, uint32 slots beyond two_shot region):
//   slot[s, wg] = kFusedRing2DSignalBaseU32 + s * num_wgs + wg
//   s = 0..2  : Phase A intra-group fwd flags (left -> me, step k)
//   s = 3     : Phase B cross-group flag (partner -> me, phase A done)
//   s = 4..6  : Phase C intra-group fwd flags (left -> me, step k)
//   s = 7     : back-sync slot (intra-group, ring around 4)
//   s = 8     : back-sync slot (inter-group partner)
// total = 9 * num_wgs (max 9*24 = 216 slots). base 1600 + 216 = 1816 < 2304 ✓.
constexpr int kFusedRing2DSignalBaseU32 = 1600;
constexpr int kRing2DGroupSize = 4;

template <typename scalar_t>
struct FusedRing2DAllReduceSumKernel {
  static constexpr int kN = elems_per_vec<scalar_t>();
  using Vec = VecT<scalar_t, kN>;
  static constexpr int kGS = kRing2DGroupSize; // 4
  static constexpr int kRsAgSteps = kGS - 1;   // 3
  // Slot offsets:
  static constexpr int kSlotPhaseA0 = 0;
  static constexpr int kSlotPhaseB = kRsAgSteps;            // 3
  static constexpr int kSlotPhaseC0 = kRsAgSteps + 1;       // 4
  static constexpr int kSlotBackIntra = 2 * kRsAgSteps + 1; // 7
  static constexpr int kSlotBackInter = 2 * kRsAgSteps + 2; // 8

  scalar_t** peer_ptrs;
  uint32_t** signal_pads;
  int64_t shard_numel; // = total_numel / kGS (NOT total_numel / world_size!)
  int64_t total_numel;
  int my_rank;
  int group_id;        // my_rank / kGS
  int pos;             // my_rank % kGS
  int left_in_group;   // ((pos - 1 + kGS) % kGS) + group_id * kGS
  int right_in_group;  // ((pos + 1) % kGS) + group_id * kGS
  int partner;         // my_rank XOR kGS  (= my_rank XOR 4)
  int num_wgs;

  static inline uint32_t* slot(
      uint32_t** pads, int owner, int s, int wg, int num_wgs_) {
    return pads[owner] + kFusedRing2DSignalBaseU32 +
        (int64_t)s * num_wgs_ + wg;
  }

  void operator()(sycl::nd_item<1> item) const {
    const int wg_id = static_cast<int>(item.get_group(0));
    const auto lid = item.get_local_id(0);
    const auto local_size = static_cast<int64_t>(item.get_local_range(0));

    const int64_t vec_total = shard_numel / kN;
    const int64_t vec_per_wg = (vec_total + num_wgs - 1) / num_wgs;
    const int64_t lo_vec = static_cast<int64_t>(wg_id) * vec_per_wg;
    const int64_t hi_vec =
        sycl::min(static_cast<int64_t>(wg_id + 1) * vec_per_wg, vec_total);

    scalar_t* my_buf = peer_ptrs[my_rank];
    const scalar_t* left_buf = peer_ptrs[left_in_group];
    const scalar_t* partner_buf = peer_ptrs[partner];

    auto reduce_add_range = [&](scalar_t* dst, const scalar_t* src) {
      for (int64_t v = lo_vec + (int64_t)lid; v < hi_vec; v += local_size) {
        const int64_t e = v * kN;
        const Vec rhs = *reinterpret_cast<const Vec*>(src + e);
        Vec lhs = *reinterpret_cast<const Vec*>(dst + e);
#pragma unroll
        for (int i = 0; i < kN; ++i) {
          lhs.data[i] = static_cast<scalar_t>(
              static_cast<float>(lhs.data[i]) +
              static_cast<float>(rhs.data[i]));
        }
        *reinterpret_cast<Vec*>(dst + e) = lhs;
      }
    };
    auto copy_range = [&](scalar_t* dst, const scalar_t* src) {
      for (int64_t v = lo_vec + (int64_t)lid; v < hi_vec; v += local_size) {
        const int64_t e = v * kN;
        *reinterpret_cast<Vec*>(dst + e) =
            *reinterpret_cast<const Vec*>(src + e);
      }
    };

    // ===== Phase A: intra-group ring reduce-scatter =====
    // At step k (k=0..2): receive shard idx_a from left_in_group, add into
    // self at the same offset.
    for (int k = 0; k < kRsAgSteps; ++k) {
      const int recv_idx = (pos - k - 1 + kGS) % kGS;
      const int64_t shard_off = (int64_t)recv_idx * shard_numel;

      if (k > 0 && lid == 0) {
        ::c10d::symmetric_memory::wait_signal<std::memory_order_acquire>(
            slot(signal_pads, my_rank, kSlotPhaseA0 + k - 1, wg_id, num_wgs));
      }
      item.barrier(sycl::access::fence_space::local_space);

      const bool in_range = (shard_off + shard_numel) <= total_numel;
      if (in_range && lo_vec < hi_vec) {
        reduce_add_range(my_buf + shard_off, left_buf + shard_off);
      }
      item.barrier(sycl::access::fence_space::local_space);

      // Skip post on last step (k == kRsAgSteps-1): right has no Phase A wait
      // for it, and Phase B uses its own bilateral barrier instead.
      if (k < kRsAgSteps - 1 && lid == 0) {
        ::c10d::symmetric_memory::put_signal_release_only<std::memory_order_release>(
            slot(signal_pads,
                 right_in_group,
                 kSlotPhaseA0 + k,
                 wg_id,
                 num_wgs));
      }
    }

    // ===== Phase B: cross-group exchange =====
    // Bilateral barrier: each partner posts/waits one slot. Then read partner's
    // shard `pos` (which holds partner's group's local sum) and add into self.
    if (lid == 0) {
      ::c10d::symmetric_memory::put_signal_release_only<std::memory_order_release>(
          slot(signal_pads, partner, kSlotPhaseB, wg_id, num_wgs));
      ::c10d::symmetric_memory::wait_signal<std::memory_order_acquire>(
          slot(signal_pads, my_rank, kSlotPhaseB, wg_id, num_wgs));
    }
    item.barrier(sycl::access::fence_space::local_space);

    {
      const int64_t shard_off = (int64_t)pos * shard_numel;
      const bool in_range = (shard_off + shard_numel) <= total_numel;
      if (in_range && lo_vec < hi_vec) {
        reduce_add_range(my_buf + shard_off, partner_buf + shard_off);
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    // After Phase B: every rank now holds shard `pos` with full 8-rank sum.

    // ===== Phase C: intra-group ring all-gather =====
    // Post initial start signal to right_in_group at slot kSlotPhaseC0+0
    // (right's step-0 wait), indicating "my shard pos is full-sum, you can
    // read it as your shard pos-1".
    if (lid == 0) {
      ::c10d::symmetric_memory::put_signal_release_only<std::memory_order_release>(
          slot(signal_pads,
               right_in_group,
               kSlotPhaseC0 + 0,
               wg_id,
               num_wgs));
    }

    for (int k = 0; k < kRsAgSteps; ++k) {
      const int recv_idx = (pos - k - 1 + kGS) % kGS;
      const int64_t shard_off = (int64_t)recv_idx * shard_numel;

      if (lid == 0) {
        ::c10d::symmetric_memory::wait_signal<std::memory_order_acquire>(
            slot(signal_pads, my_rank, kSlotPhaseC0 + k, wg_id, num_wgs));
      }
      item.barrier(sycl::access::fence_space::local_space);

      const bool in_range = (shard_off + shard_numel) <= total_numel;
      if (in_range && lo_vec < hi_vec) {
        copy_range(my_buf + shard_off, left_buf + shard_off);
      }
      item.barrier(sycl::access::fence_space::local_space);

      // Signal right that shard recv_idx is now ready for their step k+1.
      if (k < kRsAgSteps - 1 && lid == 0) {
        ::c10d::symmetric_memory::put_signal_release_only<std::memory_order_release>(
            slot(signal_pads,
                 right_in_group,
                 kSlotPhaseC0 + k + 1,
                 wg_id,
                 num_wgs));
      }
    }

    // End-of-iter back sync: full 8-way (intra-group ring + inter-group pair)
    // to ensure all partners are done before any rank may start the next AR.
    if (lid == 0) {
      ::c10d::symmetric_memory::put_signal_release_only<std::memory_order_release>(
          slot(signal_pads, left_in_group, kSlotBackIntra, wg_id, num_wgs));
      ::c10d::symmetric_memory::put_signal_release_only<std::memory_order_release>(
          slot(signal_pads, partner, kSlotBackInter, wg_id, num_wgs));
      ::c10d::symmetric_memory::wait_signal<std::memory_order_acquire>(
          slot(signal_pads, my_rank, kSlotBackIntra, wg_id, num_wgs));
      ::c10d::symmetric_memory::wait_signal<std::memory_order_acquire>(
          slot(signal_pads, my_rank, kSlotBackInter, wg_id, num_wgs));
    }
  }
};

namespace {

// ============================================================================
// Launch helpers
// ============================================================================

template <typename scalar_t, int kWorldSize>
static void launch_fused_one_shot_all_reduce_sum(
    scalar_t** peer_ptrs,
    scalar_t* output_ptr,
    uint32_t** signal_pads,
    int my_rank,
    int64_t input_offset,
    int64_t numel,
    sycl::queue& q) {
  constexpr int kN = elems_per_vec<scalar_t>();
  int64_t num_groups = 0, num_threads = 0;
  init_launch_cfg_1d(
      numel,
      kN,
      kOneShotMaxNumGroups,
      kOneShotMaxNumThreads,
      num_groups,
      num_threads);
  // Ensure local_size >= kWorldSize so the first kWorldSize threads can each
  // handle one peer for the inlined barrier.
  if (num_threads < kWorldSize) {
    num_threads = kWorldSize;
  }

  FusedOneShotAllReduceSumKernel<scalar_t, kWorldSize> ker{
      peer_ptrs,
      output_ptr,
      signal_pads,
      input_offset,
      numel,
      my_rank};
  sycl_kernel_submit(
      /*global_range=*/num_groups * num_threads,
      /*local_range=*/num_threads,
      q,
      ker);
}

template <typename scalar_t, int kWorldSize>
static void launch_fused_two_shot_all_reduce_sum(
    scalar_t** peer_ptrs,
    scalar_t* self_ptr,
    uint32_t** signal_pads,
    int my_rank,
    int64_t shard_numel,
    int64_t total_numel,
    sycl::queue& q) {
  constexpr int kN = elems_per_vec<scalar_t>();
  int64_t num_groups = 0, num_threads = 0;
  init_launch_cfg_1d(
      total_numel,
      kN,
      kTwoShotMaxNumGroups,
      kTwoShotMaxNumThreads,
      num_groups,
      num_threads);
  // Need at least kWorldSize threads per WG so each non-self peer gets a
  // dedicated thread for the inlined signal exchange.
  if (num_threads < kWorldSize) {
    num_threads = kWorldSize;
  }

  FusedTwoShotAllReduceSumKernel<scalar_t, kWorldSize> ker{
      peer_ptrs, self_ptr, signal_pads, shard_numel, total_numel, my_rank};
  sycl_kernel_submit(
      num_groups * num_threads, num_threads, q, ker);
}

template <typename scalar_t, int kWorldSize>
static void launch_fused_ring_all_reduce_sum(
    scalar_t** peer_ptrs,
    uint32_t** signal_pads,
    int my_rank,
    int64_t shard_numel,
    int64_t total_numel,
    sycl::queue& q) {
  // Launch num_groups WGs that collectively cover the shard's vector range.
  // Each WG owns a slice; per-(step, wg) signal slot syncs cross-rank.
  constexpr int kN = elems_per_vec<scalar_t>();
  int64_t num_groups = 0, num_threads = 0;
  init_launch_cfg_1d(
      shard_numel,
      kN,
      kRingMaxNumGroups,
      kRingMaxNumThreads,
      num_groups,
      num_threads);
  const int left_rank = (my_rank - 1 + kWorldSize) % kWorldSize;
  FusedRingAllReduceSumKernel<scalar_t, kWorldSize> ker{
      peer_ptrs,
      signal_pads,
      shard_numel,
      total_numel,
      my_rank,
      left_rank,
      static_cast<int>(num_groups)};
  sycl_kernel_submit(
      num_groups * num_threads, num_threads, q, ker);
}

// 2D hierarchical ring all-reduce launcher (ws=8 only). Splits 8 ranks into
// 2 groups of 4. shard_numel here = total_numel / kRing2DGroupSize (= /4),
// NOT total_numel / world_size. We also pass num_groups derived from the
// per-rank Phase-A/C work which acts on shard_numel-sized chunks.
template <typename scalar_t>
static void launch_fused_ring2d_all_reduce_sum(
    scalar_t** peer_ptrs,
    uint32_t** signal_pads,
    int my_rank,
    int64_t total_numel,
    sycl::queue& q) {
  constexpr int kN = elems_per_vec<scalar_t>();
  const int64_t shard_numel = total_numel / kRing2DGroupSize;
  int64_t num_groups = 0, num_threads = 0;
  init_launch_cfg_1d(
      shard_numel,
      kN,
      kRingMaxNumGroups,
      kRingMaxNumThreads,
      num_groups,
      num_threads);
  const int group_id = my_rank / kRing2DGroupSize;
  const int pos = my_rank % kRing2DGroupSize;
  const int left_in_group =
      ((pos - 1 + kRing2DGroupSize) % kRing2DGroupSize) +
      group_id * kRing2DGroupSize;
  const int right_in_group =
      ((pos + 1) % kRing2DGroupSize) + group_id * kRing2DGroupSize;
  const int partner = my_rank ^ kRing2DGroupSize;
  FusedRing2DAllReduceSumKernel<scalar_t> ker{
      peer_ptrs,
      signal_pads,
      shard_numel,
      total_numel,
      my_rank,
      group_id,
      pos,
      left_in_group,
      right_in_group,
      partner,
      static_cast<int>(num_groups)};
  sycl_kernel_submit(
      num_groups * num_threads, num_threads, q, ker);
}

// ============================================================================
// scalar_t dispatch helper
// ============================================================================
// Maps at::ScalarType to the sycl-compatible scalar_t used by the kernels.
// fp32       -> float
// fp16       -> sycl::half
// bfloat16   -> sycl::ext::oneapi::bfloat16
#define XPU_DISPATCH_FLOAT_HALF_BF16(TYPE, NAME, ...)                  \
  [&] {                                                                \
    switch (TYPE) {                                                    \
      case at::kFloat: {                                               \
        using scalar_t = float;                                        \
        return __VA_ARGS__();                                          \
      }                                                                \
      case at::kHalf: {                                                \
        using scalar_t = sycl::half;                                   \
        return __VA_ARGS__();                                          \
      }                                                                \
      case at::kBFloat16: {                                            \
        using scalar_t = sycl::ext::oneapi::bfloat16;                  \
        return __VA_ARGS__();                                          \
      }                                                                \
      default:                                                         \
        TORCH_CHECK(                                                   \
            false,                                                     \
            NAME,                                                      \
            ": unsupported dtype ",                                    \
            TYPE,                                                      \
            " (supported: float32, float16, bfloat16).");              \
    }                                                                  \
  }()

// ============================================================================
// Op entry points
// ============================================================================

at::Tensor one_shot_all_reduce_out_impl(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out) {
  TORCH_CHECK(
      input.is_contiguous(),
      "one_shot_all_reduce(xpu): input must be contiguous.");
  TORCH_CHECK(
      out.is_contiguous(),
      "one_shot_all_reduce(xpu): output must be contiguous.");
  TORCH_CHECK(
      out.sizes() == input.sizes(),
      "one_shot_all_reduce(xpu): input/output size mismatch.");
  TORCH_CHECK(
      reduce_op == "sum",
      "one_shot_all_reduce(xpu): only sum is supported (got ",
      reduce_op,
      ").");
  const auto dtype = input.scalar_type();
  TORCH_CHECK(
      dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16,
      "one_shot_all_reduce(xpu): only float32/float16/bfloat16 are supported (got ",
      dtype,
      ").");

  if (input.numel() == 0) {
    return out;
  }

  c10::DeviceGuard guard(input.device());
  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "one_shot_all_reduce(xpu): input must be allocated with symm_mem.empty().");
  const int world_size = symm_mem->get_world_size();
  const int my_rank = symm_mem->get_rank();
  const int64_t numel = input.numel();
  const int64_t input_offset = input.storage_offset();

  // Kernel-fused signal-pad barrier (inlined put/wait) — single supported
  // path for one_shot. Legacy host-barrier wrap was removed (dead code).

  sycl::queue& q = at::xpu::getCurrentXPUStream().queue();

  void** peer_ptrs_raw = symm_mem->get_buffer_ptrs_dev();
  void* out_ptr_raw = out.data_ptr();

  uint32_t** signal_pads =
      reinterpret_cast<uint32_t**>(symm_mem->get_signal_pad_ptrs_dev());
  XPU_DISPATCH_FLOAT_HALF_BF16(dtype, "one_shot_all_reduce(xpu)", [&]() {
    scalar_t** peer_ptrs = reinterpret_cast<scalar_t**>(peer_ptrs_raw);
    scalar_t* out_ptr = reinterpret_cast<scalar_t*>(out_ptr_raw);
    XPU_DISPATCH_WORLD_SIZES(world_size, [&]() {
      launch_fused_one_shot_all_reduce_sum<scalar_t, k_world_size>(
          peer_ptrs,
          out_ptr,
          signal_pads,
          my_rank,
          input_offset,
          numel,
          q);
    });
  });
  return out;
}

at::Tensor one_shot_all_reduce_xpu(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name) {
  auto out = at::empty_like(input);
  return one_shot_all_reduce_out_impl(
      input, std::move(reduce_op), std::move(group_name), out);
}

at::Tensor one_shot_all_reduce_out_xpu(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out) {
  return one_shot_all_reduce_out_impl(
      input, std::move(reduce_op), std::move(group_name), out);
}

at::Tensor one_shot_all_reduce_copy_xpu(
    const at::Tensor& symm_buffer,
    const at::Tensor& local_input,
    std::string reduce_op,
    std::string group_name) {
  TORCH_CHECK(
      local_input.is_contiguous() && symm_buffer.is_contiguous(),
      "one_shot_all_reduce_copy(xpu): both tensors must be contiguous.");
  TORCH_CHECK(
      local_input.numel() <= symm_buffer.numel(),
      "one_shot_all_reduce_copy(xpu): local_input must fit in symm_buffer.");
  auto slot = symm_buffer.narrow(0, 0, local_input.numel());
  slot.copy_(local_input);
  auto full = one_shot_all_reduce_xpu(
      symm_buffer, std::move(reduce_op), std::move(group_name));
  return full.narrow(0, 0, local_input.numel()).clone();
}

at::Tensor one_shot_all_reduce_copy_out_xpu(
    const at::Tensor& symm_buffer,
    const at::Tensor& local_input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out) {
  TORCH_CHECK(
      local_input.is_contiguous() && symm_buffer.is_contiguous() &&
          out.is_contiguous(),
      "one_shot_all_reduce_copy_out(xpu): tensors must be contiguous.");
  auto slot = symm_buffer.narrow(0, 0, local_input.numel());
  slot.copy_(local_input);
  auto full_out = at::empty_like(symm_buffer);
  one_shot_all_reduce_out_impl(
      symm_buffer, std::move(reduce_op), std::move(group_name), full_out);
  out.copy_(full_out.narrow(0, 0, local_input.numel()));
  return out;
}

at::Tensor two_shot_all_reduce_out_impl(
    at::Tensor& input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor output) {
  TORCH_CHECK(
      input.is_contiguous(),
      "two_shot_all_reduce(xpu): input must be contiguous.");
  TORCH_CHECK(
      output.is_contiguous(),
      "two_shot_all_reduce(xpu): output must be contiguous.");
  TORCH_CHECK(
      output.sizes() == input.sizes(),
      "two_shot_all_reduce(xpu): input/output size mismatch.");
  TORCH_CHECK(
      reduce_op == "sum",
      "two_shot_all_reduce(xpu): only sum is supported.");
  const auto dtype = input.scalar_type();
  TORCH_CHECK(
      dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16,
      "two_shot_all_reduce(xpu): only float32/float16/bfloat16 are supported (got ",
      dtype,
      ").");

  if (input.numel() == 0) {
    return output;
  }

  c10::DeviceGuard guard(input.device());
  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "two_shot_all_reduce(xpu): input must be allocated with symm_mem.empty().");
  const int world_size = symm_mem->get_world_size();
  const int rank = symm_mem->get_rank();
  const int64_t numel = input.numel();

  // Round shard_numel up to the dtype's vec unit so each rank's shard is
  // vec-aligned. Use the worst case (fp32 = 4 elems) which is compatible with
  // bf16/fp16 (8 elems) -> pick the dtype-specific value.
  const int64_t elems_per_vec_val =
      dtype == at::kFloat ? 4 : 8;  // 16B / sizeof(scalar_t)
  int64_t shard_numel =
      ((at::ceil_div(numel, (int64_t)world_size) + elems_per_vec_val - 1) /
       elems_per_vec_val) *
      elems_per_vec_val;

  sycl::queue& q = at::xpu::getCurrentXPUStream().queue();

  void** peer_ptrs_raw = symm_mem->get_buffer_ptrs_dev();
  void* self_ptr_raw = symm_mem->get_buffer_ptrs()[rank];
  uint32_t** signal_pads =
      reinterpret_cast<uint32_t**>(symm_mem->get_signal_pad_ptrs_dev());

  // Auto routing by world_size:
  //   ws == 8 : 2D hierarchical ring (4-group x 2), 7 steps. ~3x faster than
  //             the python-pull baseline at large messages.
  //   ws == 4 : flat fused ring (Rabenseifner) with double-buffered halves.
  //   ws == 2 : fused two_shot single-kernel (degenerate ring == two_shot).
  //   other   : not supported.
  TORCH_CHECK(
      world_size == 2 || world_size == 4 || world_size == 8,
      "two_shot_all_reduce(xpu): unsupported world_size=",
      world_size,
      " (supported: 2, 4, 8).");
  XPU_DISPATCH_FLOAT_HALF_BF16(dtype, "two_shot_all_reduce(xpu)", [&]() {
    scalar_t** peer_ptrs = reinterpret_cast<scalar_t**>(peer_ptrs_raw);
    scalar_t* self_ptr = reinterpret_cast<scalar_t*>(self_ptr_raw);
    if (world_size == 8) {
      launch_fused_ring2d_all_reduce_sum<scalar_t>(
          peer_ptrs, signal_pads, rank, numel, q);
    } else if (world_size == 4) {
      launch_fused_ring_all_reduce_sum<scalar_t, /*kWorldSize=*/4>(
          peer_ptrs, signal_pads, rank, shard_numel, numel, q);
    } else {
      // world_size == 2
      launch_fused_two_shot_all_reduce_sum<scalar_t, /*kWorldSize=*/2>(
          peer_ptrs, self_ptr, signal_pads, rank, shard_numel, numel, q);
    }
  });
  if (!output.is_same(input)) {
    output.copy_(input);
  }
  return output;
}

at::Tensor two_shot_all_reduce_(
    at::Tensor input,
    std::string reduce_op,
    std::string group_name) {
  two_shot_all_reduce_out_impl(
      input, std::move(reduce_op), std::move(group_name), input);
  return input;
}

at::Tensor two_shot_all_reduce_out(
    at::Tensor input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor output) {
  return two_shot_all_reduce_out_impl(
      input, std::move(reduce_op), std::move(group_name), output);
}

} // namespace

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl(
      "one_shot_all_reduce",
      ::c10d::symmetric_memory::one_shot_all_reduce_xpu);
  m.impl(
      "one_shot_all_reduce_out",
      ::c10d::symmetric_memory::one_shot_all_reduce_out_xpu);
  m.impl(
      "one_shot_all_reduce_copy",
      ::c10d::symmetric_memory::one_shot_all_reduce_copy_xpu);
  m.impl(
      "one_shot_all_reduce_copy_out",
      ::c10d::symmetric_memory::one_shot_all_reduce_copy_out_xpu);
  m.impl(
      "two_shot_all_reduce_",
      ::c10d::symmetric_memory::two_shot_all_reduce_);
  m.impl(
      "two_shot_all_reduce_out",
      ::c10d::symmetric_memory::two_shot_all_reduce_out);
}

} // namespace symmetric_memory
} // namespace c10d
