// InternodeDispatchRdmaSender.cpp
//
// Standalone port of DeepSymm's `InternodeDispatchRDMASenderKernel`
// (intel-sandbox/DeepSymm: csrc/modules/moe_ep/internode.cpp), i.e. the FIRST
// stage of the two-level internode (legacy DeepEP) dispatch pipeline. Only the
// RDMA-sender stage is ported here (the forward/receive stages that further
// fan tokens out across NVLink peers on the destination node are out of
// scope); the observable effect of this op is that every "RDMA rank" (node)
// ends up with, in its own symmetric receive region, the token payloads any
// other node routed to it plus a ready-count per source node.
//
// Topology: `num_ranks` PEs are grouped into nodes of `kNumMaxNvlPeers` PEs
// each (`num_rdma_ranks = num_ranks / kNumMaxNvlPeers`); `rank`'s node index
// is `rank / kNumMaxNvlPeers` ("my_rdma") and its lane within the node is
// `rank % kNumMaxNvlPeers` ("my_nvl"). Each token carries a per-GLOBAL-rank
// `is_token_in_rank` bit; this kernel ORs those bits per destination node into
// a small per-node bitmask (which NVL lanes on that node still want the
// token) and stages+sends the token's payload (hidden + optional scales +
// source metadata + top-k indices/weights) to every node whose bitmask is
// non-zero -- via a local copy for the sender's own node, and an ISHMEM
// work-group RDMA put for every other node.
//
// Only ISHMEM APIs are used for the inter-node hop:
//   - ishmemx_putmem_nbi_work_group  (work-group-cooperative RDMA write)
//   - ishmemx_quiet_work_group / ishmemx_fence_work_group (order data/flag)
//   - ishmem_int_atomic_set          (leader publishes the ready count)
//   - ishmem_malloc / ishmem_free / ishmem_barrier_all (symmetric heap)
//
// Registered op: symm_mem::internode_dispatch_rdma_sender

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <ishmem.h>
#include <ishmemx.h>
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <type_traits>

namespace {

// One node holds at most this many NVLink/IPC peers (matches DeepSymm).
constexpr int kNumMaxNvlPeers = 2;

// Max bytes issued per ishmemx_putmem_nbi_work_group() call. Historically
// this was paired with an ishmemx_quiet_work_group() drain after every
// chunk to keep the number of outstanding un-drained WQEs bounded (see
// DeepSymm internode.cpp for the IBGDA-hang rationale this guarded against
// -- see also repro_ishmem_hang/README.md). The RDMA send path below no
// longer quiets after each chunk (it now streams chunks as they become
// ready and only quiets once, at the end, to overlap copying with sending
// -- see mark_slot_ready() in InternodeDispatchRdmaSenderKernel); this
// constant now only bounds how much of a team's already-ready range is put
// in a single ISHMEM call. Overridable via env for perf experimentation.
size_t put_chunk_bytes() {
  static const size_t v = [] {
    const char* s = std::getenv("INTERNODE_DISPATCH_PUT_CHUNK_BYTES");
    if (s != nullptr && *s != '\0') {
      long parsed = std::atol(s);
      if (parsed > 0) {
        return static_cast<size_t>(parsed);
      }
    }
    return static_cast<size_t>(32 * 1024);
  }();
  return v;
}

// Matches DeepSymm's kNumThreads (csrc/modules/moe_ep/internode.cpp): the
// per-token/local-copy vectorized paths below assume 32-lane sub-groups
// cooperating within a work-group of this size, and a smaller work-group
// changes the sender/forward work-group occupancy math (see kSenderWgsPerNode
// below), so this must stay aligned with the DeepSymm kernel to reproduce its
// performance characteristics.
constexpr int32_t kThreads = 512;
// Default sender work-groups cooperating per destination node.
constexpr int kSenderWgsPerNode = 4;

// 16-byte vector type (mirrors DeepSymm's csrc/sycl/configs.h `int4`), used
// below for 128-bit vectorized global-memory copies.
struct int4_t {
  int x, y, z, w;
};

// d32x4 vector store — single LSC instruction for 16 bytes per lane (mirrors
// DeepSymm's csrc/sycl/utils.hpp st_na_global_v).
template <typename T>
inline void st_na_global_v(T* ptr, T value) {
#ifdef __SYCL_DEVICE_ONLY__
  static_assert(sizeof(T) == 16, "st_na_global_v requires sizeof(T) == 16");
  using vec4_t = uint32_t __attribute__((ext_vector_type(4)));
  vec4_t tmp;
  __builtin_memcpy(&tmp, &value, 16);
  auto* addr = reinterpret_cast<void*>(ptr);
  asm volatile("lsc_store.ugm.wb.wb (M1, 32) flat[%0]:a64 %1:d32x4" : : "rw"(addr), "rw"(tmp) : "memory");
#else
  *ptr = value;
#endif
}

template <typename T>
inline T ld_nc_global_v(const T* ptr) {
#ifdef __SYCL_DEVICE_ONLY__
  static_assert(sizeof(T) == 16, "ld_nc_global_v requires sizeof(T) == 16");
  using vec4_t = uint32_t __attribute__((ext_vector_type(4)));
  vec4_t tmp;
  auto* addr = reinterpret_cast<const void*>(ptr);
  asm volatile("lsc_load.ugm.uc.ca (M1, 32) %0:d32x4 flat[%1]:a64" : "=rw"(tmp) : "rw"(addr));
  T result;
  __builtin_memcpy(&result, &tmp, 16);
  return result;
#else
  return *ptr;
#endif
}

// Cooperative copy: intended to be called by ALL lanes of a sub-group
// (0..31) with the SAME (dst, src, n) triple; the 32 lanes collectively
// cover the whole buffer (mirrors DeepSymm's UNROLLED_WARP_COPY(5, ...)).
#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                                                     \
  {                                                                                                                                   \
    constexpr int kLoopStride = 32 * (UNROLL_FACTOR);                                                                                \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)];                             \
    auto __src = (SRC);                                                                                                              \
    auto __dst = (DST);                                                                                                              \
    for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) {                                         \
      _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32);      \
      _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);       \
    }                                                                                                                                 \
    {                                                                                                                                 \
      int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID);                                                                       \
      _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                            \
        if (__i + __j * 32 < (N)) {                                                                                                  \
          unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32);                                                                    \
        }                                                                                                                             \
      }                                                                                                                               \
      _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                            \
        if (__i + __j * 32 < (N)) {                                                                                                  \
          ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);                                                                     \
        }                                                                                                                             \
      }                                                                                                                               \
    }                                                                                                                                 \
  }

// Sub-group-cooperative vectorized copy: all lanes of a sub-group call this
// with the SAME (dst, src, n) triple (e.g. a single, whole-buffer local copy
// shared by a work-group). NOT safe when each lane owns a distinct
// token/row -- use the single-thread overload below for that case (mirrors
// DeepSymm's csrc/modules/moe_ep/internode.cpp copy_bytes_vectorized).
inline void copy_bytes_vectorized(uint8_t* dst, const uint8_t* src, size_t n, int lane_id) {
  if (n == 0) {
    return;
  }
  const size_t num_int4 = n / sizeof(int4_t);
  const size_t tail_bytes = n % sizeof(int4_t);
  if (num_int4 > 0) {
    auto* dst_int4 = reinterpret_cast<int4_t*>(dst);
    auto* src_int4 = reinterpret_cast<const int4_t*>(src);
    UNROLLED_WARP_COPY(5, lane_id, static_cast<int>(num_int4), dst_int4, src_int4, ld_nc_global_v, st_na_global_v);
  }
  if (tail_bytes > 0 && lane_id == 0) {
    const size_t base = num_int4 * sizeof(int4_t);
    for (size_t b = 0; b < tail_bytes; ++b) {
      dst[base + b] = src[base + b];
    }
  }
}

// Whole-WORK-GROUP-cooperative vectorized copy: splits [src, src+n) into
// num_sgs contiguous slices (one per sub-group in the work-group), then each
// sub-group lane-cooperatively copies its own slice via the 32-lane
// UNROLLED_WARP_COPY above. This is the fix for a real perf bug: calling the
// sub-group overload directly with a bare `lane_id` (0..31) from EVERY
// sub-group in a multi-sub-group work-group makes all of them redundantly
// copy the exact SAME byte range (idempotent, but up to (work-group
// size/32)x wasted memory bandwidth) instead of splitting the range across
// them -- use this overload instead for any whole-work-group shared local
// copy (e.g. the own-node copy in the sender kernel below).
inline void copy_bytes_vectorized_wg(uint8_t* dst, const uint8_t* src, size_t n, sycl::nd_item<1>& item) {
  if (n == 0) {
    return;
  }
  auto sg = item.get_sub_group();
  const int32_t sg_size = static_cast<int32_t>(sg.get_local_range()[0]);
  const int32_t lsize = static_cast<int32_t>(item.get_local_range(0));
  const int32_t num_sgs = lsize / sg_size;
  const int32_t sg_id = static_cast<int32_t>(sg.get_group_id()[0]);
  const int32_t lane_id = static_cast<int32_t>(sg.get_local_id()[0]);

  const size_t num_int4 = n / sizeof(int4_t);
  const size_t tail_bytes = n % sizeof(int4_t);
  if (num_int4 > 0) {
    const size_t start = (num_int4 * static_cast<size_t>(sg_id)) / static_cast<size_t>(num_sgs);
    const size_t end = (num_int4 * static_cast<size_t>(sg_id + 1)) / static_cast<size_t>(num_sgs);
    auto* dst_int4 = reinterpret_cast<int4_t*>(dst);
    auto* src_int4 = reinterpret_cast<const int4_t*>(src);
    UNROLLED_WARP_COPY(
        5, lane_id, static_cast<int>(end - start), dst_int4 + start, src_int4 + start, ld_nc_global_v, st_na_global_v);
  }
  if (tail_bytes > 0 && sg_id == num_sgs - 1 && lane_id == 0) {
    const size_t base = num_int4 * sizeof(int4_t);
    for (size_t b = 0; b < tail_bytes; ++b) {
      dst[base + b] = src[base + b];
    }
  }
}

// Single-thread vectorized copy: the calling thread alone copies the ENTIRE
// [src, src+n) range using 128-bit (int4) loads/stores where possible. Use
// this whenever each thread/lane is responsible for its own distinct
// item/token (e.g. the per-token copies in the sender kernel below).
inline void copy_bytes_vectorized(uint8_t* dst, const uint8_t* src, size_t n) {
  if (n == 0) {
    return;
  }
  const size_t num_int4 = n / sizeof(int4_t);
  const size_t tail_bytes = n % sizeof(int4_t);
  if (num_int4 > 0) {
    auto* dst_int4 = reinterpret_cast<int4_t*>(dst);
    auto* src_int4 = reinterpret_cast<const int4_t*>(src);
    for (size_t i = 0; i < num_int4; ++i) {
      st_na_global_v(dst_int4 + i, ld_nc_global_v(src_int4 + i));
    }
  }
  if (tail_bytes > 0) {
    const size_t base = num_int4 * sizeof(int4_t);
    for (size_t b = 0; b < tail_bytes; ++b) {
      dst[base + b] = src[base + b];
    }
  }
}

// Per-token metadata carried alongside the payload: which node the token
// originated from, and (as a bitmask) which NVL lanes on the receiving node
// still want it.
struct SourceMeta {
  int32_t src_rdma_rank;
  int32_t is_token_in_nvl_rank_bits;
};

template <typename T>
constexpr T align_up_val(T value, T alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// Byte layout of a single token's payload inside the symmetric send/recv
// buffers: hidden data + optional per-token scales + SourceMeta + top-k
// indices (int64) + top-k weights (float), padded to 16 bytes.
struct PayloadLayout {
  size_t hidden_off = 0;
  size_t scales_off = 0;
  size_t src_meta_off = 0;
  size_t topk_idx_off = 0;
  size_t topk_weights_off = 0;
  size_t bytes_per_token = 0;

  PayloadLayout(int64_t hidden, size_t elem_size, int64_t num_scales, int64_t num_topk) {
    hidden_off = 0;
    scales_off = hidden_off + static_cast<size_t>(hidden) * elem_size;
    src_meta_off = scales_off + static_cast<size_t>(num_scales) * sizeof(float);
    topk_idx_off = src_meta_off + sizeof(SourceMeta);
    topk_weights_off = topk_idx_off + static_cast<size_t>(num_topk) * sizeof(int64_t);
    bytes_per_token =
        align_up_val<size_t>(topk_weights_off + static_cast<size_t>(num_topk) * sizeof(float), 16);
  }
};

struct SenderState {
  std::mutex mutex;
  bool initialized = false;
  void* symm = nullptr; // [send_count | send_data | recv_count | recv_data]
  size_t symm_bytes = 0;
};

SenderState& get_state() {
  static SenderState state;
  return state;
}

bool env_enabled(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

bool debug_enabled() {
  return env_enabled("INTERNODE_DISPATCH_RDMA_SENDER_DEBUG");
}

void debug_log(int64_t pe, const char* msg) {
  if (debug_enabled()) {
    std::cerr << "[internode_dispatch_rdma_sender pe " << pe << "] " << msg
               << std::endl;
  }
}

int env_positive_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (v != nullptr && *v != '\0') {
    const int parsed = std::atoi(v);
    if (parsed > 0) {
      return parsed;
    }
  }
  return fallback;
}

// Lazily bring up ISHMEM. Safe to co-exist with other extensions that also
// initialize ISHMEM: we only call init if nobody else has.
void ensure_ishmem_initialized(int device_index) {
  auto& state = get_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.initialized) {
    return;
  }
  int initialized = 0;
  ishmemx_query_initialized(&initialized);
  if (!initialized) {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    ishmemx_attr_t attr{};
    attr.device_idx = device_index;
    attr.gpu = true;
    attr.initialize_runtime = !mpi_initialized;
    ishmemx_init_attr(&attr);
  }
  state.initialized = true;
}

// Ensure the symmetric heap (send_count + send_data + recv_count + recv_data)
// can hold `bytes`. Collective (all PEs must call with the same size).
uint8_t* ensure_symmetric(size_t bytes) {
  constexpr size_t kMinBytes = 1 * 1024 * 1024;
  const size_t alloc = std::max(bytes, kMinBytes);
  auto& state = get_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.symm != nullptr && state.symm_bytes >= bytes) {
    return static_cast<uint8_t*>(state.symm);
  }
  if (state.symm != nullptr) {
    ishmem_barrier_all();
    ishmem_free(state.symm);
    ishmem_barrier_all();
    state.symm = nullptr;
    state.symm_bytes = 0;
  }
  state.symm = ishmem_malloc(alloc);
  TORCH_CHECK(
      state.symm != nullptr,
      "internode_dispatch_rdma_sender: ishmem_malloc failed for ",
      alloc,
      " bytes");
  state.symm_bytes = alloc;
  ishmem_barrier_all();
  return static_cast<uint8_t*>(state.symm);
}

// Faithful port of InternodeDispatchRDMASenderKernel, including DeepSymm's
// 128-bit (int4) vectorized copy paths (see copy_bytes_vectorized above),
// needed to reproduce its performance characteristics.
struct InternodeDispatchRdmaSenderKernel {
  const uint8_t* x_bytes;
  const float* x_scales_ptr; // nullable
  const int64_t* topk_idx_ptr;
  const float* topk_weights_ptr;
  const bool* is_token_in_rank_ptr; // [num_tokens, num_ranks]
  int* send_count; // [num_rdma_ranks]
  int* stage_arrive; // [num_rdma_ranks], scratch, zero-initialized per call
  int* put_arrive; // [num_rdma_ranks], scratch, zero-initialized per call
  int* ready_lock; // [num_rdma_ranks], scratch, zero-initialized per call: spinlock guarding ready_window/ready_tail
  int* ready_window; // [num_rdma_ranks], scratch, zero-initialized per call: bitmap of completed slots ahead of ready_tail
  int* ready_tail; // [num_rdma_ranks], scratch, zero-initialized per call: monotonic count of contiguous completed slots [0, cap)
  uint8_t* send_data; // [num_rdma_ranks, cap, bytes_per_token]
  uint8_t* recv_data; // this PE's node-local recv region: [num_rdma_ranks(src), cap, bytes_per_token]
  int* recv_count; // this PE's node-local recv counters: [num_rdma_ranks(src)]

  int32_t num_tokens;
  int32_t num_ranks;
  int32_t num_rdma_ranks;
  int32_t hidden;
  int32_t num_scales;
  int32_t num_topk;
  int32_t cap;
  int32_t my_rdma;
  int32_t my_nvl;
  int32_t sender_wgs;

  size_t hidden_bytes;
  size_t scales_bytes;
  size_t topk_idx_bytes;
  size_t topk_weights_bytes;
  size_t nbpt; // bytes per token
  size_t node_stride; // cap * nbpt
  size_t chunk_bytes; // max bytes per ishmemx_putmem_nbi_work_group() call
  PayloadLayout pl;
  int32_t debug_put; // when non-zero, lane 0 prints each RDMA put's size

  void operator()(sycl::nd_item<1> item) const {
    auto group = item.get_group();
    auto sg = item.get_sub_group();
    const int32_t gid = static_cast<int32_t>(item.get_group(0));
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lsize = static_cast<int32_t>(item.get_local_range(0));
    const int32_t sg_size = static_cast<int32_t>(sg.get_local_range()[0]);
    const int32_t num_sgs = lsize / sg_size;
    const int32_t sg_id = static_cast<int32_t>(sg.get_group_id()[0]);
    const int32_t lane_id = static_cast<int32_t>(sg.get_local_id()[0]);

    auto atomic_add_dev = [](int* addr, int v) {
      sycl::atomic_ref<
          int,
          sycl::memory_order::relaxed,
          sycl::memory_scope::device,
          sycl::access::address_space::global_space>
          ref(*addr);
      return ref.fetch_add(v);
    };
    // Per-token copy: one whole sub-group (32 lanes) cooperatively packs a
    // single token's fields via the lane-strided vectorized copy, instead of
    // one lane alone serially copying the entire ~16KB token. This lets the
    // hardware coalesce the 32 lanes' 128-bit accesses into wide bursts
    // against one contiguous range, rather than 32 lanes each independently
    // hammering 32 unrelated, non-contiguous token addresses.
    auto copy_bytes = [lane_id](uint8_t* dst, const uint8_t* src, size_t n) {
      copy_bytes_vectorized(dst, src, n, lane_id);
    };

    // Enables "copy while send": as soon as a token's slot has been fully
    // written into send_data, the writer marks it ready here instead of
    // waiting for every other team to finish copying too. Because slots are
    // assigned via an atomic counter, completion order across teams is not
    // sequential (a later slot can finish before an earlier one) -- so this
    // uses the same lock + 32-bit sliding-window trick as DeepEP's RDMA
    // sender (csrc/kernels/legacy/internode.cu kRDMASender) to turn
    // out-of-order slot completions into a monotonically advancing,
    // contiguous "ready_tail" count that a concurrently-running sender can
    // safely stream from.
    auto mark_slot_ready = [&](int32_t r, int32_t slot) {
      sycl::atomic_ref<
          int,
          sycl::memory_order::relaxed,
          sycl::memory_scope::device,
          sycl::access::address_space::global_space>
          lock_ref(ready_lock[r]);
      sycl::atomic_ref<
          int,
          sycl::memory_order::relaxed,
          sycl::memory_scope::device,
          sycl::access::address_space::global_space>
          tail_ref(ready_tail[r]);
      sycl::atomic_ref<
          int,
          sycl::memory_order::relaxed,
          sycl::memory_scope::device,
          sycl::access::address_space::global_space>
          window_ref(ready_window[r]);

      auto acquire = [&]() {
        int expected = 0;
        while (!lock_ref.compare_exchange_strong(
            expected, 1, sycl::memory_order::acquire, sycl::memory_order::relaxed)) {
          expected = 0;
        }
      };
      auto release = [&]() { lock_ref.store(0, sycl::memory_order::release); };

      acquire();
      int tail = tail_ref.load(sycl::memory_order::relaxed);
      uint32_t window = static_cast<uint32_t>(window_ref.load(sycl::memory_order::relaxed));
      int offset = slot - tail;
      // Not enough room in the window yet (too far ahead of the published
      // tail) -- release and retry until earlier slots free up space.
      while (offset >= 32) {
        release();
        acquire();
        tail = tail_ref.load(sycl::memory_order::relaxed);
        window = static_cast<uint32_t>(window_ref.load(sycl::memory_order::relaxed));
        offset = slot - tail;
      }
      window |= (1u << offset);
      if (offset == 0) {
        int num_ready = 0;
        while (window & 1u) {
          window >>= 1;
          ++num_ready;
        }
        // Release ordering: any sender reading ready_tail with `acquire`
        // is guaranteed to see this slot's payload writes too.
        tail_ref.store(tail + num_ready, sycl::memory_order::release);
      }
      window_ref.store(static_cast<int>(window), sycl::memory_order::relaxed);
      release();
    };
    auto load_ready_tail = [&](int32_t r) {
      sycl::atomic_ref<
          int,
          sycl::memory_order::relaxed,
          sycl::memory_scope::device,
          sycl::access::address_space::global_space>
          ref(ready_tail[r]);
      return ref.load(sycl::memory_order::acquire);
    };
    auto copy_fully_done = [&](int32_t r, int32_t nwg_col_for_rd) {
      sycl::atomic_ref<
          int,
          sycl::memory_order::relaxed,
          sycl::memory_scope::device,
          sycl::access::address_space::global_space>
          ref(stage_arrive[r]);
      return ref.load(sycl::memory_order::acquire) >= nwg_col_for_rd;
    };

    const int32_t G = (sender_wgs < num_rdma_ranks) ? sender_wgs : num_rdma_ranks;
    const int32_t col = gid % G;
    const int32_t sub = gid / G;
    const int32_t nwg_col = (sender_wgs - col + G - 1) / G;

    // DeepEP-style role split: dedicate the LAST team in each column group as
    // a pure "coordinator" (like kRDMASenderCoordinator) that never copies
    // tokens and instead spends its whole lifetime polling ready_tail and
    // streaming RDMA puts -- so sends for early rd's can start while the
    // remaining `num_copy_teams` teams are still packing later tokens,
    // instead of every team alternating between copying its own share and
    // then also having to poll/send afterwards. Falls back to the old
    // single-team-does-both-roles behavior when there is only one team.
    const bool has_coordinator = nwg_col > 1;
    const int32_t num_copy_teams = has_coordinator ? (nwg_col - 1) : nwg_col;
    const bool is_coordinator = has_coordinator && (sub == nwg_col - 1);
    // A column with a single team has no dedicated coordinator to fall back
    // on, so that lone team must both copy AND send its remote `rd`; without
    // this its staged send_data would never be RDMA-put.
    const bool is_single_team = (nwg_col == 1);
    const bool is_sender = is_single_team || is_coordinator;

    for (int32_t rd = col; rd < num_rdma_ranks; rd += G) {
      for (int32_t t = sub * num_sgs + sg_id; !is_coordinator && t < num_tokens;
           t += num_copy_teams * num_sgs) {
        const bool* in_rank_row = is_token_in_rank_ptr + static_cast<size_t>(t) * num_ranks;
        int32_t bits = 0;
        for (int32_t j = 0; j < kNumMaxNvlPeers; ++j) {
          if (in_rank_row[rd * kNumMaxNvlPeers + j]) {
            bits |= (1 << j);
          }
        }
        if (bits == 0) {
          continue;
        }
        // Only lane 0 performs the atomic slot allocation, then broadcasts
        // the result to the rest of the sub-group so all 32 lanes agree on
        // the destination address before cooperatively copying.
        int32_t slot = 0;
        if (lane_id == 0) {
          slot = atomic_add_dev(&send_count[rd], 1);
        }
        slot = sycl::group_broadcast(sg, slot, 0);
        if (slot >= cap) {
          continue;
        }
        // Local destination is written straight into this PE's recv_data (no
        // staging + second copy); remote destinations stage into send_data
        // and are RDMA-put later by the sender team.
        const bool is_local = (rd == my_rdma);
        uint8_t* data_base = is_local
            ? recv_data + static_cast<size_t>(my_rdma) * node_stride
            : send_data + static_cast<size_t>(rd) * node_stride;
        uint8_t* payload_ptr = data_base + static_cast<size_t>(slot) * nbpt;
        copy_bytes(
            payload_ptr + pl.hidden_off,
            x_bytes + static_cast<size_t>(t) * hidden_bytes,
            hidden_bytes);
        if (num_scales > 0) {
          copy_bytes(
              payload_ptr + pl.scales_off,
              reinterpret_cast<const uint8_t*>(x_scales_ptr + static_cast<size_t>(t) * num_scales),
              scales_bytes);
        }
        if (lane_id == 0) {
          auto* meta = reinterpret_cast<SourceMeta*>(payload_ptr + pl.src_meta_off);
          meta->src_rdma_rank = my_rdma;
          meta->is_token_in_nvl_rank_bits = bits;
        }
        copy_bytes(
            payload_ptr + pl.topk_idx_off,
            reinterpret_cast<const uint8_t*>(topk_idx_ptr + static_cast<size_t>(t) * num_topk),
            topk_idx_bytes);
        copy_bytes(
            payload_ptr + pl.topk_weights_off,
            reinterpret_cast<const uint8_t*>(topk_weights_ptr + static_cast<size_t>(t) * num_topk),
            topk_weights_bytes);
        // Synchronize the whole sub-group so ALL lanes have finished writing
        // this token's payload before lane 0 publishes it; without this a
        // reader that observes the publish could see a partially-written
        // payload (lane 0's release alone cannot order the other lanes'
        // writes). Only remote destinations advance ready_tail -- the sender
        // team streams from it; local destinations are published via
        // recv_count once every copy team has finished.
        sycl::group_barrier(sg);
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
        if (!is_local && lane_id == 0) {
          mark_slot_ready(rd, slot);
        }
      }
      // Global-and-local barrier so every lane's payload writes (into
      // send_data for remote, or directly into recv_data for local) are
      // visible work-group-wide before lane 0 publishes completion; the
      // subsequent release store then makes them visible device-wide.
      item.barrier(sycl::access::fence_space::global_and_local);

      // Declare this team done copying `rd` -- but, unlike before, do NOT
      // block here waiting for the OTHER teams to also finish. The
      // coordinator team's send path below streams out data as ready_tail
      // advances (driven by whichever copy team finishes a slot), so it can
      // start well before every copy team has reached this point; only the
      // own-node local-copy path still needs to wait for the full count.
      // The dedicated coordinator team (if any) never touches stage_arrive:
      // it does no copying, so it must not count towards the threshold the
      // other paths wait on. Release ordering pairs with the acquire loads in
      // copy_fully_done()/the local publish path.
      if (!is_coordinator && lid == 0) {
        sycl::atomic_ref<
            int,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>
            a(stage_arrive[rd]);
        a.fetch_add(1, sycl::memory_order::release);
      }
      item.barrier(sycl::access::fence_space::local_space);

      // Own-node local copy: single copy team (sub==0) is enough since this
      // is an on-device memcpy, not network-bound; but ALL of that team's
      // threads must cooperate to actually split the range (see
      // copy_bytes_vectorized_wg) instead of each of its 16 sub-groups
      // redundantly copying the whole thing. sub==0 is always a copy team
      // (the coordinator, if present, is the LAST team), so this is
      // unaffected by the role split.
      if (rd == my_rdma) {
        // Local tokens were written straight into recv_data during the copy
        // loop, so there is no second copy here -- a single team (sub==0)
        // just waits for every copy team to finish (acquire pairs with their
        // release on stage_arrive, making all recv_data writes visible) and
        // then publishes the final count.
        if (sub == 0 && lid == 0) {
          sycl::atomic_ref<
              int,
              sycl::memory_order::relaxed,
              sycl::memory_scope::device,
              sycl::access::address_space::global_space>
              a(stage_arrive[rd]);
          while (a.load(sycl::memory_order::acquire) < num_copy_teams) {
          }
          sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);

          int32_t count = send_count[rd];
          if (count > cap) {
            count = cap;
          }
          sycl::atomic_ref<
              int,
              sycl::memory_order::relaxed,
              sycl::memory_scope::device,
              sycl::access::address_space::global_space>
              ref(recv_count[my_rdma]);
          // Publish count+1 (0 means "not ready yet").
          ref.store(count + 1, sycl::memory_order::release);
        }
        // Other teams have nothing left to do for the own-node rd; the
        // coordinator team also skips it (own-node never goes through RDMA).
      } else if (is_sender) {
        // RDMA send to another node, overlapped with copying ("copy while
        // send"), DeepEP kRDMASenderCoordinator-style. When a column has a
        // dedicated coordinator (nwg_col > 1) this team never copies tokens
        // and spends its whole lifetime polling the shared `ready_tail`
        // (advanced by whichever of the `num_copy_teams` copy teams finishes
        // a slot), streaming out newly ready bytes as soon as they appear so
        // sends overlap the copy teams' ongoing work. When the column has a
        // single team (nwg_col == 1) that same team already finished its own
        // copy loop above, so copy_fully_done() is already true and it simply
        // streams out everything that is ready. Only lane 0 drives the
        // polling loop; the whole team then calls the collective put/quiet
        // together once lane 0 has decided how much is newly ready.
        const int32_t dst_pe = rd * kNumMaxNvlPeers + my_nvl;
        uint8_t* dst_data = recv_data + static_cast<size_t>(my_rdma) * node_stride;
        uint8_t* src_data = send_data + static_cast<size_t>(rd) * node_stride;
        int* dst_count = recv_count + my_rdma;

        const size_t team_end = static_cast<size_t>(cap) * nbpt;

        size_t sent_off = 0;
        bool done = (sent_off >= team_end);
        while (!done) {
          int decision = 0; // 0 = keep polling, 1 = more data ready, 2 = nothing left will ever arrive
          size_t ready_bytes = sent_off;
          if (lid == 0) {
            while (true) {
              const int32_t ready_slots = load_ready_tail(rd);
              const size_t rb = std::min(static_cast<size_t>(ready_slots) * nbpt, team_end);
              if (rb > sent_off) {
                decision = 1;
                ready_bytes = rb;
                break;
              }
              if (copy_fully_done(rd, num_copy_teams)) {
                decision = 2;
                break;
              }
            }
          }
          decision = sycl::group_broadcast(group, decision, 0);
          ready_bytes = sycl::group_broadcast(group, ready_bytes, 0);
          if (decision == 2) {
            done = true;
            break;
          }
          size_t off = sent_off;
          while (off < ready_bytes) {
            const size_t remain = ready_bytes - off;
            const size_t chunk = (remain < chunk_bytes) ? remain : chunk_bytes;
            // controled by INTERNODE_DISPATCH_RDMA_SENDER_DEBUG 
            if (debug_put && lid == 0) {
              sycl::ext::oneapi::experimental::printf(
                  "[internode put] my_rdma=%d rd=%d dst_pe=%d off=%lu chunk=%lu ready=%lu\n",
                  my_rdma, rd, dst_pe, static_cast<unsigned long>(off),
                  static_cast<unsigned long>(chunk), static_cast<unsigned long>(ready_bytes));
            }
            ishmemx_putmem_nbi_work_group(dst_data + off, src_data + off, chunk, dst_pe, group);
            off += chunk;
          }
          sent_off = ready_bytes;
          done = (sent_off >= team_end);
        }
        ishmemx_quiet_work_group(group);

        if (lid == 0) {
          // This is the only team ever sending for this rd, so there is no
          // "last of several teams" race to arbitrate -- by construction we
          // only just observed copy_fully_done()==true, so ready_tail[rd] is
          // already final and safe to read as the actual count.
          const int32_t count = load_ready_tail(rd);
          ishmem_int_atomic_set(dst_count, count + 1, dst_pe);
        }
      }
      // Non-sender teams handling a remote rd have nothing left to do here --
      // the sender team (coordinator, or the lone team when nwg_col == 1) owns
      // the entire send for this rd.
      item.barrier(sycl::access::fence_space::local_space);
    }
    ishmemx_quiet_work_group(group);
  }
};

// Unpacks the interleaved per-token payload slots in the symmetric recv
// region into separate, contiguous plain tensors for the caller to inspect.
// One work-item per (src node, slot).
struct UnpackRecvKernel {
  const uint8_t* recv_data; // [num_rdma_ranks, cap, bytes_per_token]
  uint8_t* recv_x_bytes; // [num_rdma_ranks, cap, hidden] (elem_size bytes/elem)
  float* recv_x_scales_ptr; // nullable, [num_rdma_ranks, cap, num_scales]
  int64_t* recv_topk_idx_ptr; // [num_rdma_ranks, cap, num_topk]
  float* recv_topk_weights_ptr; // [num_rdma_ranks, cap, num_topk]
  int32_t* recv_src_rdma_rank_ptr; // [num_rdma_ranks, cap]
  int32_t* recv_src_nvl_bits_ptr; // [num_rdma_ranks, cap]

  int32_t cap;
  int32_t num_scales;
  int32_t num_topk;
  size_t hidden_bytes;
  size_t scales_bytes;
  size_t topk_idx_bytes;
  size_t topk_weights_bytes;
  size_t nbpt;
  size_t node_stride;
  PayloadLayout pl;

  void operator()(sycl::id<1> idx) const {
    const size_t slot_global = idx[0];
    const int32_t src = static_cast<int32_t>(slot_global / cap);
    const int32_t slot = static_cast<int32_t>(slot_global % cap);
    const uint8_t* payload_ptr =
        recv_data + static_cast<size_t>(src) * node_stride + static_cast<size_t>(slot) * nbpt;

    uint8_t* dst_hidden = recv_x_bytes + slot_global * hidden_bytes;
    for (size_t b = 0; b < hidden_bytes; ++b) {
      dst_hidden[b] = payload_ptr[pl.hidden_off + b];
    }
    if (num_scales > 0 && recv_x_scales_ptr != nullptr) {
      auto* dst_scales = reinterpret_cast<uint8_t*>(recv_x_scales_ptr) + slot_global * scales_bytes;
      for (size_t b = 0; b < scales_bytes; ++b) {
        dst_scales[b] = payload_ptr[pl.scales_off + b];
      }
    }
    const auto* meta = reinterpret_cast<const SourceMeta*>(payload_ptr + pl.src_meta_off);
    recv_src_rdma_rank_ptr[slot_global] = meta->src_rdma_rank;
    recv_src_nvl_bits_ptr[slot_global] = meta->is_token_in_nvl_rank_bits;

    auto* dst_topk_idx = reinterpret_cast<uint8_t*>(recv_topk_idx_ptr) + slot_global * topk_idx_bytes;
    for (size_t b = 0; b < topk_idx_bytes; ++b) {
      dst_topk_idx[b] = payload_ptr[pl.topk_idx_off + b];
    }
    auto* dst_topk_w = reinterpret_cast<uint8_t*>(recv_topk_weights_ptr) + slot_global * topk_weights_bytes;
    for (size_t b = 0; b < topk_weights_bytes; ++b) {
      dst_topk_w[b] = payload_ptr[pl.topk_weights_off + b];
    }
  }
};

} // namespace

// Runs InternodeDispatchRDMASenderKernel and copies this PE's node-local
// receive region out into plain (non-symmetric) output tensors so the caller
// can verify correctness across ranks.
//
// recv_x / recv_x_scales / recv_topk_idx / recv_topk_weights / recv_src_rdma_rank
// / recv_src_nvl_bits are all shaped [num_rdma_ranks, cap, ...] and indexed by
// the SOURCE node id; recv_counts[src] is how many tokens `src` staged for
// this PE's node (capped at `cap`).
at::Tensor internode_dispatch_rdma_sender(
    const at::Tensor& x,
    const std::optional<at::Tensor>& x_scales,
    const at::Tensor& topk_idx,
    const at::Tensor& topk_weights,
    const at::Tensor& is_token_in_rank,
    at::Tensor recv_x,
    const std::optional<at::Tensor>& recv_x_scales,
    at::Tensor recv_topk_idx,
    at::Tensor recv_topk_weights,
    at::Tensor recv_src_rdma_rank,
    at::Tensor recv_src_nvl_bits,
    at::Tensor recv_counts,
    int64_t rank,
    int64_t num_ranks,
    int64_t num_max_tokens_per_rank,
    int64_t num_sender_wgs) {
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "internode_dispatch_rdma_sender: `x` must be 2D contiguous");
  TORCH_CHECK(
      topk_idx.scalar_type() == at::kLong, "internode_dispatch_rdma_sender: `topk_idx` must be int64");
  TORCH_CHECK(
      topk_weights.scalar_type() == at::kFloat,
      "internode_dispatch_rdma_sender: `topk_weights` must be float32");
  TORCH_CHECK(
      is_token_in_rank.scalar_type() == at::kBool,
      "internode_dispatch_rdma_sender: `is_token_in_rank` must be bool");
  TORCH_CHECK(
      num_ranks % kNumMaxNvlPeers == 0,
      "internode_dispatch_rdma_sender: num_ranks must be a multiple of ",
      kNumMaxNvlPeers);
  TORCH_CHECK(x.device().is_xpu(), "internode_dispatch_rdma_sender: tensors must be on XPU");
  TORCH_CHECK(
      is_token_in_rank.size(0) == x.size(0) && is_token_in_rank.size(1) == num_ranks,
      "internode_dispatch_rdma_sender: `is_token_in_rank` must be [num_tokens, num_ranks]");
  TORCH_CHECK(
      rank >= 0 && rank < num_ranks, "internode_dispatch_rdma_sender: rank must be in [0, num_ranks)");

  c10::Device device(c10::DeviceType::XPU, x.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();
  ensure_ishmem_initialized(x.device().index());
  TORCH_CHECK(
      ishmem_my_pe() == rank, "internode_dispatch_rdma_sender: ISHMEM PE does not match rank");
  TORCH_CHECK(
      ishmem_n_pes() == num_ranks,
      "internode_dispatch_rdma_sender: ISHMEM PE count does not match num_ranks");

  const int64_t num_tokens = x.size(0);
  const int64_t hidden = x.size(1);
  const size_t elem_size = x.element_size();
  const int64_t num_topk = topk_idx.size(1);
  const int64_t cap = num_max_tokens_per_rank;
  TORCH_CHECK(cap > 0, "internode_dispatch_rdma_sender: num_max_tokens_per_rank must be > 0");
  const int64_t num_rdma_ranks = num_ranks / kNumMaxNvlPeers;
  const int64_t my_rdma = rank / kNumMaxNvlPeers;
  const int64_t my_nvl = rank % kNumMaxNvlPeers;

  int64_t num_scales = 0;
  const float* x_scales_ptr = nullptr;
  if (x_scales.has_value()) {
    TORCH_CHECK(
        x_scales->dim() == 2 && x_scales->size(0) == num_tokens,
        "internode_dispatch_rdma_sender: bad `x_scales` shape");
    TORCH_CHECK(
        recv_x_scales.has_value(),
        "internode_dispatch_rdma_sender: `recv_x_scales` must be provided when `x_scales` is set");
    num_scales = x_scales->size(1);
    x_scales_ptr = static_cast<const float*>(x_scales->data_ptr());
  }

  const PayloadLayout pl(hidden, elem_size, num_scales, num_topk);
  const size_t nbpt = pl.bytes_per_token;
  const size_t node_stride = static_cast<size_t>(cap) * nbpt;

  TORCH_CHECK(
      recv_x.sizes() == at::IntArrayRef({num_rdma_ranks, cap, hidden}),
      "internode_dispatch_rdma_sender: recv_x must be [num_rdma_ranks, cap, hidden]");
  TORCH_CHECK(
      recv_topk_idx.sizes() == at::IntArrayRef({num_rdma_ranks, cap, num_topk}),
      "internode_dispatch_rdma_sender: recv_topk_idx must be [num_rdma_ranks, cap, num_topk]");
  TORCH_CHECK(
      recv_topk_weights.sizes() == at::IntArrayRef({num_rdma_ranks, cap, num_topk}),
      "internode_dispatch_rdma_sender: recv_topk_weights must be [num_rdma_ranks, cap, num_topk]");
  TORCH_CHECK(
      recv_src_rdma_rank.numel() == num_rdma_ranks * cap && recv_src_nvl_bits.numel() == num_rdma_ranks * cap,
      "internode_dispatch_rdma_sender: recv_src_* must have num_rdma_ranks*cap elements");
  TORCH_CHECK(
      recv_counts.scalar_type() == at::kLong && recv_counts.numel() == num_rdma_ranks,
      "internode_dispatch_rdma_sender: recv_counts must be int64 [num_rdma_ranks]");

  // Symmetric heap layout: [send_count | send_data | recv_count | recv_data],
  // each region sized for num_rdma_ranks entries.
  const size_t send_count_bytes = align_up_val<size_t>(static_cast<size_t>(num_rdma_ranks) * sizeof(int), 128);
  const size_t send_data_bytes = static_cast<size_t>(num_rdma_ranks) * node_stride;
  const size_t recv_count_bytes = send_count_bytes;
  const size_t recv_data_bytes = send_data_bytes;
  const size_t total_bytes = send_count_bytes + send_data_bytes + recv_count_bytes + recv_data_bytes;

  uint8_t* symm = ensure_symmetric(total_bytes);
  auto* send_count = reinterpret_cast<int*>(symm);
  uint8_t* send_data = symm + send_count_bytes;
  auto* recv_count = reinterpret_cast<int*>(send_data + send_data_bytes);
  uint8_t* recv_data = reinterpret_cast<uint8_t*>(recv_count) + recv_count_bytes;

  // Clear the counters this PE receives into, then barrier so no peer writes
  // before every PE has cleared (send-side counters are zeroed by every PE
  // for itself, which is race-free since only the owner increments them).
  queue.memset(send_count, 0, static_cast<size_t>(num_rdma_ranks) * sizeof(int));
  queue.memset(recv_count, 0, static_cast<size_t>(num_rdma_ranks) * sizeof(int));
  queue.wait_and_throw();
  ishmem_barrier_all();

  auto stage_arrive_tensor = at::zeros({num_rdma_ranks}, at::TensorOptions().dtype(at::kInt).device(device));
  int* stage_arrive_ptr = stage_arrive_tensor.data_ptr<int>();
  auto put_arrive_tensor = at::zeros({num_rdma_ranks}, at::TensorOptions().dtype(at::kInt).device(device));
  int* put_arrive_ptr = put_arrive_tensor.data_ptr<int>();
  // Per-rd "copy while send" bookkeeping (see mark_slot_ready() in the
  // kernel): a spinlock + sliding-window bitmap that turns out-of-order
  // slot completions into a monotonically advancing ready_tail count.
  auto ready_lock_tensor = at::zeros({num_rdma_ranks}, at::TensorOptions().dtype(at::kInt).device(device));
  int* ready_lock_ptr = ready_lock_tensor.data_ptr<int>();
  auto ready_window_tensor = at::zeros({num_rdma_ranks}, at::TensorOptions().dtype(at::kInt).device(device));
  int* ready_window_ptr = ready_window_tensor.data_ptr<int>();
  auto ready_tail_tensor = at::zeros({num_rdma_ranks}, at::TensorOptions().dtype(at::kInt).device(device));
  int* ready_tail_ptr = ready_tail_tensor.data_ptr<int>();

  int sender_wgs = static_cast<int>(num_sender_wgs);
  if (sender_wgs <= 0) {
    sender_wgs = env_positive_int(
        "INTERNODE_DISPATCH_RDMA_SENDER_WGS", kSenderWgsPerNode * static_cast<int>(num_rdma_ranks));
  }
  sender_wgs = std::max(sender_wgs, 1);

  debug_log(
      rank,
      ("launching InternodeDispatchRdmaSenderKernel: sender_wgs=" + std::to_string(sender_wgs)).c_str());

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(sender_wgs) * kThreads), sycl::range<1>(kThreads)),
        InternodeDispatchRdmaSenderKernel{
            static_cast<const uint8_t*>(x.data_ptr()),
            x_scales_ptr,
            topk_idx.data_ptr<int64_t>(),
            topk_weights.data_ptr<float>(),
            is_token_in_rank.data_ptr<bool>(),
            send_count,
            stage_arrive_ptr,
            put_arrive_ptr,
            ready_lock_ptr,
            ready_window_ptr,
            ready_tail_ptr,
            send_data,
            recv_data,
            recv_count,
            static_cast<int32_t>(num_tokens),
            static_cast<int32_t>(num_ranks),
            static_cast<int32_t>(num_rdma_ranks),
            static_cast<int32_t>(hidden),
            static_cast<int32_t>(num_scales),
            static_cast<int32_t>(num_topk),
            static_cast<int32_t>(cap),
            static_cast<int32_t>(my_rdma),
            static_cast<int32_t>(my_nvl),
            sender_wgs,
            static_cast<size_t>(hidden) * elem_size,
            static_cast<size_t>(num_scales) * sizeof(float),
            static_cast<size_t>(num_topk) * sizeof(int64_t),
            static_cast<size_t>(num_topk) * sizeof(float),
            nbpt,
            node_stride,
            put_chunk_bytes(),
            pl,
            debug_enabled() ? 1 : 0});
  });
  queue.wait_and_throw();

  // Every PE must have finished sending/local-copying before any PE reads its
  // recv region, since recv writes can arrive asynchronously via RDMA.
  ishmem_barrier_all();

  // Copy this PE's received region out into plain tensors for verification.
  auto recv_count_i32 = at::empty({num_rdma_ranks}, at::TensorOptions().dtype(at::kInt).device(device));
  queue.memcpy(
      recv_count_i32.data_ptr<int>(), recv_count, static_cast<size_t>(num_rdma_ranks) * sizeof(int));
  queue.wait_and_throw();
  // Publish flag is (count + 1); 0 means "never sent to" (should not happen
  // once every node always signals, but this kernel intentionally skips
  // signalling nodes that have no bits set for ANY token in this call).
  recv_counts.copy_((recv_count_i32.clamp_min(1) - 1).to(at::kLong));

  auto* recv_x_bytes = static_cast<uint8_t*>(recv_x.data_ptr());
  auto* recv_topk_idx_ptr = recv_topk_idx.data_ptr<int64_t>();
  auto* recv_topk_weights_ptr = recv_topk_weights.data_ptr<float>();
  auto* recv_src_rdma_rank_ptr = recv_src_rdma_rank.data_ptr<int32_t>();
  auto* recv_src_nvl_bits_ptr = recv_src_nvl_bits.data_ptr<int32_t>();
  float* recv_x_scales_ptr =
      recv_x_scales.has_value() ? static_cast<float*>(recv_x_scales->data_ptr()) : nullptr;

  const size_t total_slots = static_cast<size_t>(num_rdma_ranks) * static_cast<size_t>(cap);
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::range<1>(total_slots),
        UnpackRecvKernel{
            recv_data,
            recv_x_bytes,
            recv_x_scales_ptr,
            recv_topk_idx_ptr,
            recv_topk_weights_ptr,
            recv_src_rdma_rank_ptr,
            recv_src_nvl_bits_ptr,
            static_cast<int32_t>(cap),
            static_cast<int32_t>(num_scales),
            static_cast<int32_t>(num_topk),
            static_cast<size_t>(hidden) * elem_size,
            static_cast<size_t>(num_scales) * sizeof(float),
            static_cast<size_t>(num_topk) * sizeof(int64_t),
            static_cast<size_t>(num_topk) * sizeof(float),
            nbpt,
            node_stride,
            pl});
  });
  queue.wait_and_throw();

  return recv_x;
}

void internode_dispatch_rdma_sender_finalize(const at::Tensor&) {
  int64_t pe = -1;
  {
    int initialized = 0;
    ishmemx_query_initialized(&initialized);
    if (initialized) {
      pe = ishmem_my_pe();
    }
  }
  debug_log(pe, "finalize: enter");
  auto& state = get_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.symm != nullptr) {
    ishmem_barrier_all();
    ishmem_free(state.symm);
    state.symm = nullptr;
    state.symm_bytes = 0;
  }
  if (state.initialized) {
    int initialized = 0;
    ishmemx_query_initialized(&initialized);
    if (initialized) {
      debug_log(pe, "finalize: calling ishmem_finalize()");
      ishmem_finalize();
    }
    state.initialized = false;
  }
  debug_log(pe, "finalize: exit");
}

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "internode_dispatch_rdma_sender(Tensor x, Tensor? x_scales, Tensor topk_idx, Tensor topk_weights, "
      "Tensor is_token_in_rank, Tensor(a!) recv_x, Tensor? recv_x_scales, Tensor(b!) recv_topk_idx, "
      "Tensor(c!) recv_topk_weights, Tensor(d!) recv_src_rdma_rank, Tensor(e!) recv_src_nvl_bits, "
      "Tensor(f!) recv_counts, int rank, int num_ranks, int num_max_tokens_per_rank, int num_sender_wgs) -> Tensor(a!)");
  m.def("internode_dispatch_rdma_sender_finalize(Tensor dummy) -> ()");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("internode_dispatch_rdma_sender", internode_dispatch_rdma_sender);
  m.impl("internode_dispatch_rdma_sender_finalize", internode_dispatch_rdma_sender_finalize);
}
