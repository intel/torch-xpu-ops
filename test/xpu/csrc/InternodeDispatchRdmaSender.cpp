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
// Work is decomposed DeepEP-style into `num_channels` CHANNELS, one channel
// per work-group. A channel owns a contiguous slice of this rank's token
// stream (`get_channel_task_range`), packs those tokens per destination node
// into its OWN staging sub-region, and RDMA-streams each destination's region
// on a channel-pinned QP (qp = channel; the host requires
// ISHMEM_IBGDA_QPS_PER_PE >= num_channels so channels never share a QP) -- so
// the same destination node is fed by `num_channels` concurrent QP streams
// instead of a single one. The destination's recv region is partitioned by (src node,
// channel); the host unpack pass concatenates a source's channels back into
// one contiguous per-source range (order within a source is irrelevant here).
//
// Only ISHMEM APIs are used for the inter-node hop:
//   - ishmemx_putmem_nbi_work_group_qp   (WG-cooperative RDMA write on a QP)
//   - ishmemx_fence_work_group_qp        (order data before the count flag)
//   - ishmemx_uint64_atomic_set_nbi_qp   (leader publishes the count on same QP)
//   - ishmemx_quiet_work_group           (drain outstanding puts)
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

// Max bytes issued per ishmemx_putmem_nbi_work_group_qp() call. Bounds how
// much of a destination's already-ready range the coordinator WG puts in a
// single ISHMEM call; the send path streams chunks as they become ready and
// quiets once at the end. Overridable via env for perf experimentation.
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
// per-token vectorized paths below assume 32-lane sub-groups cooperating
// within a work-group of this size, so this must stay aligned with the
// DeepSymm kernel to reproduce its performance characteristics.
constexpr int32_t kThreads = 512;
// Default number of channels (each channel == one work-group) when the caller
// passes num_channels <= 0 and no env override is set.
constexpr int kDefaultChannels = 8;

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

// DeepEP kRDMASender / kRDMASenderCoordinator split: each channel is a PAIR of
// work-groups (channel == gid / 2). The even WG (role 0) COPIES this channel's
// tokens into per-(channel, dst node) staging and advances a sliding-window
// ready_tail; the odd WG (role 1) is the COORDINATOR that streams each dst
// node's ready bytes over the channel-pinned QP and publishes the count. The
// two WGs of a channel must run concurrently (the coordinator spins on the
// copy WG's ready_tail / copy_done), same residency requirement as DeepEP.
struct InternodeDispatchRdmaSenderKernel {
  const uint8_t* x_bytes;
  const float* x_scales_ptr; // nullable
  const int64_t* topk_idx_ptr;
  const float* topk_weights_ptr;
  const bool* is_token_in_rank_ptr; // [num_tokens, num_ranks]
  int* send_count; // [num_channels, num_rdma_ranks], per-channel slot allocator (scratch)
  int* ready_lock; // [num_channels, num_rdma_ranks], scratch: spinlock guarding ready_window/ready_tail
  int* ready_window; // [num_channels, num_rdma_ranks], scratch: bitmap of completed slots ahead of ready_tail
  int* ready_tail; // [num_channels, num_rdma_ranks], scratch: contiguous completed slot count
  int* copy_done; // [num_channels], scratch: set by the copy WG when its token scan finishes
  uint8_t* send_data; // [num_rdma_ranks(dst), num_channels, cap_ch, bytes_per_token]
  uint8_t* recv_data; // this PE's recv region: [num_rdma_ranks(src), num_channels, cap_ch, bytes_per_token]
  uint64_t* recv_count; // [num_rdma_ranks(src), num_channels], published as (count + 1)

  int32_t num_tokens;
  int32_t num_ranks;
  int32_t num_rdma_ranks;
  int32_t num_channels;
  int32_t hidden;
  int32_t num_scales;
  int32_t num_topk;
  int32_t cap_ch;
  int32_t my_rdma;
  int32_t my_nvl;
  int32_t num_max_nvl_peers;

  size_t hidden_bytes;
  size_t scales_bytes;
  size_t topk_idx_bytes;
  size_t topk_weights_bytes;
  size_t nbpt; // bytes per token
  size_t channel_stride; // cap_ch * nbpt
  size_t chunk_bytes; // max bytes per ishmemx_putmem_nbi_work_group_qp() call
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

    const int32_t channel = gid / 2;
    const int32_t role = gid % 2; // 0 = copy WG, 1 = coordinator WG
    if (channel >= num_channels) {
      return;
    }

    // One whole sub-group (32 lanes) cooperatively packs a single token.
    auto copy_bytes = [lane_id](uint8_t* dst, const uint8_t* src, size_t n) {
      copy_bytes_vectorized(dst, src, n, lane_id);
    };

    // Sliding-window release turning out-of-order slot completions into a
    // monotonically advancing contiguous ready_tail[cr] the coordinator can
    // stream from (DeepEP kRDMASender rdma_send_channel_{lock,tail,window}).
    auto mark_slot_ready = [&](int32_t cr, int32_t slot) {
      sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                       sycl::access::address_space::global_space> lock_ref(ready_lock[cr]);
      sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                       sycl::access::address_space::global_space> tail_ref(ready_tail[cr]);
      sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                       sycl::access::address_space::global_space> window_ref(ready_window[cr]);
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
        tail_ref.store(tail + num_ready, sycl::memory_order::release);
      }
      window_ref.store(static_cast<int>(window), sycl::memory_order::relaxed);
      release();
    };
    auto load_ready_tail = [&](int32_t cr) {
      sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                       sycl::access::address_space::global_space> ref(ready_tail[cr]);
      return ref.load(sycl::memory_order::acquire);
    };
    auto load_copy_done = [&]() {
      sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                       sycl::access::address_space::global_space> ref(copy_done[channel]);
      return ref.load(sycl::memory_order::acquire) != 0;
    };

    // This channel owns the contiguous token range [ts, te) (DeepEP
    // get_channel_task_range): num_channels channels evenly split the stream.
    const int32_t n_per = (num_tokens + num_channels - 1) / num_channels;
    const int32_t ts = sycl::min(n_per * channel, num_tokens);
    const int32_t te = sycl::min(ts + n_per, num_tokens);

    if (role == 0) {
      // ===================== COPY WG (kRDMASender) =====================
      // Each sub-group takes whole tokens; for every dst node the token routes
      // to, claim a slot, copy the payload, then mark the slot ready so the
      // coordinator WG can stream it while we keep packing later tokens.
      for (int32_t t = ts + sg_id; t < te; t += num_sgs) {
        const bool* in_rank_row = is_token_in_rank_ptr + static_cast<size_t>(t) * num_ranks;
        for (int32_t rd = 0; rd < num_rdma_ranks; ++rd) {
          int32_t bits = 0;
          for (int32_t j = 0; j < num_max_nvl_peers; ++j) {
            if (in_rank_row[rd * num_max_nvl_peers + j]) {
              bits |= (1 << j);
            }
          }
          if (bits == 0) {
            continue;
          }
          const int32_t cr = channel * num_rdma_ranks + rd;
          int32_t slot = 0;
          if (lane_id == 0) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space> ref(send_count[cr]);
            slot = ref.fetch_add(1);
          }
          slot = sycl::group_broadcast(sg, slot, 0);
          if (slot >= cap_ch) {
            continue;
          }
          const bool is_local = (rd == my_rdma);
          uint8_t* base = is_local
              ? recv_data + (static_cast<size_t>(my_rdma) * num_channels + channel) * channel_stride
              : send_data + (static_cast<size_t>(rd) * num_channels + channel) * channel_stride;
          uint8_t* payload_ptr = base + static_cast<size_t>(slot) * nbpt;
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
          sycl::group_barrier(sg);
          sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
          if (lane_id == 0) {
            mark_slot_ready(cr, slot);
          }
        }
      }
      // Publish "this channel's copy scan is complete" so the coordinator can
      // drain the final ready_tail and stop.
      item.barrier(sycl::access::fence_space::global_and_local);
      if (lid == 0) {
        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space> d(copy_done[channel]);
        d.store(1, sycl::memory_order::release);
      }
      return;
    }

    // ============== COORDINATOR WG (kRDMASenderCoordinator) ==============
    // Stream every dst node's ready bytes on the channel-pinned QP, overlapping
    // the copy WG's ongoing packing, then publish the per-(src, channel) count.
    // Visit remote nodes first (rotated to start right after `my_rdma`) and
    // handle the local `rd == my_rdma` branch last. The local branch busy-waits
    // on `copy_done`, which only flips once the copy WG's *entire* token scan
    // finishes; visiting it first (plain rd = 0..num_rdma_ranks-1 order) makes
    // whichever rank has my_rdma == 0 stall before issuing any RDMA puts, while
    // every other rank starts streaming remote data immediately and overlaps
    // that wait with the copy WG's ongoing packing. That asymmetry -- not the
    // NIC/PCIe topology -- was the source of the ~1.7x bandwidth gap between
    // rank0 (my_rdma == 0) and rank1 in the 2-rank test.
    for (int32_t k = 0; k < num_rdma_ranks; ++k) {
      const int32_t rd = (my_rdma + 1 + k) % num_rdma_ranks;
      const int32_t cr = channel * num_rdma_ranks + rd;
      const size_t recv_idx = static_cast<size_t>(my_rdma) * num_channels + channel;
      if (rd == my_rdma) {
        // Self-node payloads were written straight into recv_data by the copy
        // WG; wait for its scan to finish, then publish the local count.
        if (lid == 0) {
          while (!load_copy_done()) {
          }
          sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
          int32_t count = load_ready_tail(cr);
          if (count > cap_ch) {
            count = cap_ch;
          }
          sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                           sycl::access::address_space::global_space> ref(recv_count[recv_idx]);
          ref.store(static_cast<uint64_t>(count) + 1, sycl::memory_order::release);
        }
        item.barrier(sycl::access::fence_space::local_space);
        continue;
      }
      const int32_t dst_pe = rd * num_max_nvl_peers + my_nvl;
      // DeepEP-style 1:1 channel -> QP (no modulo); host guarantees enough QPs.
      const unsigned int qp = static_cast<unsigned int>(channel);
      uint8_t* dst_data = recv_data + recv_idx * channel_stride;
      uint8_t* src_data =
          send_data + (static_cast<size_t>(rd) * num_channels + channel) * channel_stride;
      const size_t team_end = static_cast<size_t>(cap_ch) * nbpt;
      size_t sent_off = 0;
      bool done = (sent_off >= team_end);
      while (!done) {
        int decision = 0; // 0 = keep polling, 1 = more ready, 2 = copy done + drained
        size_t ready_bytes = sent_off;
        if (lid == 0) {
          while (true) {
            const int32_t ready_slots = load_ready_tail(cr);
            const size_t rb = std::min(static_cast<size_t>(ready_slots) * nbpt, team_end);
            if (rb > sent_off) {
              decision = 1;
              ready_bytes = rb;
              break;
            }
            if (load_copy_done()) {
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
          if (debug_put && lid == 0) {
            sycl::ext::oneapi::experimental::printf(
                "[internode put] my_rdma=%d ch=%d rd=%d dst_pe=%d qp=%u off=%lu chunk=%lu ready=%lu\n",
                my_rdma, channel, rd, dst_pe, qp, static_cast<unsigned long>(off),
                static_cast<unsigned long>(chunk), static_cast<unsigned long>(ready_bytes));
          }
          ishmemx_putmem_nbi_work_group_qp(dst_data + off, src_data + off, chunk, dst_pe, group, qp);
          off += chunk;
        }
        sent_off = ready_bytes;
        done = (sent_off >= team_end);
      }
      // Order the data puts ahead of the count flag on this QP, then publish.
      ishmemx_fence_work_group_qp(dst_pe, group, qp);
      if (lid == 0) {
        int32_t count = load_ready_tail(cr);
        if (count > cap_ch) {
          count = cap_ch;
        }
        if (count > 0) {
          ishmemx_uint64_atomic_set_nbi_qp(
              recv_count + recv_idx, static_cast<uint64_t>(count) + 1, dst_pe, qp);
        }
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    ishmemx_quiet_work_group(group);
  }
};

// Unpacks the per-(src node, channel) payload slots in the symmetric recv
// region into contiguous plain tensors, concatenating a source's channels
// back into one per-source range. One work-item per (src, channel, slot).
struct UnpackRecvKernel {
  const uint8_t* recv_data; // [num_rdma_ranks, num_channels, cap_ch, bytes_per_token]
  uint8_t* recv_x_bytes; // [num_rdma_ranks, cap, hidden] (elem_size bytes/elem)
  float* recv_x_scales_ptr; // nullable, [num_rdma_ranks, cap, num_scales]
  int64_t* recv_topk_idx_ptr; // [num_rdma_ranks, cap, num_topk]
  float* recv_topk_weights_ptr; // [num_rdma_ranks, cap, num_topk]
  int32_t* recv_src_rdma_rank_ptr; // [num_rdma_ranks, cap]
  int32_t* recv_src_nvl_bits_ptr; // [num_rdma_ranks, cap]
  // [num_rdma_ranks, num_channels]: per (src, channel) count published as
  // (count + 1) by the sender, 0 meaning "never signalled" (treated as 0).
  const uint64_t* recv_count;
  // [num_rdma_ranks, num_channels] exclusive prefix within each source: the
  // first per-source output slot this (src, channel) region maps to.
  const int32_t* channel_prefix;

  int32_t cap;
  int32_t cap_ch;
  int32_t num_channels;
  int32_t num_scales;
  int32_t num_topk;
  size_t hidden_bytes;
  size_t scales_bytes;
  size_t topk_idx_bytes;
  size_t topk_weights_bytes;
  size_t nbpt;
  size_t channel_stride;
  PayloadLayout pl;

  void operator()(sycl::id<1> idx) const {
    const size_t g = idx[0];
    const int32_t slot = static_cast<int32_t>(g % cap_ch);
    const int32_t ch = static_cast<int32_t>((g / cap_ch) % num_channels);
    const int32_t src =
        static_cast<int32_t>(g / (static_cast<size_t>(cap_ch) * num_channels));
    const size_t ci = static_cast<size_t>(src) * num_channels + ch;
    const uint64_t published = recv_count[ci];
    const int32_t count = published > 0 ? static_cast<int32_t>(published - 1) : 0;
    if (slot >= count) {
      // No token in this slot -- leave the caller's output tensors untouched.
      return;
    }
    const int32_t global_slot = channel_prefix[ci] + slot;
    if (global_slot >= cap) {
      return;
    }
    const size_t slot_global = static_cast<size_t>(src) * cap + global_slot;
    const uint8_t* payload_ptr =
        recv_data + ci * channel_stride + static_cast<size_t>(slot) * nbpt;

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
    int64_t num_channels_arg) {
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "internode_dispatch_rdma_sender: `x` must be 2D contiguous");
  TORCH_CHECK(
      topk_idx.scalar_type() == at::kLong, "internode_dispatch_rdma_sender: `topk_idx` must be int64");
  TORCH_CHECK(
      topk_weights.scalar_type() == at::kFloat,
      "internode_dispatch_rdma_sender: `topk_weights` must be float32");
  TORCH_CHECK(
      is_token_in_rank.scalar_type() == at::kBool,
      "internode_dispatch_rdma_sender: `is_token_in_rank` must be bool");
  // Overridable so a 2-rank run can exercise the actual RDMA put path (with
  // the compiled-in default of 2, world_size=2 puts both ranks on the same
  // single "node" -- num_rdma_ranks=1 -- so the kernel only ever takes the
  // own-node local-copy branch and no ishmemx_putmem_nbi_work_group_qp is
  // ever issued). Set INTERNODE_DISPATCH_NUM_MAX_NVL_PEERS=1 to make every
  // rank its own node (num_rdma_ranks == num_ranks), forcing real RDMA puts
  // between all pairs, including with just 2 ranks.
  const int num_max_nvl_peers =
      env_positive_int("INTERNODE_DISPATCH_NUM_MAX_NVL_PEERS", kNumMaxNvlPeers);
  TORCH_CHECK(
      num_ranks % num_max_nvl_peers == 0,
      "internode_dispatch_rdma_sender: num_ranks must be a multiple of ",
      num_max_nvl_peers);
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
  const int64_t num_rdma_ranks = num_ranks / num_max_nvl_peers;
  const int64_t my_rdma = rank / num_max_nvl_peers;
  const int64_t my_nvl = rank % num_max_nvl_peers;

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

  // Channels: each channel is a PAIR of work-groups (copy + coordinator).
  int num_channels_i = static_cast<int>(num_channels_arg);
  if (num_channels_i <= 0) {
    num_channels_i = env_positive_int("INTERNODE_DISPATCH_RDMA_SENDER_CHANNELS", kDefaultChannels);
  }
  num_channels_i = std::max(num_channels_i, 1);
  if (num_tokens > 0 && static_cast<int64_t>(num_channels_i) > num_tokens) {
    num_channels_i = static_cast<int>(num_tokens);
  }
  const int64_t num_channels = num_channels_i;
  const int qps_per_pe = std::max(env_positive_int("ISHMEM_IBGDA_QPS_PER_PE", 1), 1);
  // DeepEP pins one QP per channel (qp = channel, no modulo); require ISHMEM to
  // have provisioned at least num_channels QPs so distinct channels never
  // collide on the same QP.
  TORCH_CHECK(
      static_cast<int64_t>(qps_per_pe) >= num_channels,
      "internode_dispatch_rdma_sender: ISHMEM_IBGDA_QPS_PER_PE (", qps_per_pe,
      ") must be >= num_channels (", num_channels,
      "). Set ISHMEM_IBGDA_QPS_PER_PE=", num_channels, " or higher.");

  // A channel handles at most ceil(num_tokens / num_channels) tokens, so it can
  // never route more than that many to any single node -- size each channel's
  // per-node staging region accordingly (capped by the receiver's cap).
  const int64_t n_ch = (num_tokens + num_channels - 1) / num_channels;
  const int64_t cap_ch = std::min<int64_t>(cap, std::max<int64_t>(n_ch, 1));
  const size_t channel_stride = static_cast<size_t>(cap_ch) * nbpt;

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

  // Symmetric heap layout: [recv_count | send_data | recv_data]. send_data and
  // recv_data are [num_rdma_ranks(peer), num_channels, cap_ch, bytes_per_token];
  // recv_count is [num_rdma_ranks(src), num_channels] published as (count + 1).
  const size_t count_slots = static_cast<size_t>(num_rdma_ranks) * static_cast<size_t>(num_channels);
  const size_t recv_count_bytes = align_up_val<size_t>(count_slots * sizeof(uint64_t), 128);
  const size_t region_bytes = count_slots * channel_stride;
  const size_t total_bytes = recv_count_bytes + region_bytes + region_bytes;

  uint8_t* symm = ensure_symmetric(total_bytes);
  auto* recv_count = reinterpret_cast<uint64_t*>(symm);
  uint8_t* send_data = symm + recv_count_bytes;
  uint8_t* recv_data = send_data + region_bytes;

  // Clear the per-(src, channel) counters this PE receives into, then barrier
  // so no peer writes before every PE has cleared.
  queue.memset(recv_count, 0, count_slots * sizeof(uint64_t));
  queue.wait_and_throw();
  ishmem_barrier_all();

  // Per-(channel, dst node) sender scratch (device only): slot allocator plus
  // the sliding-window ready_tail bookkeeping the coordinator WG streams from.
  const int64_t cr_slots = num_channels * num_rdma_ranks;
  auto int_opts = at::TensorOptions().dtype(at::kInt).device(device);
  auto send_count_tensor = at::zeros({cr_slots}, int_opts);
  auto ready_lock_tensor = at::zeros({cr_slots}, int_opts);
  auto ready_window_tensor = at::zeros({cr_slots}, int_opts);
  auto ready_tail_tensor = at::zeros({cr_slots}, int_opts);
  auto copy_done_tensor = at::zeros({num_channels}, int_opts);

  // Each channel == 2 concurrent WGs (copy + coordinator); the coordinator
  // spins on the copy WG, so both must be resident at once.
  const int64_t total_wgs = num_channels * 2;
  {
    const int64_t max_cu = static_cast<int64_t>(
        queue.get_device().get_info<sycl::info::device::max_compute_units>());
    if (total_wgs > max_cu) {
      TORCH_WARN(
          "internode_dispatch_rdma_sender: channel WGs (", total_wgs,
          " = 2 * num_channels) exceed compute units (", max_cu,
          "); the copy/coordinator WGs may not co-reside and could hang. "
          "Reduce num_channels.");
    }
  }

  debug_log(
      rank,
      ("launching InternodeDispatchRdmaSenderKernel: num_channels=" + std::to_string(num_channels)).c_str());

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(total_wgs) * kThreads), sycl::range<1>(kThreads)),
        InternodeDispatchRdmaSenderKernel{
            static_cast<const uint8_t*>(x.data_ptr()),
            x_scales_ptr,
            topk_idx.data_ptr<int64_t>(),
            topk_weights.data_ptr<float>(),
            is_token_in_rank.data_ptr<bool>(),
            send_count_tensor.data_ptr<int>(),
            ready_lock_tensor.data_ptr<int>(),
            ready_window_tensor.data_ptr<int>(),
            ready_tail_tensor.data_ptr<int>(),
            copy_done_tensor.data_ptr<int>(),
            send_data,
            recv_data,
            recv_count,
            static_cast<int32_t>(num_tokens),
            static_cast<int32_t>(num_ranks),
            static_cast<int32_t>(num_rdma_ranks),
            static_cast<int32_t>(num_channels),
            static_cast<int32_t>(hidden),
            static_cast<int32_t>(num_scales),
            static_cast<int32_t>(num_topk),
            static_cast<int32_t>(cap_ch),
            static_cast<int32_t>(my_rdma),
            static_cast<int32_t>(my_nvl),
            num_max_nvl_peers,
            static_cast<size_t>(hidden) * elem_size,
            static_cast<size_t>(num_scales) * sizeof(float),
            static_cast<size_t>(num_topk) * sizeof(int64_t),
            static_cast<size_t>(num_topk) * sizeof(float),
            nbpt,
            channel_stride,
            put_chunk_bytes(),
            pl,
            debug_enabled() ? 1 : 0});
  });
  queue.wait_and_throw();

  // Every PE must have finished sending/local-copying before any PE reads its
  // recv region, since recv writes can arrive asynchronously via RDMA.
  ishmem_barrier_all();

  // Read the per-(src, channel) counts and build, per source, the exclusive
  // prefix that concatenates a source's channels into one contiguous range.
  auto recv_count_raw =
      at::empty({num_rdma_ranks, num_channels}, at::TensorOptions().dtype(at::kLong).device(device));
  queue.memcpy(recv_count_raw.data_ptr<int64_t>(), recv_count, count_slots * sizeof(uint64_t));
  queue.wait_and_throw();
  // Published flag is (count + 1); 0 means "never signalled" (treated as 0).
  auto counts2d = (recv_count_raw.clamp_min(1) - 1);
  auto channel_prefix = (counts2d.cumsum(1) - counts2d).to(at::kInt).contiguous();
  auto totals = counts2d.sum(1);
  recv_counts.copy_(totals.clamp_max(cap).to(at::kLong));

  auto* recv_x_bytes = static_cast<uint8_t*>(recv_x.data_ptr());
  auto* recv_topk_idx_ptr = recv_topk_idx.data_ptr<int64_t>();
  auto* recv_topk_weights_ptr = recv_topk_weights.data_ptr<float>();
  auto* recv_src_rdma_rank_ptr = recv_src_rdma_rank.data_ptr<int32_t>();
  auto* recv_src_nvl_bits_ptr = recv_src_nvl_bits.data_ptr<int32_t>();
  float* recv_x_scales_ptr =
      recv_x_scales.has_value() ? static_cast<float*>(recv_x_scales->data_ptr()) : nullptr;

  const size_t total_slots = static_cast<size_t>(num_rdma_ranks) *
      static_cast<size_t>(num_channels) * static_cast<size_t>(cap_ch);
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
            recv_count,
            channel_prefix.data_ptr<int32_t>(),
            static_cast<int32_t>(cap),
            static_cast<int32_t>(cap_ch),
            static_cast<int32_t>(num_channels),
            static_cast<int32_t>(num_scales),
            static_cast<int32_t>(num_topk),
            static_cast<size_t>(hidden) * elem_size,
            static_cast<size_t>(num_scales) * sizeof(float),
            static_cast<size_t>(num_topk) * sizeof(int64_t),
            static_cast<size_t>(num_topk) * sizeof(float),
            nbpt,
            channel_stride,
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
      "Tensor(f!) recv_counts, int rank, int num_ranks, int num_max_tokens_per_rank, int num_channels) -> Tensor(a!)");
  m.def("internode_dispatch_rdma_sender_finalize(Tensor dummy) -> ()");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("internode_dispatch_rdma_sender", internode_dispatch_rdma_sender);
  m.impl("internode_dispatch_rdma_sender_finalize", internode_dispatch_rdma_sender_finalize);
}
