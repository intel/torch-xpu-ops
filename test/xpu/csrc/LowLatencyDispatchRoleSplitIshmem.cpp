// LowLatencyDispatchRoleSplitIshmem.cpp
//
// Minimal, self-contained reproducer of DeepSymm's
// `LowLatencyDispatchRoleSplitKernelBK` (moe_ep/internode_ll.cpp) low-latency
// MoE dispatch kernel, built purely on ISHMEM device/host APIs (modelled after
// TokenDispatchIshmem.cpp / RingAllgatherIshmem.cpp in this same directory).
//
// The production kernel splits its work-groups (WGs) into two disjoint
// ROLES, launched together as a single kernel so there is no per-step host
// round-trip:
//   - "expert" WGs  : own exactly one global expert id. They scan this rank's
//                     local tokens, and for every token whose `topk_idx`
//                     contains their expert, atomically claim a send slot,
//                     push the token's hidden vector (+ its source token
//                     index) to the owning rank's receive buffer with an
//                     ISHMEM RDMA work-group put (or a plain local copy for
//                     self-dispatch), then -- once all of this rank's tokens
//                     have been scanned -- publish the final count and raise
//                     an ISHMEM completion flag on the destination rank.
//   - "receiver" WGs: one per LOCAL expert (i.e. an expert owned by this
//                     rank). Each waits on the completion flag from every
//                     source rank (an ISHMEM device wait for remote sources,
//                     a plain spin for the local/self source), then gathers
//                     the arrived tokens into the caller-provided
//                     `packed_recv_x` / `packed_recv_src_info` buffers and
//                     records the (count, begin) layout range DeepEP-style.
//
// This reproducer keeps that exact two-role structure and the ISHMEM RDMA +
// on-device completion signalling, but drops the production kernel's extras
// (mask buffer, fp8 casting, cumulative stats, hierarchical NVLink+RDMA
// staging) so the role-split + RDMA + spin-wait synchronization pattern can
// be exercised and perf-tested in isolation.
//
// Only ISHMEM APIs are used for cross-rank communication:
//   - ishmemx_putmem_nbi_work_group_qp       (RDMA write of one token, WG-collective)
//   - ishmemx_fence_work_group_qp            (non-blocking doorbell for ordering)
//   - ishmemx_uint64_atomic_set_nbi_qp       (NBI flag on same QP — RC ordering)
//   - ishmem_uint64_wait_until               (device-side wait on our own flag slot)
//   - ishmem_malloc / ishmem_free / ishmem_barrier_all (symmetric heap)
//
// Registered op: symm_mem::low_latency_dispatch_role_split_ishmem

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <comm/SYCLHelpers.h>
#include <ishmem.h>
#include <ishmemx.h>
#include <mpi.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>

namespace {

struct DispatchState {
  std::mutex mutex;
  bool initialized = false;
  void* symm = nullptr;       // send region + recv data + recv src region
  size_t symm_bytes = 0;
  uint64_t* pad = nullptr;    // symmetric completion flags, 2 * num_local_experts * num_ranks
  int pad_slots = 0;

  int32_t* send_count = nullptr; // local-only per-expert atomic slot counters, num_experts
  int send_count_slots = 0;
  uint64_t iteration = 0;     // strictly-increasing signal tag (never reused / never cleared)
};

DispatchState& get_state() {
  static DispatchState state;
  return state;
}

bool env_enabled(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

bool debug_enabled() {
  return env_enabled("LL_ROLE_SPLIT_ISHMEM_DEBUG");
}

void debug_log(int64_t pe, const char* msg) {
  if (debug_enabled()) {
    std::cerr << "[low_latency_dispatch_role_split_ishmem pe " << pe << "] " << msg
               << std::endl;
  }
}

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
    ishmemx_attr_t attr;
    attr.device_idx = device_index;
    attr.gpu = true;
    attr.initialize_runtime = !mpi_initialized;
    ishmemx_init_attr(&attr);
  }
  state.initialized = true;
}

// Ensure the symmetric data buffer (send region + recv data + recv src) can
// hold `bytes`. Collective (all PEs must call with the same size in order).
void ensure_symmetric(size_t bytes) {
  constexpr size_t kMinBytes = 8 * 1024 * 1024;
  const size_t alloc = std::max(bytes, kMinBytes);
  auto& state = get_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.symm_bytes >= bytes) {
    return;
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
      "low_latency_dispatch_role_split_ishmem: ishmem_malloc failed for ",
      alloc,
      " bytes");
  state.symm_bytes = alloc;
  ishmem_barrier_all();
}

void* current_symmetric() {
  auto& state = get_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.symm;
}

uint64_t* ensure_u64_array(uint64_t*& ptr, int& have_slots, int slots, sycl::queue& queue) {
  if (ptr != nullptr && have_slots >= slots) {
    return ptr;
  }
  if (ptr != nullptr) {
    ishmem_barrier_all();
    ishmem_free(ptr);
    ishmem_barrier_all();
    ptr = nullptr;
    have_slots = 0;
  }
  const size_t bytes = static_cast<size_t>(slots) * sizeof(uint64_t);
  ptr = static_cast<uint64_t*>(ishmem_malloc(bytes));
  TORCH_CHECK(
      ptr != nullptr,
      "low_latency_dispatch_role_split_ishmem: ishmem_malloc failed for ",
      bytes,
      " bytes");
  have_slots = slots;
  queue.memset(ptr, 0, bytes).wait_and_throw();
  ishmem_barrier_all();
  return ptr;
}

// Local-only (never RDMA-addressed) per-expert atomic slot counters. Plain
// device memory is fine here -- no symmetric heap required.
int32_t* ensure_send_count(int32_t*& ptr, int& have_slots, int slots, sycl::queue& queue) {
  if (ptr != nullptr && have_slots >= slots) {
    return ptr;
  }
  if (ptr != nullptr) {
    sycl::free(ptr, queue);
    ptr = nullptr;
    have_slots = 0;
  }
  ptr = sycl::malloc_device<int32_t>(slots, queue);
  TORCH_CHECK(
      ptr != nullptr,
      "low_latency_dispatch_role_split_ishmem: device alloc failed for send_count");
  have_slots = slots;
  return ptr;
}

inline uint64_t pack_range(int32_t count, int32_t begin) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(begin)) << 32) |
      static_cast<uint32_t>(count);
}

// Single-kernel ISHMEM low-latency dispatch, split into "expert" (sender) and
// "receiver" work-group roles -- see file header for the full description.
// All WGs are launched together in ONE nd_range so the receiver WGs can spin
// on completion flags signalled by the expert WGs (own-rank flags are
// written by a sibling WG in this SAME launch; remote-rank flags arrive via
// ISHMEM from another PE's kernel). Because SYCL gives no forward-progress
// guarantee across work-groups within a single launch, correctness here
// relies on every WG in the launch being able to run concurrently -- keep
// WG size small (`kWGSize`) and `total_wgs` within the device's resident
// work-group budget (the same constraint the production DeepSymm kernel
// relies on).
struct LowLatencyDispatchRoleSplitIshmemKernel {
  const at::BFloat16* send_data;  // [num_tokens, hidden], this rank's tokens (symmetric)
  at::BFloat16* recv_data;        // [num_local_experts*num_ranks*capacity, hidden] (symmetric)
  int32_t* recv_src;              // [num_local_experts*num_ranks*capacity] (symmetric)
  int32_t* send_count;            // [num_experts] local atomic slot counters
  uint64_t* pad;                  // this call's half: [num_local_experts*num_ranks]
  const int32_t* topk_idx;        // [num_tokens, num_topk]
  at::BFloat16* packed_recv_x;        // [num_local_experts*num_ranks*capacity, hidden]
  int32_t* packed_recv_src_info;      // [num_local_experts*num_ranks*capacity]
  int32_t* packed_recv_count;         // [num_local_experts]
  int64_t* packed_recv_layout_range;  // [num_local_experts*num_ranks]
  int64_t num_tokens;
  int64_t hidden;
  int64_t num_topk;
  int64_t capacity;
  int32_t num_local_experts;
  int32_t rank;
  int32_t num_ranks;
  int32_t num_expert_wgs;  // == num_experts
  uint64_t tag;

  void operator()(sycl::nd_item<1> item) const {
    const int32_t wg_id = static_cast<int32_t>(item.get_group(0));
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lsize = static_cast<int32_t>(item.get_local_range(0));
    auto grp = item.get_group();

    // ===================== EXPERT (SENDER) WG =====================
    // WG `wg_id` owns global expert id `wg_id` for wg_id < num_expert_wgs.
    if (wg_id < num_expert_wgs) {
      const int32_t expert = wg_id;
      const int32_t dst_rank = expert / num_local_experts;
      const int32_t local_expert = expert % num_local_experts;

      for (int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
        bool owns_token = false;
        for (int64_t k = 0; k < num_topk; ++k) {
          if (topk_idx[token_idx * num_topk + k] == expert) {
            owns_token = true;
            break;
          }
        }
        if (!owns_token) {
          continue;
        }

        int32_t slot = 0;
        if (lid == 0) {
          sycl::atomic_ref<
              int32_t,
              sycl::memory_order::relaxed,
              sycl::memory_scope::device,
              sycl::access::address_space::global_space>
              slot_ref(send_count[expert]);
          slot = slot_ref.fetch_add(1);
        }
        slot = sycl::group_broadcast(grp, slot, 0);
        if (slot >= capacity) {
          continue;
        }

        const int64_t dst_slot =
            (static_cast<int64_t>(local_expert) * num_ranks + rank) * capacity + slot;
        const int64_t src_off = token_idx * hidden;
        const int64_t dst_off = dst_slot * hidden;

        if (dst_rank == rank) {
          // Self-dispatch: plain local copy, no RDMA needed.
          for (int64_t h = lid; h < hidden; h += lsize) {
            recv_data[dst_off + h] = send_data[src_off + h];
          }
          if (lid == 0) {
            recv_src[dst_slot] = static_cast<int32_t>(token_idx);
          }
        } else {
          // NBI put for payload on pinned QP
          ishmemx_putmem_nbi_work_group_qp(
              static_cast<void*>(recv_data + dst_off),
              static_cast<const void*>(send_data + src_off),
              static_cast<size_t>(hidden) * sizeof(at::BFloat16),
              dst_rank,
              grp,
              static_cast<unsigned int>(local_expert));
          if (lid == 0) {
            ishmem_int_atomic_set(recv_src + dst_slot, static_cast<int32_t>(token_idx), dst_rank);
          }
        }
      }

      // ---- Completion protocol ----
      //
      // Pack count into the pad value: high 32 bits = tag, low 32 bits = count.
      // Post the flag on the SAME QP as the data put via blocking inline
      // RDMA Write.  The function internally fence_slot-spins to drain
      // pending NBI WQEs (safe with shared QPs), doorbells, and polls CQ.
      // RC SQ FIFO guarantees: data arrives before flag at the remote.
      sycl::group_barrier(grp);

      if (lid == 0) {
        const int32_t count = sycl::min(send_count[expert], static_cast<int32_t>(capacity));
        const int32_t flag_idx = local_expert * num_ranks + rank;
        const uint64_t packed = (tag << 32) | static_cast<uint64_t>(static_cast<uint32_t>(count));
        if (dst_rank == rank) {
          sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
          sycl::atomic_ref<
              uint64_t,
              sycl::memory_order::relaxed,
              sycl::memory_scope::device,
              sycl::access::address_space::global_space>
              flag_ref(pad[flag_idx]);
          flag_ref.store(packed, sycl::memory_order::release);
        } else {
          ishmemx_uint64_atomic_set_nbi_qp(
              pad + flag_idx, packed, dst_rank,
              static_cast<unsigned int>(local_expert));
        }
      }
      return;
    }

    // ===================== RECEIVER WG =====================
    const int32_t local_expert = wg_id - num_expert_wgs;
    if (local_expert >= num_local_experts) {
      return;
    }

    int32_t begin = 0;
    for (int32_t src_rank = 0; src_rank < num_ranks; ++src_rank) {
      const int32_t flag_idx = local_expert * num_ranks + src_rank;
      uint64_t pad_val = 0;
      if (src_rank == rank) {
        // Self-PE: use atomic acquire load
        if (lid == 0) {
          sycl::atomic_ref<
              uint64_t,
              sycl::memory_order::relaxed,
              sycl::memory_scope::device,
              sycl::access::address_space::global_space>
              flag_ref(pad[flag_idx]);
          while (true) {
            pad_val = flag_ref.load(sycl::memory_order::acquire);
            if ((pad_val >> 32) == static_cast<uint32_t>(tag & 0xFFFFFFFFu)) break;
          }
        }
      } else if (lid == 0) {
        // Remote: use ishmem_uint64_wait_until with CMP_GE.
        // Packed format: (tag << 32) | count. Wait for value >= (tag << 32)
        // which means the high 32 bits >= tag (our tag or a newer one).
        const uint64_t wait_val = tag << 32;
        ishmem_uint64_wait_until(pad + flag_idx, ISHMEM_CMP_GE, wait_val);
        // Read the actual value with acquire fence to see AMO update
        sycl::atomic_ref<
            uint64_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::system,
            sycl::access::address_space::global_space>
            pad_ref(pad[flag_idx]);
        pad_val = pad_ref.load(sycl::memory_order::acquire);
      }
      pad_val = sycl::group_broadcast(item.get_group(), pad_val, 0);
      sycl::group_barrier(grp);

      const int32_t count = static_cast<int32_t>(pad_val & 0xFFFFFFFFu);
      if (lid == 0) {
        packed_recv_layout_range[flag_idx] = static_cast<int64_t>(pack_range(count, begin));
        if (src_rank == 0) {
          packed_recv_count[local_expert] = 0;
        }
        packed_recv_count[local_expert] += count;
      }
      sycl::group_barrier(grp);

      const int64_t src_base = (static_cast<int64_t>(local_expert) * num_ranks + src_rank) * capacity;
      const int64_t dst_base = static_cast<int64_t>(local_expert) * num_ranks * capacity + begin;
      for (int32_t slot = 0; slot < count; ++slot) {
        for (int64_t h = lid; h < hidden; h += lsize) {
          packed_recv_x[(dst_base + slot) * hidden + h] = recv_data[(src_base + slot) * hidden + h];
        }
        if (lid == 0) {
          packed_recv_src_info[dst_base + slot] = recv_src[src_base + slot];
        }
      }
      sycl::group_barrier(grp);
      begin += count;
    }
  }
};

} // namespace

at::Tensor low_latency_dispatch_role_split_ishmem(
    const at::Tensor& x,
    const at::Tensor& topk_idx,
    at::Tensor packed_recv_x,
    at::Tensor packed_recv_src_info,
    at::Tensor packed_recv_count,
    at::Tensor packed_recv_layout_range,
    int64_t capacity,
    int64_t num_experts,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(x.dim() == 2, "low_latency_dispatch_role_split_ishmem: x must be 2D");
  TORCH_CHECK(x.is_contiguous(), "low_latency_dispatch_role_split_ishmem: x must be contiguous");
  TORCH_CHECK(
      x.scalar_type() == at::kBFloat16,
      "low_latency_dispatch_role_split_ishmem: x must be bfloat16");
  TORCH_CHECK(
      topk_idx.dim() == 2 && topk_idx.size(0) == x.size(0),
      "low_latency_dispatch_role_split_ishmem: topk_idx must be [num_tokens, num_topk]");
  TORCH_CHECK(
      topk_idx.scalar_type() == at::kInt,
      "low_latency_dispatch_role_split_ishmem: topk_idx must be int32");
  TORCH_CHECK(
      topk_idx.is_contiguous(),
      "low_latency_dispatch_role_split_ishmem: topk_idx must be contiguous");
  TORCH_CHECK(
      num_experts % world_size == 0,
      "low_latency_dispatch_role_split_ishmem: num_experts must be divisible by world_size");
  TORCH_CHECK(
      rank >= 0 && rank < world_size,
      "low_latency_dispatch_role_split_ishmem: rank must be in [0, world_size)");

  const int32_t num_local_experts = static_cast<int32_t>(num_experts / world_size);
  const int64_t hidden = x.size(1);
  const int64_t num_tokens = x.size(0);
  const int64_t num_topk = topk_idx.size(1);

  TORCH_CHECK(
      packed_recv_x.is_contiguous() && packed_recv_x.scalar_type() == at::kBFloat16 &&
          packed_recv_x.numel() ==
              static_cast<int64_t>(num_local_experts) * world_size * capacity * hidden,
      "low_latency_dispatch_role_split_ishmem: packed_recv_x must be "
      "[num_local_experts, num_ranks, capacity, hidden] bfloat16");
  TORCH_CHECK(
      packed_recv_src_info.is_contiguous() && packed_recv_src_info.scalar_type() == at::kInt &&
          packed_recv_src_info.numel() ==
              static_cast<int64_t>(num_local_experts) * world_size * capacity,
      "low_latency_dispatch_role_split_ishmem: packed_recv_src_info must be "
      "[num_local_experts, num_ranks, capacity] int32");
  TORCH_CHECK(
      packed_recv_count.is_contiguous() && packed_recv_count.scalar_type() == at::kInt &&
          packed_recv_count.numel() == num_local_experts,
      "low_latency_dispatch_role_split_ishmem: packed_recv_count must be [num_local_experts] int32");
  TORCH_CHECK(
      packed_recv_layout_range.is_contiguous() &&
          packed_recv_layout_range.scalar_type() == at::kLong &&
          packed_recv_layout_range.numel() == static_cast<int64_t>(num_local_experts) * world_size,
      "low_latency_dispatch_role_split_ishmem: packed_recv_layout_range must be "
      "[num_local_experts, num_ranks] int64");

  c10::Device device(c10::DeviceType::XPU, x.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  ensure_ishmem_initialized(x.device().index());
  TORCH_CHECK(
      ishmem_my_pe() == rank,
      "low_latency_dispatch_role_split_ishmem: ISHMEM PE does not match rank");
  TORCH_CHECK(
      ishmem_n_pes() == world_size,
      "low_latency_dispatch_role_split_ishmem: ISHMEM PE count does not match world_size");

  if (num_tokens == 0) {
    packed_recv_count.zero_();
    packed_recv_layout_range.zero_();
    return packed_recv_x;
  }

  const size_t elem_bytes = sizeof(at::BFloat16);
  const size_t send_bytes = static_cast<size_t>(num_tokens) * hidden * elem_bytes;
  const size_t recv_slots =
      static_cast<size_t>(num_local_experts) * world_size * capacity;
  const size_t recv_data_bytes = recv_slots * hidden * elem_bytes;
  const size_t recv_src_bytes = recv_slots * sizeof(int32_t);

  ensure_symmetric(send_bytes + recv_data_bytes + recv_src_bytes);
  auto* symm_base = static_cast<uint8_t*>(current_symmetric());
  auto* symm_send = reinterpret_cast<at::BFloat16*>(symm_base);
  auto* symm_recv_data =
      reinterpret_cast<at::BFloat16*>(symm_base + send_bytes);
  auto* symm_recv_src =
      reinterpret_cast<int32_t*>(symm_base + send_bytes + recv_data_bytes);

  const int32_t flag_slots_per_half = num_local_experts * static_cast<int32_t>(world_size);
  uint64_t* pad;
  int32_t* send_count;
  {
    auto& state = get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    pad = ensure_u64_array(state.pad, state.pad_slots, 2 * flag_slots_per_half, queue);
    send_count = ensure_send_count(
        state.send_count, state.send_count_slots, static_cast<int32_t>(num_experts), queue);
  }

  // Seed the symmetric send region with this rank's local tokens.
  sycl::event seed_dep = queue.memcpy(symm_send, x.data_ptr(), send_bytes);
  // Per-expert send-slot counters are purely local scratch: reset every call.
  sycl::event clear_dep =
      queue.memset(send_count, 0, static_cast<size_t>(num_experts) * sizeof(int32_t));

  uint64_t tag;
  {
    auto& state = get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    tag = ++state.iteration;
  }
  // Odd/even calls use different pad+counts halves so no barrier/clear is
  // needed between calls (the tag comparison rejects stale flags).
  const int32_t half = (tag % 2) * flag_slots_per_half;
  uint64_t* pad_half = pad + half;

  const int32_t num_expert_wgs = static_cast<int32_t>(num_experts);
  const int32_t num_receiver_wgs = num_local_experts;
  const int32_t total_wgs = num_expert_wgs + num_receiver_wgs;
  constexpr int64_t kWGSize = 32;

  // Require QPS_PER_PE >= num_local_experts.  Standard ishmem AMOs (e.g.
  // ishmem_int_atomic_set) use round-robin QP and ordered commit that deadlocks
  // if they land on a QP with uncommitted NBI WQEs from another WG.
  {
    const char* qps_env = std::getenv("ISHMEM_IBGDA_QPS_PER_PE");
    int qps_per_pe = qps_env ? std::atoi(qps_env) : 1;
    TORCH_CHECK(
        qps_per_pe >= num_local_experts,
        "low_latency_dispatch_role_split_ishmem: ISHMEM_IBGDA_QPS_PER_PE (",
        qps_per_pe, ") must be >= num_local_experts (", num_local_experts,
        "). Set ISHMEM_IBGDA_QPS_PER_PE=", num_local_experts, " or higher.");
  }

  {
    const auto& dev = queue.get_device();
    const int64_t max_concurrent =
        static_cast<int64_t>(dev.get_info<sycl::info::device::max_compute_units>());
    if (total_wgs > max_concurrent) {
      TORCH_WARN(
          "low_latency_dispatch_role_split_ishmem: total work-groups (",
          total_wgs,
          " = num_experts + num_local_experts) exceeds this device's compute "
          "unit count (",
          max_concurrent,
          "); the combined sender/receiver kernel may not have enough "
          "concurrent occupancy for every work-group to make forward "
          "progress and could hang. Reduce num_experts/world_size or run on "
          "a device with more compute units.");
    }
  }

  auto submit_kernel = [&](int32_t expert_wgs, int32_t receiver_wgs,
                           const std::vector<sycl::event>& deps) {
    const int32_t kernel_wgs = expert_wgs + receiver_wgs;
    return queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(deps);
      cgh.parallel_for(
          sycl::nd_range<1>(
              sycl::range<1>(static_cast<size_t>(kernel_wgs) * kWGSize),
              sycl::range<1>(kWGSize)),
          LowLatencyDispatchRoleSplitIshmemKernel{
              symm_send,
              symm_recv_data,
              symm_recv_src,
              send_count,
              pad_half,
              topk_idx.data_ptr<int32_t>(),
              packed_recv_x.data_ptr<at::BFloat16>(),
              packed_recv_src_info.data_ptr<int32_t>(),
              packed_recv_count.data_ptr<int32_t>(),
              packed_recv_layout_range.data_ptr<int64_t>(),
              num_tokens,
              hidden,
              num_topk,
              capacity,
              num_local_experts,
              static_cast<int32_t>(rank),
              static_cast<int32_t>(world_size),
              expert_wgs,
              tag});
    });
  };

  submit_kernel(num_expert_wgs, num_receiver_wgs, {seed_dep, clear_dep});

  return packed_recv_x;
}

void low_latency_dispatch_role_split_ishmem_finalize(const at::Tensor&) {
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
  if (state.pad != nullptr) {
    ishmem_barrier_all();
    ishmem_free(state.pad);
    state.pad = nullptr;
    state.pad_slots = 0;
  }

  // state.send_count is plain device memory (not symmetric heap); leaking it
  // across process exit is fine, but free it anyway if we still have the queue.
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
      "low_latency_dispatch_role_split_ishmem(Tensor x, Tensor topk_idx, "
      "Tensor(a!) packed_recv_x, Tensor(b!) packed_recv_src_info, "
      "Tensor(c!) packed_recv_count, Tensor(d!) packed_recv_layout_range, "
      "int capacity, int num_experts, int rank, int world_size) -> Tensor(a!)");
  m.def("low_latency_dispatch_role_split_ishmem_finalize(Tensor dummy) -> ()");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("low_latency_dispatch_role_split_ishmem", low_latency_dispatch_role_split_ishmem);
  m.impl(
      "low_latency_dispatch_role_split_ishmem_finalize",
      low_latency_dispatch_role_split_ishmem_finalize);
}
