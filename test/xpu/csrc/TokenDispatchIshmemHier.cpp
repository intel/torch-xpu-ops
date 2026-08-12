// TokenDispatchIshmemHier.cpp
//
// Hierarchical (2-level) MoE token dispatch for a two-PCIE-domain topology,
// implemented with ISHMEM device/host APIs and TWO on-device kernels. The
// output is byte-identical to the flat TokenDispatchIshmem.cpp: the receive
// buffer on the destination PE is laid out as recv[src * capacity + j] = the
// j-th token (in the source's original order) that original source `src` sent
// to this PE, and recv_counts[src] is that count.
//
// Topology: PCIE_DOMAIN == P cards per domain, EXACTLY two domains, so
// world_size == 2 * P. domain(r) = r / P, pos(r) = r % P. The cross-domain
// RDMA is done in "mirror" fashion: rank r's cross-domain traffic is sent to
// its mirror partner mirror(r) = (r + P) % world_size (flip domain, keep pos),
// which is a bijection between the two domains, so every mirror receives from
// exactly ONE remote source.
//
// Byte-identical trick: every local token carries the ABSOLUTE final slot it
// must land in on its destination PE, final_slot = src * capacity + j (computed
// in Python, exactly as the flat op computes it). Because the slot is explicit,
// forwarding order is irrelevant and sources never get mixed up.
//
// Kernel 1 (cross-domain RDMA, mirror exchange): rank r pushes every
// cross-domain token to mirror(r)'s staging region along with, per token, its
// final_slot and dest_pos (the destination's position within the domain,
// 0..P-1). It writes the staged count, fences, and raises the staging flag
// (ALWAYS, even for zero cross tokens). Each work-group then waits on its OWN
// local staging flag, so kernel 1 also RECEIVES the mirror's push: the
// cross-domain send+recv is fully closed before kernel 1 returns.
//
// Kernel 2 (intra-domain switch): the cross-domain staging is already in place,
// so this kernel is a pure local switch. One work-group per domain peer e
// (final dest = my_domain*P + e):
//   Part A: push local SAME-domain tokens with dest_pos==e to dest's final slot.
//   Part B: push staged tokens with dest_pos==e to dest's final slot.
// After quiet+fence it raises the intra-domain flag on the dest peer, then waits
// for all domain peers to finish writing to it before the host copies out.
//
// recv_counts are precomputed in Python (a tiny world_size x world_size count
// matrix all-gather) and passed in; the op just copies them out. This avoids any
// per-(src,dst) count signalling in the kernels.
//
// Registered op: symm_mem::token_dispatch_ishmem_hier

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <comm/SYCLHelpers.h>
#include <ishmem.h>
#include <ishmemx.h>
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>

namespace {

struct HierState {
  std::mutex mutex;
  bool initialized = false;
  void* symm = nullptr;   // final region + send region + stage region
  size_t symm_bytes = 0;
  int32_t* stage_slot = nullptr;    // symmetric [S] final slots of staged tokens
  int stage_slot_slots = 0;
  int32_t* stage_destpos = nullptr; // symmetric [S] dest position of staged tokens
  int stage_destpos_slots = 0;
  uint64_t* stage_ready = nullptr;  // symmetric [2] per-PE staging-arrived flag (tag parity)
  int stage_ready_slots = 0;
  uint64_t* done_counter = nullptr; // local [2] device WG completion counter (tag parity)
  int done_counter_slots = 0;
  uint64_t* stage_count = nullptr;  // symmetric [2] staged token count (tag parity)
  int stage_count_slots = 0;
  uint64_t* intra_pad = nullptr;    // symmetric [2*P] intra-domain flags (tag parity)
  int intra_pad_slots = 0;
  uint64_t iteration = 0;
};

HierState& get_state() {
  static HierState state;
  return state;
}

bool env_enabled(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

bool debug_enabled() {
  return env_enabled("TOKEN_DISPATCH_ISHMEM_HIER_DEBUG");
}

void debug_log(int64_t pe, const char* msg) {
  if (debug_enabled()) {
    std::cerr << "[token_dispatch_ishmem_hier pe " << pe << "] " << msg
              << std::endl;
  }
}

// Number of work-groups kernel 1 launches for the cross-domain push. Fixed per
// run so the mirror partner's kernel 2 knows exactly how many per-WG staging
// flags to wait on. Compile-time default, overridable via the DISPATCH_MAX_WG
// env var (must be identical on every rank).
constexpr int32_t DISPATCH_MAX_WG = 16;

int32_t dispatch_max_wg() {
  static const int32_t cached = [] {
    const char* v = std::getenv("DISPATCH_MAX_WG");
    if (v != nullptr && v[0] != '\0') {
      const int val = std::atoi(v);
      if (val >= 1 && val <= 1024) {
        return val;
      }
    }
    return static_cast<int>(DISPATCH_MAX_WG);
  }();
  return cached;
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
      "token_dispatch_ishmem_hier: ishmem_malloc failed for ",
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

uint64_t* ensure_u64_array(
    uint64_t*& ptr,
    int& have_slots,
    int slots,
    sycl::queue& queue) {
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
      "token_dispatch_ishmem_hier: ishmem_malloc failed for ",
      bytes,
      " bytes");
  have_slots = slots;
  queue.memset(ptr, 0, bytes).wait_and_throw();
  ishmem_barrier_all();
  return ptr;
}

int32_t* ensure_i32_array(int32_t*& ptr, int& have_slots, int slots) {
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
  const size_t bytes = static_cast<size_t>(slots) * sizeof(int32_t);
  ptr = static_cast<int32_t*>(ishmem_malloc(bytes));
  TORCH_CHECK(
      ptr != nullptr,
      "token_dispatch_ishmem_hier: ishmem_malloc failed for ",
      bytes,
      " bytes");
  have_slots = slots;
  ishmem_barrier_all();
  return ptr;
}

// Kernel 1: nwg work-groups push the cross-domain tokens (plus their
// final_slot / dest_pos metadata) to the mirror partner's staging region. Each
// work-group takes a contiguous chunk of the cross tokens and, after quieting
// its own puts, bumps a device-wide completion counter. The LAST work-group to
// finish publishes the staged count and raises a SINGLE per-PE staging-arrived
// flag on the mirror; only WG0 spin-waits for the mirror's flag. Bounding the
// spinning work-groups to one per PE means a large nwg cannot cause an
// occupancy (scheduling) deadlock. nwg is fixed (== dispatch_max_wg) so both
// sides launch the same number of pushers.
struct TokenDispatchIshmemHierK1 {
  uint8_t* symm_send;           // local [S, token_bytes]
  uint8_t* stage;               // symmetric staging base (write on mirror)
  int32_t* stage_slot;          // symmetric [S] (write on mirror)
  int32_t* stage_destpos;       // symmetric [S] (write on mirror)
  uint64_t* stage_count;        // symmetric [1] this-call half (write on mirror)
  uint64_t* stage_ready;        // symmetric [1] this-call half (write on mirror)
  uint64_t* done_counter;       // local [1] this-call half (device WG counter)
  const int32_t* cross_order;   // local [num_cross]
  const int32_t* cross_slot;    // local [num_cross]
  const int32_t* cross_destpos; // local [num_cross]
  int64_t token_bytes;
  int32_t num_cross;
  int32_t nwg;
  int32_t mirror;
  uint64_t tag;

  void operator()(sycl::nd_item<1> item) const {
    auto grp = item.get_group();
    const int32_t wg = static_cast<int32_t>(item.get_group(0));
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    if (wg >= nwg) {
      return;
    }

    // Contiguous chunk of the cross tokens for this work-group.
    const int32_t chunk = (num_cross + nwg - 1) / nwg;
    int32_t start = wg * chunk;
    int32_t end = start + chunk;
    if (start > num_cross) {
      start = num_cross;
    }
    if (end > num_cross) {
      end = num_cross;
    }

    for (int32_t k = start; k < end; ++k) {
      const int32_t tok = cross_order[k];
      const int64_t src_off = static_cast<int64_t>(tok) * token_bytes;
      const int64_t dst_off = static_cast<int64_t>(k) * token_bytes;
      ishmemx_putmem_nbi_work_group(
          static_cast<void*>(stage + dst_off),
          static_cast<const void*>(symm_send + src_off),
          static_cast<size_t>(token_bytes),
          mirror,
          grp);
    }
    const int32_t cnt = end - start;
    if (cnt > 0) {
      ishmemx_putmem_nbi_work_group(
          static_cast<void*>(stage_slot + start),
          static_cast<const void*>(cross_slot + start),
          static_cast<size_t>(cnt) * sizeof(int32_t),
          mirror,
          grp);
      ishmemx_putmem_nbi_work_group(
          static_cast<void*>(stage_destpos + start),
          static_cast<const void*>(cross_destpos + start),
          static_cast<size_t>(cnt) * sizeof(int32_t),
          mirror,
          grp);
    }

    // Elect the last work-group to finish. Only it publishes the staged count
    // and raises the single per-PE staging-arrived flag on the mirror; only WG0
    // spin-waits for the incoming flag. This bounds spinning work-groups to one
    // per PE, so a large nwg cannot deadlock on GPU scheduling: every non-WG0
    // work-group increments and exits, freeing slots for the rest to run.
    if (lid == 0) {
      sycl::atomic_ref<
          uint64_t,
          sycl::memory_order::relaxed,
          sycl::memory_scope::device,
          sycl::access::address_space::global_space>
          ctr(done_counter[0]);
      const uint64_t done = ctr.fetch_add(static_cast<uint64_t>(1)) + 1;
      if (done == static_cast<uint64_t>(nwg)) {
        ishmem_uint64_atomic_set(
            stage_count, static_cast<uint64_t>(num_cross), mirror);
        ishmemx_fence_work_group(grp); // order count before the ready flag on the mirror
        ishmem_uint64_atomic_set(stage_ready, tag, mirror);
      }
    }

    // Only WG0 waits for the mirror's single staging-arrived flag; wait_until
    // polls our LOCAL copy of the symmetric offset. No trailing barrier/fence
    // is needed: nothing in this kernel reads the staging after the wait; the
    // host memcpy (ordered after kernel completion) and kernel 2 (own acquire
    // fence at entry) handle visibility of the mirror's RDMA writes.
    if (wg == 0 && lid == 0) {
      ishmem_uint64_wait_until(stage_ready, ISHMEM_CMP_EQ, tag);
    }
  }
};

// Kernel 2: one work-group per domain peer performs the intra-domain switch of
// both the local same-domain tokens (Part A) and the staged cross-domain tokens
// received in kernel 1 (Part B). Kernel 1 already closed the cross-domain
// exchange, so this kernel does no cross-domain wait.
struct TokenDispatchIshmemHierK2 {
  uint8_t* symm_send;           // local [S, token_bytes]
  uint8_t* stage;               // local staging base (filled by kernel 1)
  uint8_t* final_buf;           // symmetric final base (write on dest peer)
  int32_t* stage_slot;          // local [S]
  int32_t* stage_destpos;       // local [S]
  uint64_t* stage_count;        // local [1] this-call half (read)
  uint64_t* intra_pad;          // symmetric [P] this-call half (write on dest / wait local)
  const int32_t* local_order;   // local [num_local]
  const int32_t* local_slot;    // local [num_local]
  const int32_t* local_destpos; // local [num_local]
  int64_t token_bytes;
  int32_t num_local;
  int32_t P;
  int32_t my_domain;
  int32_t my_pos;
  uint64_t tag;

  void operator()(sycl::nd_item<1> item) const {
    auto grp = item.get_group();
    const int32_t e = static_cast<int32_t>(item.get_group(0));
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    if (e >= P) {
      return;
    }
    const int32_t dest = my_domain * P + e;

    // Cross-domain staging already landed in kernel 1; this kernel is a pure
    // intra-domain switch. Acquire so the mirror's RDMA writes are visible.
    sycl::atomic_fence(
        sycl::memory_order::acquire, sycl::memory_scope::system);

    sycl::atomic_ref<
        uint64_t,
        sycl::memory_order::relaxed,
        sycl::memory_scope::system,
        sycl::access::address_space::global_space>
        count_ref(stage_count[0]);
    const int32_t staged = static_cast<int32_t>(count_ref.load());

    // Part A: local same-domain tokens destined for peer e.
    for (int32_t k = 0; k < num_local; ++k) {
      if (local_destpos[k] != e) {
        continue;
      }
      const int32_t tok = local_order[k];
      const int64_t src_off = static_cast<int64_t>(tok) * token_bytes;
      const int64_t dst_off = static_cast<int64_t>(local_slot[k]) * token_bytes;
      ishmemx_putmem_nbi_work_group(
          static_cast<void*>(final_buf + dst_off),
          static_cast<const void*>(symm_send + src_off),
          static_cast<size_t>(token_bytes),
          dest,
          grp);
    }

    // Part B: staged cross-domain tokens destined for peer e.
    for (int32_t k = 0; k < staged; ++k) {
      if (stage_destpos[k] != e) {
        continue;
      }
      const int64_t src_off = static_cast<int64_t>(k) * token_bytes;
      const int64_t dst_off = static_cast<int64_t>(stage_slot[k]) * token_bytes;
      ishmemx_putmem_nbi_work_group(
          static_cast<void*>(final_buf + dst_off),
          static_cast<const void*>(stage + src_off),
          static_cast<size_t>(token_bytes),
          dest,
          grp);
    }

    ishmemx_quiet_work_group(grp);
    ishmemx_fence_work_group(grp);
    if (lid == 0) {
      ishmem_uint64_atomic_set(intra_pad + my_pos, tag, dest);
    }

    // Wait until peer e has finished writing to us.
    if (lid == 0) {
      ishmem_uint64_wait_until(intra_pad + e, ISHMEM_CMP_EQ, tag);
    }
    item.barrier(sycl::access::fence_space::local_space);
    sycl::atomic_fence(
        sycl::memory_order::acquire, sycl::memory_scope::system);
  }
};

at::Tensor token_dispatch_ishmem_hier(
    const at::Tensor& tokens,
    const at::Tensor& cross_order,
    const at::Tensor& cross_slot,
    const at::Tensor& cross_destpos,
    const at::Tensor& local_order,
    const at::Tensor& local_slot,
    const at::Tensor& local_destpos,
    at::Tensor recv_buffer,
    at::Tensor recv_counts,
    const at::Tensor& recv_counts_in,
    int64_t capacity,
    int64_t rank,
    int64_t world_size,
    int64_t pcie_domain) {
  TORCH_CHECK(tokens.dim() == 2, "token_dispatch_ishmem_hier: tokens must be 2D");
  TORCH_CHECK(
      tokens.is_contiguous(),
      "token_dispatch_ishmem_hier: tokens must be contiguous");
  TORCH_CHECK(
      recv_buffer.is_contiguous() && recv_buffer.dim() == 2 &&
          recv_buffer.size(1) == tokens.size(1),
      "token_dispatch_ishmem_hier: recv_buffer must be [world_size*capacity, hidden]");
  TORCH_CHECK(
      recv_buffer.size(0) == world_size * capacity,
      "token_dispatch_ishmem_hier: recv_buffer rows must equal world_size*capacity");
  TORCH_CHECK(
      tokens.scalar_type() == recv_buffer.scalar_type(),
      "token_dispatch_ishmem_hier: dtype mismatch between tokens and recv_buffer");
  TORCH_CHECK(
      pcie_domain > 0 && world_size == 2 * pcie_domain,
      "token_dispatch_ishmem_hier: requires exactly two domains (world_size == 2 * pcie_domain)");
  TORCH_CHECK(
      cross_order.scalar_type() == at::kInt &&
          cross_slot.scalar_type() == at::kInt &&
          cross_destpos.scalar_type() == at::kInt &&
          local_order.scalar_type() == at::kInt &&
          local_slot.scalar_type() == at::kInt &&
          local_destpos.scalar_type() == at::kInt,
      "token_dispatch_ishmem_hier: routing tensors must be int32");
  TORCH_CHECK(
      cross_order.is_contiguous() && cross_slot.is_contiguous() &&
          cross_destpos.is_contiguous() && local_order.is_contiguous() &&
          local_slot.is_contiguous() && local_destpos.is_contiguous(),
      "token_dispatch_ishmem_hier: routing tensors must be contiguous");
  TORCH_CHECK(
      cross_order.numel() == cross_slot.numel() &&
          cross_order.numel() == cross_destpos.numel(),
      "token_dispatch_ishmem_hier: cross_* tensors must be the same length");
  TORCH_CHECK(
      local_order.numel() == local_slot.numel() &&
          local_order.numel() == local_destpos.numel(),
      "token_dispatch_ishmem_hier: local_* tensors must be the same length");
  TORCH_CHECK(
      recv_counts.scalar_type() == at::kLong && recv_counts.is_contiguous() &&
          recv_counts.numel() == world_size,
      "token_dispatch_ishmem_hier: recv_counts must be contiguous int64 [world_size]");
  TORCH_CHECK(
      recv_counts_in.scalar_type() == at::kLong &&
          recv_counts_in.is_contiguous() &&
          recv_counts_in.numel() == world_size,
      "token_dispatch_ishmem_hier: recv_counts_in must be contiguous int64 [world_size]");
  TORCH_CHECK(
      rank >= 0 && rank < world_size,
      "token_dispatch_ishmem_hier: rank must be in [0, world_size)");
  TORCH_CHECK(
      capacity >= tokens.size(0),
      "token_dispatch_ishmem_hier: capacity must be >= number of local tokens");

  const int64_t S = tokens.size(0);
  const int64_t H = tokens.size(1);
  const int64_t P = pcie_domain;
  const int32_t dispatch_wg = dispatch_max_wg();
  const int32_t num_cross = static_cast<int32_t>(cross_order.numel());
  const int32_t num_local = static_cast<int32_t>(local_order.numel());

  c10::Device device(c10::DeviceType::XPU, tokens.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  ensure_ishmem_initialized(tokens.device().index());
  TORCH_CHECK(
      ishmem_my_pe() == rank,
      "token_dispatch_ishmem_hier: ISHMEM PE does not match rank");
  TORCH_CHECK(
      ishmem_n_pes() == world_size,
      "token_dispatch_ishmem_hier: ISHMEM PE count does not match world_size");

  // recv_counts are precomputed; just publish them.
  queue.memcpy(
      recv_counts.data_ptr(),
      recv_counts_in.data_ptr(),
      static_cast<size_t>(world_size) * sizeof(int64_t));

  if (S == 0) {
    return recv_buffer;
  }

  const int64_t token_bytes = H * tokens.element_size();
  const size_t final_bytes =
      static_cast<size_t>(world_size * capacity) * token_bytes;
  const size_t send_bytes = static_cast<size_t>(S) * token_bytes;
  // Staging is double-buffered by tag parity: the mirror partner has no direct
  // per-call handshake with us, so a one-iteration drift could otherwise let it
  // overwrite our staging before kernel 2 reads it.
  const size_t stage_half_bytes = static_cast<size_t>(S) * token_bytes;
  const size_t stage_bytes = 2 * stage_half_bytes;

  ensure_symmetric(final_bytes + send_bytes + stage_bytes);
  auto* symm_base = static_cast<uint8_t*>(current_symmetric());
  auto* symm_final = symm_base;
  auto* symm_send = symm_base + final_bytes;
  auto* symm_stage = symm_send + send_bytes;

  int32_t* stage_slot;
  int32_t* stage_destpos;
  uint64_t* stage_ready;
  uint64_t* done_counter;
  uint64_t* stage_count;
  uint64_t* intra_pad;
  {
    auto& state = get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    stage_slot = ensure_i32_array(
        state.stage_slot, state.stage_slot_slots, 2 * static_cast<int>(S));
    stage_destpos = ensure_i32_array(
        state.stage_destpos, state.stage_destpos_slots, 2 * static_cast<int>(S));
    stage_ready =
        ensure_u64_array(state.stage_ready, state.stage_ready_slots, 2, queue);
    done_counter = ensure_u64_array(
        state.done_counter, state.done_counter_slots, 2, queue);
    stage_count =
        ensure_u64_array(state.stage_count, state.stage_count_slots, 2, queue);
    intra_pad = ensure_u64_array(
        state.intra_pad, state.intra_pad_slots, 2 * static_cast<int>(P), queue);
  }

  // Seed the send region with the local tokens.
  sycl::event seed = queue.memcpy(symm_send, tokens.data_ptr(), send_bytes);

  uint64_t tag;
  {
    auto& state = get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    tag = ++state.iteration;
  }
  const int64_t parity = static_cast<int64_t>(tag % 2);
  uint64_t* stage_ready_half = stage_ready + parity;
  uint64_t* done_counter_half = done_counter + parity;
  uint64_t* stage_count_half = stage_count + parity;
  uint64_t* intra_pad_half = intra_pad + parity * P;
  uint8_t* symm_stage_half = symm_stage + parity * stage_half_bytes;
  int32_t* stage_slot_half = stage_slot + parity * S;
  int32_t* stage_destpos_half = stage_destpos + parity * S;

  // Reset this call's per-PE WG completion counter before launching kernel 1.
  sycl::event zero_ctr = queue.memset(done_counter_half, 0, sizeof(uint64_t));

  const int32_t my_domain = static_cast<int32_t>(rank / P);
  const int32_t my_pos = static_cast<int32_t>(rank % P);
  const int32_t mirror = static_cast<int32_t>((rank + P) % world_size);

  constexpr int64_t threads = 256;
  const bool dbg = debug_enabled();

  if (dbg) {
    debug_log(rank, "submitting kernel 1 (cross-domain mirror push)");
  }
  // Kernel 1: dispatch_wg work-groups, cross-domain mirror push.
  auto k1 = queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(seed);
    cgh.depends_on(zero_ctr);
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(dispatch_wg) * threads),
            sycl::range<1>(threads)),
        TokenDispatchIshmemHierK1{
            symm_send,
            symm_stage_half,
            stage_slot_half,
            stage_destpos_half,
            stage_count_half,
            stage_ready_half,
            done_counter_half,
            cross_order.data_ptr<int32_t>(),
            cross_slot.data_ptr<int32_t>(),
            cross_destpos.data_ptr<int32_t>(),
            token_bytes,
            num_cross,
            dispatch_wg,
            mirror,
            tag});
  });
  if (dbg) {
    k1.wait_and_throw();
    debug_log(rank, "kernel 1 done; submitting kernel 2 (intra-domain)");
  }

  // Mirror-dispatch only: this op currently runs kernel 1 alone, so expose the
  // cross-domain tokens this rank received from its mirror partner (front of the
  // staging half) in recv_buffer for the host/UT to verify.
  queue.submit([&](sycl::handler& cgh) {
    cgh.memcpy(recv_buffer.data_ptr(), symm_stage_half, stage_half_bytes);
  });
  /*
  // Kernel 2: one work-group per domain peer, intra-domain dispatch.
  auto k2 = queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(k1);
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(P) * threads),
            sycl::range<1>(threads)),
        TokenDispatchIshmemHierK2{
            symm_send,
            symm_stage_half,
            symm_final,
            stage_slot_half,
            stage_destpos_half,
            stage_count_half,
            intra_pad_half,
            local_order.data_ptr<int32_t>(),
            local_slot.data_ptr<int32_t>(),
            local_destpos.data_ptr<int32_t>(),
            token_bytes,
            num_local,
            static_cast<int32_t>(P),
            my_domain,
            my_pos,
            tag});
  });
  if (dbg) {
    k2.wait_and_throw();
    debug_log(rank, "kernel 2 done");
  }

  queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(k2);
    cgh.memcpy(recv_buffer.data_ptr(), symm_final, final_bytes);
  });
  */

  return recv_buffer;
}

void token_dispatch_ishmem_hier_finalize(const at::Tensor&) {
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
  auto free_symm = [](void*& p) {
    if (p != nullptr) {
      ishmem_barrier_all();
      ishmem_free(p);
      p = nullptr;
    }
  };
  if (state.symm != nullptr) {
    ishmem_barrier_all();
    ishmem_free(state.symm);
    state.symm = nullptr;
    state.symm_bytes = 0;
  }
  free_symm(reinterpret_cast<void*&>(state.stage_slot));
  state.stage_slot_slots = 0;
  free_symm(reinterpret_cast<void*&>(state.stage_destpos));
  state.stage_destpos_slots = 0;
  free_symm(reinterpret_cast<void*&>(state.stage_ready));
  state.stage_ready_slots = 0;
  free_symm(reinterpret_cast<void*&>(state.done_counter));
  state.done_counter_slots = 0;
  free_symm(reinterpret_cast<void*&>(state.stage_count));
  state.stage_count_slots = 0;
  free_symm(reinterpret_cast<void*&>(state.intra_pad));
  state.intra_pad_slots = 0;
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

} // namespace

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "token_dispatch_ishmem_hier(Tensor tokens, Tensor cross_order, "
      "Tensor cross_slot, Tensor cross_destpos, Tensor local_order, "
      "Tensor local_slot, Tensor local_destpos, Tensor(a!) recv_buffer, "
      "Tensor(b!) recv_counts, Tensor recv_counts_in, int capacity, int rank, "
      "int world_size, int pcie_domain) -> Tensor(a!)");
  m.def("token_dispatch_ishmem_hier_finalize(Tensor dummy) -> ()");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("token_dispatch_ishmem_hier", token_dispatch_ishmem_hier);
  m.impl("token_dispatch_ishmem_hier_finalize", token_dispatch_ishmem_hier_finalize);
}
