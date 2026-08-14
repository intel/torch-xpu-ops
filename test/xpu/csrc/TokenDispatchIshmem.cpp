// TokenDispatchIshmem.cpp
//
// Standalone MoE-style token dispatch (all-to-all-v) implemented purely with
// ISHMEM device/host APIs, as a SINGLE on-device kernel modelled after
// RingAllgatherIshmem.cpp. Each local token has a (randomly chosen) destination
// rank; every token is PUSHed to its destination PE's receive buffer with an
// ISHMEM work-group-collective RDMA write, and all cross-rank completion
// signalling is done ON-DEVICE with ISHMEM signal pads -- there are no per-step
// host launches and no host barriers inside the dispatch.
//
// Routing (precomputed on the host / in Python and passed in):
//   order           : local token indices sorted by destination (stable)
//   expert_ids[t]   : global expert id of token row t (aligned with `tokens`)
//   send_offsets[d] : start index in `order` for destination d
//   send_counts[d]  : number of local tokens destined for d
//
// Receive layout (per PE): a symmetric buffer of world_size * capacity token
// slots. Slot (src * capacity + j) holds the j-th token that source `src` sent
// to this PE. `capacity` is chosen large enough for the worst case (== S).
//
// Work assignment: ONE work-group per destination (num_wg == world_size). The
// work-group index doubles as both axes:
//   - as a SENDER to destination d: WG d pushes all local tokens destined for d
//     into PE d's receive slots, work-group-fences to order the data before the
//     flag, then the leader writes the per-source count and finally raises the
//     signal pad on PE d (count ordered before the flag by a work-group fence).
//   - as a RECEIVER from source d: WG d waits on its own pad[d] slot for source
//     d's flag, after which recv slots [d*capacity ..] and counts[d] are valid.
// Because every (source, destination) pair always signals (even for zero tokens)
// and a send never waits on a receive, the kernel cannot deadlock.
//
// Only ISHMEM APIs are used for communication:
//   - ishmemx_putmem_nbi_qp          (QP-selected RDMA write per token)
//   - ishmem_fence                   (order count before the flag)
//   - ishmem_uint64_atomic_set       (leader writes the count / the flag)
//   - ishmem_uint64_wait_until       (device-side wait on our own pad slot)
//   - ishmem_malloc / ishmem_free / ishmem_barrier_all (symmetric heap)
//
// Registered op: symm_mem::token_dispatch_ishmem

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <comm/SYCLHelpers.h>
#include <ishmem.h>
#include <ishmemx.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>

namespace {

struct DispatchState {
  std::mutex mutex;
  bool initialized = false;
  void* symm = nullptr;      // recv region + send region
  size_t symm_bytes = 0;
  uint64_t* pad = nullptr;   // symmetric signal pad, 2 * world_size uint64
  int pad_slots = 0;
  uint64_t* counts = nullptr; // symmetric per-source counts, 2 * world_size u64
  int counts_slots = 0;
  uint64_t iteration = 0;    // strictly-increasing signal tag (never reused)
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
  return env_enabled("TOKEN_DISPATCH_ISHMEM_DEBUG");
}

void debug_log(int64_t pe, const char* msg) {
  if (debug_enabled()) {
    std::cerr << "[token_dispatch_ishmem pe " << pe << "] " << msg << std::endl;
  }
}

// Opt-in, host-side kernel timing: when enabled, the caller does an extra
// queue.wait() right before submitting the kernel (so the "start" timestamp is
// clean) and another right after the kernel completes, and reports the elapsed
// wall time in microseconds for JUST the dispatch kernel. This is a dedicated
// diagnostic path (NOT used by default) since forcing a queue.wait() here
// serializes the queue and will perturb steady-state pipelining/throughput --
// only turn it on when you specifically want a single-kernel latency number.
bool kernel_timing_enabled() {
  return env_enabled("TOKEN_DISPATCH_ISHMEM_KERNEL_TIME");
}

// Lazily bring up ISHMEM. Safe to co-exist with another extension that also
// initialises ISHMEM: we only call init if nobody else has.
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

// Ensure the symmetric data buffer (recv region + send region) can hold
// `bytes`. Collective (all PEs must call with the same size in the same order).
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
      "token_dispatch_ishmem: ishmem_malloc failed for ",
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

// Ensure a symmetric uint64 array holds `slots` entries, zero-init. Collective
// on first allocation / resize.
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
      "token_dispatch_ishmem: ishmem_malloc failed for ",
      bytes,
      " bytes");
  have_slots = slots;
  queue.memset(ptr, 0, bytes).wait_and_throw();
  ishmem_barrier_all();
  return ptr;
}

// Single-kernel ISHMEM token dispatch. One work-group per destination rank.
struct TokenDispatchIshmemKernel {
  uint8_t* symm_recv;          // [world_size * capacity, token_bytes]
  uint8_t* symm_send;          // [S, token_bytes] seeded local tokens
  uint64_t* pad;               // this call's half: [world_size] uint64
  uint64_t* counts;            // this call's half: [world_size] uint64
  const int32_t* order;        // [S], stable destination-sorted token indices
  const int32_t* expert_ids;   // [S], global expert id for each token row
  const int32_t* send_offsets; // [world_size]
  const int32_t* send_counts;  // [world_size]
  int64_t token_bytes;
  int64_t capacity;
  int32_t rank;
  int32_t world_size;
  int32_t experts_per_rank;
  uint64_t tag;

  void operator()(sycl::nd_item<1> item) const {
    const int32_t d = static_cast<int32_t>(item.get_group(0)); // destination
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lsize = static_cast<int32_t>(item.get_local_range(0));
    if (d >= world_size) {
      return;
    }

    const int32_t off = send_offsets[d];
    const int32_t cnt = send_counts[d];

    // SEND: non-contiguous per-token RDMA writes selected by `order`,
    // parallelized across threads in this work-group.
    const int64_t dst_slot_base = static_cast<int64_t>(rank) * capacity;
    for (int32_t j = lid; j < cnt; j += lsize) {
      const int32_t tok = order[off + j];
      const int64_t src_off = static_cast<int64_t>(tok) * token_bytes;
      const int64_t dst_slot = dst_slot_base + j;
      const int64_t dst_off = dst_slot * token_bytes;
      const int32_t expert = expert_ids[tok];
      const int32_t qp = expert - d * experts_per_rank;
      ishmem_putmem_nbi(
          static_cast<void*>(symm_recv + dst_off),
          static_cast<const void*>(symm_send + src_off),
          static_cast<size_t>(token_bytes),
          d);
      /*
      ishmemx_putmem_nbi_qp(
          static_cast<void*>(symm_recv + dst_off),
          static_cast<const void*>(symm_send + src_off),
          static_cast<size_t>(token_bytes),
          d,
          qp);
      */
    }

    // Ensure all work-group threads have issued their puts before leader
    // orders count/flag publication.
    item.barrier(sycl::access::fence_space::local_space);
    if (lid == 0) {
      ishmem_fence();
      ishmem_uint64_atomic_set(
          counts + rank, static_cast<uint64_t>(cnt), d);
      ishmem_uint64_atomic_set(pad + rank, tag, d);
    }

    // RECEIVE: wait for source d to finish sending to us. After the flag is
    // observed, recv slots [d*capacity ..] and counts[d] on this PE are valid.
    if (lid == 0) {
      ishmem_uint64_wait_until(pad + d, ISHMEM_CMP_EQ, tag);
    }
  }
};

at::Tensor token_dispatch_ishmem(
    const at::Tensor& tokens,
    const at::Tensor& order,
  const at::Tensor& expert_ids,
    const at::Tensor& send_offsets,
    const at::Tensor& send_counts,
    at::Tensor recv_buffer,
    at::Tensor recv_counts,
    int64_t capacity,
    int64_t rank,
  int64_t world_size,
  int64_t experts_per_rank) {
  TORCH_CHECK(tokens.dim() == 2, "token_dispatch_ishmem: tokens must be 2D");
  TORCH_CHECK(
      tokens.is_contiguous(), "token_dispatch_ishmem: tokens must be contiguous");
  TORCH_CHECK(
      recv_buffer.is_contiguous(),
      "token_dispatch_ishmem: recv_buffer must be contiguous");
  TORCH_CHECK(
      recv_buffer.dim() == 2 && recv_buffer.size(1) == tokens.size(1),
      "token_dispatch_ishmem: recv_buffer must be [world_size*capacity, hidden]");
  TORCH_CHECK(
      recv_buffer.size(0) == world_size * capacity,
      "token_dispatch_ishmem: recv_buffer rows must equal world_size*capacity");
  TORCH_CHECK(
      tokens.scalar_type() == recv_buffer.scalar_type(),
      "token_dispatch_ishmem: dtype mismatch between tokens and recv_buffer");
  TORCH_CHECK(
      order.scalar_type() == at::kInt && send_offsets.scalar_type() == at::kInt &&
          send_counts.scalar_type() == at::kInt,
      "token_dispatch_ishmem: order/send_offsets/send_counts must be int32");
    TORCH_CHECK(
      expert_ids.scalar_type() == at::kInt,
      "token_dispatch_ishmem: expert_ids must be int32");
  TORCH_CHECK(
      order.is_contiguous() && send_offsets.is_contiguous() &&
          send_counts.is_contiguous(),
      "token_dispatch_ishmem: routing tensors must be contiguous");
    TORCH_CHECK(
      expert_ids.is_contiguous(),
      "token_dispatch_ishmem: expert_ids must be contiguous");
  TORCH_CHECK(
      order.numel() == tokens.size(0),
      "token_dispatch_ishmem: order length must equal number of tokens");
    TORCH_CHECK(
      expert_ids.numel() == tokens.size(0),
      "token_dispatch_ishmem: expert_ids length must equal number of tokens");
  TORCH_CHECK(
      send_offsets.numel() == world_size && send_counts.numel() == world_size,
      "token_dispatch_ishmem: send_offsets/send_counts length must equal world_size");
  TORCH_CHECK(
      recv_counts.scalar_type() == at::kLong && recv_counts.is_contiguous() &&
          recv_counts.numel() == world_size,
      "token_dispatch_ishmem: recv_counts must be contiguous int64 [world_size]");
  TORCH_CHECK(
      rank >= 0 && rank < world_size,
      "token_dispatch_ishmem: rank must be in [0, world_size)");
  TORCH_CHECK(
      capacity >= tokens.size(0),
      "token_dispatch_ishmem: capacity must be >= number of local tokens");
    TORCH_CHECK(
      experts_per_rank > 0,
      "token_dispatch_ishmem: experts_per_rank must be > 0");

  const int64_t S = tokens.size(0);
  const int64_t H = tokens.size(1);

  c10::Device device(c10::DeviceType::XPU, tokens.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  ensure_ishmem_initialized(tokens.device().index());
  TORCH_CHECK(
      ishmem_my_pe() == rank,
      "token_dispatch_ishmem: ISHMEM PE does not match rank");
  TORCH_CHECK(
      ishmem_n_pes() == world_size,
      "token_dispatch_ishmem: ISHMEM PE count does not match world_size");

  const int64_t token_bytes = H * tokens.element_size();
  const size_t recv_bytes =
      static_cast<size_t>(world_size * capacity) * token_bytes;
  const size_t send_bytes = static_cast<size_t>(S) * token_bytes;

  if (S == 0) {
    return recv_buffer;
  }

  ensure_symmetric(recv_bytes + send_bytes);
  auto* symm_base = static_cast<uint8_t*>(current_symmetric());
  auto* symm_recv = symm_base;
  auto* symm_send = symm_base + recv_bytes;

  uint64_t* pad;
  uint64_t* counts;
  {
    auto& state = get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    pad = ensure_u64_array(
        state.pad, state.pad_slots, 2 * static_cast<int>(world_size), queue);
    counts = ensure_u64_array(
        state.counts,
        state.counts_slots,
        2 * static_cast<int>(world_size),
        queue);
  }

  // Seed the send region with local tokens. Kernel reads by `order` and does
  // non-contiguous per-token puts.
  sycl::event dep =
      queue.memcpy(symm_send, tokens.data_ptr(), send_bytes);

  uint64_t tag;
  {
    auto& state = get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    tag = ++state.iteration;
  }
  // Odd/even calls use different pad+count halves so no barrier is needed
  // between calls.
  const int64_t half = (tag % 2) * static_cast<int64_t>(world_size);
  uint64_t* pad_half = pad + half;
  uint64_t* counts_half = counts + half;

  constexpr int64_t threads = 256;
  const bool time_kernel = kernel_timing_enabled();
  std::chrono::high_resolution_clock::time_point t0;
  if (time_kernel) {
    // Drain everything queued so far (memcpy seed) so the timer below only
    // covers the dispatch kernel itself.
    queue.wait();
    t0 = std::chrono::high_resolution_clock::now();
  }
  auto dispatch_event = queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(dep);
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(world_size) * threads),
            sycl::range<1>(threads)),
        TokenDispatchIshmemKernel{
            symm_recv,
            symm_send,
            pad_half,
            counts_half,
            order.data_ptr<int32_t>(),
            expert_ids.data_ptr<int32_t>(),
            send_offsets.data_ptr<int32_t>(),
            send_counts.data_ptr<int32_t>(),
            token_bytes,
            capacity,
            static_cast<int32_t>(rank),
            static_cast<int32_t>(world_size),
            static_cast<int32_t>(experts_per_rank),
            tag});
  });
  if (time_kernel) {
    queue.wait();
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::cerr << "[token_dispatch_ishmem pe " << rank << "] kernel tag=" << tag
               << " elapsed=" << us << " us" << std::endl;
  }

  queue.memcpy(recv_buffer.data_ptr(), symm_recv, recv_bytes);
  queue.memcpy(
      recv_counts.data_ptr(),
      counts_half,
      static_cast<size_t>(world_size) * sizeof(uint64_t));

  return recv_buffer;
}

void token_dispatch_ishmem_finalize(const at::Tensor&) {
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
  if (state.counts != nullptr) {
    ishmem_barrier_all();
    ishmem_free(state.counts);
    state.counts = nullptr;
    state.counts_slots = 0;
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

} // namespace

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "token_dispatch_ishmem(Tensor tokens, Tensor order, Tensor expert_ids, Tensor send_offsets, "
      "Tensor send_counts, Tensor(a!) recv_buffer, Tensor(b!) recv_counts, "
      "int capacity, int rank, int world_size, int experts_per_rank) -> Tensor(a!)");
  m.def("token_dispatch_ishmem_finalize(Tensor dummy) -> ()");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("token_dispatch_ishmem", token_dispatch_ishmem);
  m.impl("token_dispatch_ishmem_finalize", token_dispatch_ishmem_finalize);
}
