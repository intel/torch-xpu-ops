#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <comm/SYCLHelpers.h>

namespace {

constexpr int32_t WG_SIZE = 256;
constexpr int32_t MAX_EXPERTS = 512;

inline int32_t owner_expert_start(int32_t rank, int32_t num_experts, int32_t world_size) {
  const int32_t base = num_experts / world_size;
  const int32_t rem = num_experts % world_size;
  return rank * base + (rank < rem ? rank : rem);
}

inline int32_t owner_expert_count(int32_t rank, int32_t num_experts, int32_t world_size) {
  const int32_t base = num_experts / world_size;
  const int32_t rem = num_experts % world_size;
  return base + (rank < rem ? 1 : 0);
}

// ---------------------------------------------------------------------------
// Kernel 1: Count tokens per expert using SLM histogram, then exclusive
//           prefix-sum → expert_offsets.  Single work-group.
// ---------------------------------------------------------------------------
struct NotifyDispatchCountPrefixKernel : __SYCL_KER_CONFIG_CONVENTION__ {
  const int64_t* topk_rank_ptrs;
  int32_t* expert_offsets;
  int32_t* psum_num_recv_tokens_per_expert;
  int32_t num_tokens_per_rank;
  int32_t topk;
  int32_t topk_storage_stride;
  int32_t num_experts;
  int32_t world_size;
  int32_t local_expert_start;
  int32_t local_expert_end;
  int64_t total_pairs;

  sycl::local_accessor<int32_t, 1> slm;

  void sycl_ker_config_convention(sycl::handler& cgh) {
    slm = sycl::local_accessor<int32_t, 1>(MAX_EXPERTS, cgh);
  }

  void operator()(sycl::nd_item<1> item) const {
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lsize = static_cast<int32_t>(item.get_local_range(0));

    // Zero SLM histogram.
    for (int32_t e = lid; e < num_experts; e += lsize)
      slm[e] = 0;
    item.barrier(sycl::access::fence_space::local_space);

    // Count with SLM atomics (no global contention).
    for (int64_t idx = lid; idx < total_pairs; idx += lsize) {
      const int32_t token_idx = static_cast<int32_t>(idx / topk);
      const int32_t k = static_cast<int32_t>(idx % topk);
      const int32_t src_rank = token_idx / num_tokens_per_rank;
      const int32_t src_token = token_idx % num_tokens_per_rank;
      const int32_t* src_topk =
          reinterpret_cast<const int32_t*>(topk_rank_ptrs[src_rank]);
      const int32_t expert = src_topk[src_token * topk_storage_stride + k];
      if (expert >= 0 && expert < num_experts) {
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group,
                         sycl::access::address_space::local_space>
            ref(slm[expert]);
        ref.fetch_add(1);
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    // Write local recv counts from histogram (one thread per local expert).
    for (int32_t e = local_expert_start + lid; e < local_expert_end;
         e += lsize) {
      psum_num_recv_tokens_per_expert[e - local_expert_start] = slm[e];
    }

    // Exclusive prefix sum (single thread) → expert_offsets.
    if (lid == 0) {
      int32_t running = 0;
      for (int32_t e = 0; e < num_experts; ++e) {
        const int32_t c = slm[e];
        expert_offsets[e] = running;
        running += c;
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Kernel 2: Assign scatter_idx using SLM offset reservation.
//
// Each work-group:
//   (a) Build a local histogram in SLM (first pass over data).
//   (b) Reserve a contiguous chunk of scatter positions per expert from
//       global expert_offsets (one atomic per expert per WG).
//   (c) Second pass: assign scatter_idx using per-WG reserved chunk
//       and a local SLM counter.  Also write global_topk_idx.
//
// This reduces global atomics from O(total_pairs) to O(num_experts × num_wgs).
// ---------------------------------------------------------------------------
struct NotifyDispatchAssignKernel : __SYCL_KER_CONFIG_CONVENTION__ {
  const int64_t* topk_rank_ptrs;
  int32_t* global_topk_idx;
  int32_t* scatter_idx;
  int32_t* expert_offsets;
  int32_t num_tokens_per_rank;
  int32_t topk;
  int32_t topk_storage_stride;
  int32_t num_experts;
  int32_t world_size;
  int64_t total_pairs;

  // SLM layout: [0..num_experts) = histogram / local counter,
  //             [MAX_EXPERTS..MAX_EXPERTS+num_experts) = reserved base offset.
  sycl::local_accessor<int32_t, 1> slm;

  void sycl_ker_config_convention(sycl::handler& cgh) {
    slm = sycl::local_accessor<int32_t, 1>(2 * MAX_EXPERTS, cgh);
  }

  void operator()(sycl::nd_item<1> item) const {
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lsize = static_cast<int32_t>(item.get_local_range(0));
    const int64_t wg_start =
        static_cast<int64_t>(item.get_group(0)) * lsize;
    const int64_t wg_end =
        (wg_start + lsize < total_pairs) ? wg_start + lsize : total_pairs;

    // (a) Zero local histogram.
    for (int32_t e = lid; e < num_experts; e += lsize)
      slm[e] = 0;
    item.barrier(sycl::access::fence_space::local_space);

    // (a) Count entries per expert for this WG's chunk.
    for (int64_t idx = wg_start + lid; idx < wg_end; idx += lsize) {
      const int32_t token_idx = static_cast<int32_t>(idx / topk);
      const int32_t k = static_cast<int32_t>(idx % topk);
      const int32_t src_rank = token_idx / num_tokens_per_rank;
      const int32_t src_token = token_idx % num_tokens_per_rank;
      const int32_t* src_topk =
          reinterpret_cast<const int32_t*>(topk_rank_ptrs[src_rank]);
      const int32_t expert = src_topk[src_token * topk_storage_stride + k];
      if (expert >= 0 && expert < num_experts) {
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group,
                         sycl::access::address_space::local_space>
            ref(slm[expert]);
        ref.fetch_add(1);
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    // (b) Reserve chunk from global expert_offsets and reset local counter.
    for (int32_t e = lid; e < num_experts; e += lsize) {
      const int32_t count = slm[e];
      if (count > 0) {
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device>
            gref(expert_offsets[e]);
        slm[MAX_EXPERTS + e] = gref.fetch_add(count);
      } else {
        slm[MAX_EXPERTS + e] = 0;
      }
      slm[e] = 0;  // reset for use as assignment counter
    }
    item.barrier(sycl::access::fence_space::local_space);

    // (c) Second pass: assign scatter_idx.
    for (int64_t idx = wg_start + lid; idx < wg_end; idx += lsize) {
      const int32_t token_idx = static_cast<int32_t>(idx / topk);
      const int32_t k = static_cast<int32_t>(idx % topk);
      const int32_t src_rank = token_idx / num_tokens_per_rank;
      const int32_t src_token = token_idx % num_tokens_per_rank;
      const int32_t out_idx = token_idx * topk + k;
      const int32_t* src_topk =
          reinterpret_cast<const int32_t*>(topk_rank_ptrs[src_rank]);
      const int32_t expert = src_topk[src_token * topk_storage_stride + k];
      global_topk_idx[out_idx] = expert;
      if (expert < 0 || expert >= num_experts) {
        scatter_idx[out_idx] = -1;
      } else {
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group,
                         sycl::access::address_space::local_space>
            cref(slm[expert]);
        const int32_t pos = cref.fetch_add(1);
        scatter_idx[out_idx] = slm[MAX_EXPERTS + expert] + pos;
      }
    }
  }
};

} // namespace

at::Tensor notify_dispatch(
    const at::Tensor& topk_rank_ptrs,
    at::Tensor global_topk_idx,
    at::Tensor scatter_idx,
    at::Tensor psum_num_recv_tokens_per_expert,
    int64_t num_tokens_per_rank,
    int64_t topk,
    int64_t topk_storage_stride,
    int64_t num_experts,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(
      topk_rank_ptrs.dim() == 1 && topk_rank_ptrs.size(0) == world_size,
      "notify_dispatch: topk_rank_ptrs must be 1D with size == world_size");
  TORCH_CHECK(
      topk_rank_ptrs.scalar_type() == at::kLong,
      "notify_dispatch: topk_rank_ptrs must be int64");
  TORCH_CHECK(
      global_topk_idx.dim() == 2,
      "notify_dispatch: global_topk_idx must be 2D");
  TORCH_CHECK(
      global_topk_idx.scalar_type() == at::kInt,
      "notify_dispatch: global_topk_idx must be int32");
  TORCH_CHECK(
      global_topk_idx.is_contiguous(),
      "notify_dispatch: global_topk_idx must be contiguous");
  TORCH_CHECK(
      scatter_idx.dim() == 2,
      "notify_dispatch: scatter_idx must be 2D");
  TORCH_CHECK(
      scatter_idx.scalar_type() == at::kInt,
      "notify_dispatch: scatter_idx must be int32");
  TORCH_CHECK(
      scatter_idx.is_contiguous(),
      "notify_dispatch: scatter_idx must be contiguous");
  TORCH_CHECK(
      psum_num_recv_tokens_per_expert.dim() == 1,
      "notify_dispatch: psum_num_recv_tokens_per_expert must be 1D");
  TORCH_CHECK(
      psum_num_recv_tokens_per_expert.scalar_type() == at::kInt,
      "notify_dispatch: psum_num_recv_tokens_per_expert must be int32");
  TORCH_CHECK(
      rank >= 0 && rank < world_size,
      "notify_dispatch: rank must be in [0, world_size)");
  TORCH_CHECK(
      num_tokens_per_rank >= 0,
      "notify_dispatch: num_tokens_per_rank must be >= 0");
  TORCH_CHECK(
      topk >= 0,
      "notify_dispatch: topk must be >= 0");
  TORCH_CHECK(
      topk_storage_stride >= topk,
      "notify_dispatch: topk_storage_stride must be >= topk");
  TORCH_CHECK(
      num_experts > 0 && num_experts <= MAX_EXPERTS,
      "notify_dispatch: num_experts must be in (0, ", MAX_EXPERTS, "]");

  const int64_t num_tokens = num_tokens_per_rank * world_size;
  TORCH_CHECK(
      global_topk_idx.size(0) == num_tokens && global_topk_idx.size(1) == topk,
      "notify_dispatch: global_topk_idx shape mismatch");
  TORCH_CHECK(
      scatter_idx.size(0) == num_tokens && scatter_idx.size(1) == topk,
      "notify_dispatch: scatter_idx shape mismatch");

  const int32_t local_start = owner_expert_start(
      static_cast<int32_t>(rank),
      static_cast<int32_t>(num_experts),
      static_cast<int32_t>(world_size));
  const int32_t local_count = owner_expert_count(
      static_cast<int32_t>(rank),
      static_cast<int32_t>(num_experts),
      static_cast<int32_t>(world_size));
  const int32_t local_end = local_start + local_count;

  TORCH_CHECK(
      psum_num_recv_tokens_per_expert.size(0) == local_count,
      "notify_dispatch: psum_num_recv_tokens_per_expert size mismatch");

  if (num_tokens == 0 || topk == 0) {
    global_topk_idx.zero_();
    scatter_idx.zero_();
    psum_num_recv_tokens_per_expert.zero_();
    return scatter_idx;
  }

  c10::Device device(c10::DeviceType::XPU, scatter_idx.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  // Pre-allocate expert_offsets (zeroed by memset).
  auto expert_offsets = at::empty(
      {num_experts}, scatter_idx.options().dtype(at::kInt));
  queue.memset(expert_offsets.data_ptr<int32_t>(), 0,
               num_experts * sizeof(int32_t));

  const int64_t total_pairs =
      static_cast<int64_t>(world_size) * num_tokens_per_rank * topk;

  // --- Kernel 1: count + prefix-sum (single WG, SLM histogram) ---
  auto k1 = NotifyDispatchCountPrefixKernel{
      {},
      topk_rank_ptrs.data_ptr<int64_t>(),
      expert_offsets.data_ptr<int32_t>(),
      psum_num_recv_tokens_per_expert.data_ptr<int32_t>(),
      static_cast<int32_t>(num_tokens_per_rank),
      static_cast<int32_t>(topk),
      static_cast<int32_t>(topk_storage_stride),
      static_cast<int32_t>(num_experts),
      static_cast<int32_t>(world_size),
      local_start,
      local_end,
      total_pairs,
      {}};
  sycl_kernel_submit(
      sycl::range<1>(WG_SIZE), sycl::range<1>(WG_SIZE), queue, k1);

  // --- Kernel 2: assign scatter_idx (multi-WG, SLM reservation) ---
  const int64_t num_wgs = (total_pairs + WG_SIZE - 1) / WG_SIZE;
  const int64_t global_range = num_wgs * WG_SIZE;
  auto k2 = NotifyDispatchAssignKernel{
      {},
      topk_rank_ptrs.data_ptr<int64_t>(),
      global_topk_idx.data_ptr<int32_t>(),
      scatter_idx.data_ptr<int32_t>(),
      expert_offsets.data_ptr<int32_t>(),
      static_cast<int32_t>(num_tokens_per_rank),
      static_cast<int32_t>(topk),
      static_cast<int32_t>(topk_storage_stride),
      static_cast<int32_t>(num_experts),
      static_cast<int32_t>(world_size),
      total_pairs,
      {}};
  sycl_kernel_submit(
      sycl::range<1>(global_range), sycl::range<1>(WG_SIZE), queue, k2);

  return scatter_idx;
}

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "notify_dispatch(Tensor topk_rank_ptrs, Tensor(a!) global_topk_idx, "
      "Tensor(a!) scatter_idx, Tensor(a!) psum_num_recv_tokens_per_expert, "
      "int num_tokens_per_rank, int topk, int topk_storage_stride, "
      "int num_experts, int rank, int world_size) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("notify_dispatch", notify_dispatch);
}
