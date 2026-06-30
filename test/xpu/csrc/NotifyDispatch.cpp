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
  const int64_t* weights_rank_ptrs;      // nullable: symm_mem pointers to per-rank weights
  float* global_topk_weights_out;        // nullable: output buffer for gathered weights

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

    // (c) Second pass: assign scatter_idx and optionally gather weights.
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
      // Gather weights from symm_mem (same access pattern as topk_idx).
      if (weights_rank_ptrs != nullptr) {
        const float* src_weights =
            reinterpret_cast<const float*>(weights_rank_ptrs[src_rank]);
        global_topk_weights_out[out_idx] =
            src_weights[src_token * topk_storage_stride + k];
      }
    }
  }
};

// ---------------------------------------------------------------------------
// V2 Kernel: single multi-WG kernel that merges K1 (count+prefix-sum) and
// K2 (assign scatter_idx) into one pass over symmetric memory.
//
// Follows vllm RowsPerExpertCount 4-phase pattern:
//   Phase 1: zero SLM histogram
//   Phase 2: local atomic count + assign local offset + write global_topk_idx/weights
//   Phase 3: global atomic reserve chunk from rows_per_expert
//   Phase 4: fix up scatter_idx with global base
//
// Output scatter_idx values are EXPERT-RELATIVE (not absolute).
// Downstream v2 kernels convert to absolute via SLM prefix-sum of rows_per_expert.
// ---------------------------------------------------------------------------
struct NotifyDispatchV2Kernel : __SYCL_KER_CONFIG_CONVENTION__ {
  const int64_t* topk_rank_ptrs;
  int32_t* global_topk_idx;
  int32_t* scatter_idx;
  int32_t* rows_per_expert;      // [num_experts], zero-initialized by caller
  int32_t num_tokens_per_rank;
  int32_t topk;
  int32_t topk_storage_stride;
  int32_t num_experts;
  int32_t world_size;
  int64_t total_pairs;
  const int64_t* weights_rank_ptrs;      // nullable
  float* global_topk_weights_out;        // nullable
  const int64_t* scale_rank_ptrs;        // nullable: per-rank [num_tokens_per_rank] f32
  float* global_scale_out;               // nullable: [num_tokens] gathered per-token scale

  // SLM layout: [0..num_experts) = histogram/counter
  //             [MAX_EXPERTS..2*MAX_EXPERTS) = reserved base offset
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

    // Phase 1: Zero SLM histogram
    for (int32_t e = lid; e < num_experts; e += lsize)
      slm[e] = 0;
    item.barrier(sycl::access::fence_space::local_space);

    // Phase 2: Count + assign local offset + write global_topk_idx/weights.
    // Each thread caches its expert_id for reuse in Phase 4.
    int32_t my_expert = -1;
    int32_t my_out_idx = -1;
    const int64_t my_idx = wg_start + lid;
    if (my_idx < wg_end) {
      const int32_t token_idx = static_cast<int32_t>(my_idx / topk);
      const int32_t k = static_cast<int32_t>(my_idx % topk);
      const int32_t src_rank = token_idx / num_tokens_per_rank;
      const int32_t src_token = token_idx % num_tokens_per_rank;
      my_out_idx = token_idx * topk + k;

      const int32_t* src_topk =
          reinterpret_cast<const int32_t*>(topk_rank_ptrs[src_rank]);
      const int32_t expert = src_topk[src_token * topk_storage_stride + k];
      my_expert = expert;
      global_topk_idx[my_out_idx] = expert;

      if (expert >= 0 && expert < num_experts) {
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group,
                         sycl::access::address_space::local_space>
            ref(slm[expert]);
        scatter_idx[my_out_idx] = ref.fetch_add(1);  // local offset within WG
      } else {
        scatter_idx[my_out_idx] = -1;
      }

      // Gather weights from symm_mem
      if (weights_rank_ptrs != nullptr) {
        const float* src_weights =
            reinterpret_cast<const float*>(weights_rank_ptrs[src_rank]);
        global_topk_weights_out[my_out_idx] =
            src_weights[src_token * topk_storage_stride + k];
      }

      // Gather per-token scale from symm_mem.  The scale is one value per token
      // (not per top-k slot), so only the k==0 lane writes it.
      if (scale_rank_ptrs != nullptr && k == 0) {
        const float* src_scale =
            reinterpret_cast<const float*>(scale_rank_ptrs[src_rank]);
        global_scale_out[token_idx] = src_scale[src_token];
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    // Phase 3: Reserve chunk from global rows_per_expert
    for (int32_t e = lid; e < num_experts; e += lsize) {
      const int32_t count = slm[e];
      if (count > 0) {
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device>
            gref(rows_per_expert[e]);
        slm[MAX_EXPERTS + e] = gref.fetch_add(count);
      } else {
        slm[MAX_EXPERTS + e] = 0;
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    // Phase 4: Fix up scatter_idx with global base offset
    if (my_expert >= 0 && my_expert < num_experts) {
      scatter_idx[my_out_idx] += slm[MAX_EXPERTS + my_expert];
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
    int64_t world_size,
    const std::optional<at::Tensor>& weights_rank_ptrs,
    const std::optional<at::Tensor>& global_topk_weights) {
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

  // Extract optional weights pointers for K2.
  const int64_t* w_rank_ptrs = nullptr;
  float* w_out_ptr = nullptr;
  if (weights_rank_ptrs.has_value() && global_topk_weights.has_value()) {
    TORCH_CHECK(
        weights_rank_ptrs->dim() == 1 && weights_rank_ptrs->size(0) == world_size,
        "notify_dispatch: weights_rank_ptrs must be 1D with size == world_size");
    TORCH_CHECK(
        weights_rank_ptrs->scalar_type() == at::kLong,
        "notify_dispatch: weights_rank_ptrs must be int64");
    TORCH_CHECK(
        global_topk_weights->dim() == 2 &&
        global_topk_weights->size(0) == num_tokens &&
        global_topk_weights->size(1) == topk,
        "notify_dispatch: global_topk_weights shape mismatch");
    TORCH_CHECK(
        global_topk_weights->scalar_type() == at::kFloat,
        "notify_dispatch: global_topk_weights must be float32");
    TORCH_CHECK(
        global_topk_weights->is_contiguous(),
        "notify_dispatch: global_topk_weights must be contiguous");
    w_rank_ptrs = weights_rank_ptrs->data_ptr<int64_t>();
    w_out_ptr = global_topk_weights->data_ptr<float>();
  }

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
      w_rank_ptrs,
      w_out_ptr,
      {}};
  sycl_kernel_submit(
      sycl::range<1>(global_range), sycl::range<1>(WG_SIZE), queue, k2);

  return scatter_idx;
}

at::Tensor notify_dispatch_v2(
    const at::Tensor& topk_rank_ptrs,
    at::Tensor global_topk_idx,
    at::Tensor scatter_idx,
    at::Tensor rows_per_expert,
    int64_t num_tokens_per_rank,
    int64_t topk,
    int64_t topk_storage_stride,
    int64_t num_experts,
    int64_t rank,
    int64_t world_size,
    const std::optional<at::Tensor>& weights_rank_ptrs,
    const std::optional<at::Tensor>& global_topk_weights,
    const std::optional<at::Tensor>& scale_rank_ptrs,
    const std::optional<at::Tensor>& global_scale) {
  TORCH_CHECK(
      topk_rank_ptrs.dim() == 1 && topk_rank_ptrs.size(0) == world_size,
      "notify_dispatch_v2: topk_rank_ptrs must be 1D with size == world_size");
  TORCH_CHECK(
      topk_rank_ptrs.scalar_type() == at::kLong,
      "notify_dispatch_v2: topk_rank_ptrs must be int64");
  TORCH_CHECK(
      global_topk_idx.dim() == 2,
      "notify_dispatch_v2: global_topk_idx must be 2D");
  TORCH_CHECK(
      global_topk_idx.scalar_type() == at::kInt,
      "notify_dispatch_v2: global_topk_idx must be int32");
  TORCH_CHECK(
      global_topk_idx.is_contiguous(),
      "notify_dispatch_v2: global_topk_idx must be contiguous");
  TORCH_CHECK(
      scatter_idx.dim() == 2,
      "notify_dispatch_v2: scatter_idx must be 2D");
  TORCH_CHECK(
      scatter_idx.scalar_type() == at::kInt,
      "notify_dispatch_v2: scatter_idx must be int32");
  TORCH_CHECK(
      scatter_idx.is_contiguous(),
      "notify_dispatch_v2: scatter_idx must be contiguous");
  TORCH_CHECK(
      rows_per_expert.dim() == 1 && rows_per_expert.size(0) == num_experts,
      "notify_dispatch_v2: rows_per_expert must be 1D with size == num_experts");
  TORCH_CHECK(
      rows_per_expert.scalar_type() == at::kInt,
      "notify_dispatch_v2: rows_per_expert must be int32");
  TORCH_CHECK(
      rank >= 0 && rank < world_size,
      "notify_dispatch_v2: rank must be in [0, world_size)");
  TORCH_CHECK(
      num_tokens_per_rank >= 0,
      "notify_dispatch_v2: num_tokens_per_rank must be >= 0");
  TORCH_CHECK(
      topk >= 0,
      "notify_dispatch_v2: topk must be >= 0");
  TORCH_CHECK(
      topk_storage_stride >= topk,
      "notify_dispatch_v2: topk_storage_stride must be >= topk");
  TORCH_CHECK(
      num_experts > 0 && num_experts <= MAX_EXPERTS,
      "notify_dispatch_v2: num_experts must be in (0, ", MAX_EXPERTS, "]");

  const int64_t num_tokens = num_tokens_per_rank * world_size;
  TORCH_CHECK(
      global_topk_idx.size(0) == num_tokens && global_topk_idx.size(1) == topk,
      "notify_dispatch_v2: global_topk_idx shape mismatch");
  TORCH_CHECK(
      scatter_idx.size(0) == num_tokens && scatter_idx.size(1) == topk,
      "notify_dispatch_v2: scatter_idx shape mismatch");

  if (num_tokens == 0 || topk == 0) {
    global_topk_idx.zero_();
    scatter_idx.zero_();
    rows_per_expert.zero_();
    return scatter_idx;
  }

  c10::Device device(c10::DeviceType::XPU, scatter_idx.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  // Zero rows_per_expert (used as global atomic counter)
  queue.memset(rows_per_expert.data_ptr<int32_t>(), 0,
               num_experts * sizeof(int32_t));

  const int64_t total_pairs =
      static_cast<int64_t>(world_size) * num_tokens_per_rank * topk;

  // Extract optional weights pointers
  const int64_t* w_rank_ptrs = nullptr;
  float* w_out_ptr = nullptr;
  if (weights_rank_ptrs.has_value() && global_topk_weights.has_value()) {
    TORCH_CHECK(
        weights_rank_ptrs->dim() == 1 && weights_rank_ptrs->size(0) == world_size,
        "notify_dispatch_v2: weights_rank_ptrs must be 1D with size == world_size");
    TORCH_CHECK(
        weights_rank_ptrs->scalar_type() == at::kLong,
        "notify_dispatch_v2: weights_rank_ptrs must be int64");
    TORCH_CHECK(
        global_topk_weights->dim() == 2 &&
        global_topk_weights->size(0) == num_tokens &&
        global_topk_weights->size(1) == topk,
        "notify_dispatch_v2: global_topk_weights shape mismatch");
    TORCH_CHECK(
        global_topk_weights->scalar_type() == at::kFloat,
        "notify_dispatch_v2: global_topk_weights must be float32");
    TORCH_CHECK(
        global_topk_weights->is_contiguous(),
        "notify_dispatch_v2: global_topk_weights must be contiguous");
    w_rank_ptrs = weights_rank_ptrs->data_ptr<int64_t>();
    w_out_ptr = global_topk_weights->data_ptr<float>();
  }

  // Extract optional per-token scale pointers
  const int64_t* s_rank_ptrs = nullptr;
  float* s_out_ptr = nullptr;
  if (scale_rank_ptrs.has_value() && global_scale.has_value()) {
    TORCH_CHECK(
        scale_rank_ptrs->dim() == 1 && scale_rank_ptrs->size(0) == world_size,
        "notify_dispatch_v2: scale_rank_ptrs must be 1D with size == world_size");
    TORCH_CHECK(
        scale_rank_ptrs->scalar_type() == at::kLong,
        "notify_dispatch_v2: scale_rank_ptrs must be int64");
    TORCH_CHECK(
        global_scale->dim() == 1 && global_scale->size(0) == num_tokens,
        "notify_dispatch_v2: global_scale must be 1D with size == num_tokens");
    TORCH_CHECK(
        global_scale->scalar_type() == at::kFloat,
        "notify_dispatch_v2: global_scale must be float32");
    TORCH_CHECK(
        global_scale->is_contiguous(),
        "notify_dispatch_v2: global_scale must be contiguous");
    s_rank_ptrs = scale_rank_ptrs->data_ptr<int64_t>();
    s_out_ptr = global_scale->data_ptr<float>();
  }

  // Single multi-WG kernel
  const int64_t num_wgs = (total_pairs + WG_SIZE - 1) / WG_SIZE;
  const int64_t global_range = num_wgs * WG_SIZE;
  auto kfn = NotifyDispatchV2Kernel{
      {},
      topk_rank_ptrs.data_ptr<int64_t>(),
      global_topk_idx.data_ptr<int32_t>(),
      scatter_idx.data_ptr<int32_t>(),
      rows_per_expert.data_ptr<int32_t>(),
      static_cast<int32_t>(num_tokens_per_rank),
      static_cast<int32_t>(topk),
      static_cast<int32_t>(topk_storage_stride),
      static_cast<int32_t>(num_experts),
      static_cast<int32_t>(world_size),
      total_pairs,
      w_rank_ptrs,
      w_out_ptr,
      s_rank_ptrs,
      s_out_ptr,
      {}};
  sycl_kernel_submit(
      sycl::range<1>(global_range), sycl::range<1>(WG_SIZE), queue, kfn);

  return scatter_idx;
}

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "notify_dispatch(Tensor topk_rank_ptrs, Tensor(a!) global_topk_idx, "
      "Tensor(a!) scatter_idx, Tensor(a!) psum_num_recv_tokens_per_expert, "
      "int num_tokens_per_rank, int topk, int topk_storage_stride, "
      "int num_experts, int rank, int world_size, "
      "Tensor? weights_rank_ptrs=None, Tensor? global_topk_weights=None) -> Tensor(a!)");
  m.def(
      "notify_dispatch_v2(Tensor topk_rank_ptrs, Tensor(a!) global_topk_idx, "
      "Tensor(a!) scatter_idx, Tensor(a!) rows_per_expert, "
      "int num_tokens_per_rank, int topk, int topk_storage_stride, "
      "int num_experts, int rank, int world_size, "
      "Tensor? weights_rank_ptrs=None, Tensor? global_topk_weights=None, "
      "Tensor? scale_rank_ptrs=None, Tensor? global_scale=None) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("notify_dispatch", notify_dispatch);
  m.impl("notify_dispatch_v2", notify_dispatch_v2);
}
