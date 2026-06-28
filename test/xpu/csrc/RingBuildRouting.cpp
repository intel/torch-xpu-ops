#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <comm/SYCLHelpers.h>

// ---------------------------------------------------------------------------
// ring_build_routing
//
// Dedicated single-kernel routing builder for the ring allgather/permute +
// ring reduce-scatter/unpermute fusion path.  Modeled on
// torch.ops.symm_mem.notify_dispatch_v2, but it buckets per OWNER RANK
// (world_size buckets) instead of per expert, so it directly emits the
// owner-local compacted destination rows that the ring kernels consume.
//
// It reads every rank's topk_idx / topk_weights straight from symmetric
// memory (via the per-rank pointer tables) so no all_gather / host sync is
// needed.  Outputs (all in the global block layout, block r == rank r):
//   global_topk_idx     [num_tokens, topk] int32
//   owner_scatter_idx   [num_tokens, topk] int32  (owner-local absolute row)
//   global_topk_weights [num_tokens, topk] float32
//   rows_per_owner      [world_size] int32 (per-owner total slot counts)
// ---------------------------------------------------------------------------

namespace {

constexpr int32_t WG_SIZE = 256;
constexpr int32_t MAX_OWNERS = 64;

// Contiguous-block expert -> owner-rank map (matches the ring kernels).
inline int32_t owner_of_expert(
    int32_t expert, int32_t num_experts, int32_t world_size) {
  const int32_t base = num_experts / world_size;
  const int32_t rem = num_experts % world_size;
  const int32_t boundary = rem * (base + 1);
  if (expert < boundary)
    return expert / (base + 1);
  return rem + (expert - boundary) / base;
}

// ---------------------------------------------------------------------------
// Single multi-WG kernel (4-phase reservation pattern, mirrors
// NotifyDispatchV2Kernel but on a per-owner histogram):
//   Phase 1: zero SLM histogram (world_size buckets)
//   Phase 2: read expert from symm mem, compute owner, write
//            global_topk_idx / global_topk_weights, local atomic offset
//   Phase 3: reserve chunk from global rows_per_owner (device atomic)
//   Phase 4: fix up owner_scatter_idx with the reserved global base
// ---------------------------------------------------------------------------
struct RingBuildRoutingKernel : __SYCL_KER_CONFIG_CONVENTION__ {
  const int64_t* topk_rank_ptrs;
  const int64_t* weights_rank_ptrs;   // nullable
  int32_t* global_topk_idx;
  int32_t* owner_scatter_idx;
  float* global_topk_weights_out;     // nullable
  int32_t* rows_per_owner;            // [world_size], zero-initialized by caller
  int32_t num_tokens_per_rank;
  int32_t topk;
  int32_t topk_storage_stride;
  int32_t num_experts;
  int32_t world_size;
  int64_t total_pairs;

  // SLM layout: [0..world_size)            = histogram / local counter
  //             [MAX_OWNERS..MAX_OWNERS+ws) = reserved base offset
  sycl::local_accessor<int32_t, 1> slm;

  void sycl_ker_config_convention(sycl::handler& cgh) {
    slm = sycl::local_accessor<int32_t, 1>(2 * MAX_OWNERS, cgh);
  }

  void operator()(sycl::nd_item<1> item) const {
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lsize = static_cast<int32_t>(item.get_local_range(0));
    const int64_t wg_start =
        static_cast<int64_t>(item.get_group(0)) * lsize;
    const int64_t wg_end =
        (wg_start + lsize < total_pairs) ? wg_start + lsize : total_pairs;

    // Phase 1: zero SLM histogram.
    for (int32_t o = lid; o < world_size; o += lsize)
      slm[o] = 0;
    item.barrier(sycl::access::fence_space::local_space);

    // Phase 2: read expert, compute owner, write outputs, local offset.
    int32_t my_owner = -1;
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
      global_topk_idx[my_out_idx] = expert;

      if (expert >= 0 && expert < num_experts) {
        my_owner = owner_of_expert(expert, num_experts, world_size);
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group,
                         sycl::access::address_space::local_space>
            ref(slm[my_owner]);
        owner_scatter_idx[my_out_idx] = ref.fetch_add(1);  // local offset in WG
      } else {
        owner_scatter_idx[my_out_idx] = -1;
      }

      if (weights_rank_ptrs != nullptr) {
        const float* src_weights =
            reinterpret_cast<const float*>(weights_rank_ptrs[src_rank]);
        global_topk_weights_out[my_out_idx] =
            src_weights[src_token * topk_storage_stride + k];
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    // Phase 3: reserve a contiguous chunk from global rows_per_owner.
    for (int32_t o = lid; o < world_size; o += lsize) {
      const int32_t count = slm[o];
      if (count > 0) {
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device>
            gref(rows_per_owner[o]);
        slm[MAX_OWNERS + o] = gref.fetch_add(count);
      } else {
        slm[MAX_OWNERS + o] = 0;
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    // Phase 4: fix up owner_scatter_idx with the reserved global base.
    if (my_owner >= 0 && my_owner < world_size) {
      owner_scatter_idx[my_out_idx] += slm[MAX_OWNERS + my_owner];
    }
  }
};

} // namespace

at::Tensor ring_build_routing(
    const at::Tensor& topk_rank_ptrs,
    at::Tensor global_topk_idx,
    at::Tensor owner_scatter_idx,
    at::Tensor rows_per_owner,
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
      "ring_build_routing: topk_rank_ptrs must be 1D with size == world_size");
  TORCH_CHECK(
      topk_rank_ptrs.scalar_type() == at::kLong,
      "ring_build_routing: topk_rank_ptrs must be int64");
  TORCH_CHECK(
      global_topk_idx.dim() == 2 &&
          global_topk_idx.scalar_type() == at::kInt &&
          global_topk_idx.is_contiguous(),
      "ring_build_routing: global_topk_idx must be 2D contiguous int32");
  TORCH_CHECK(
      owner_scatter_idx.dim() == 2 &&
          owner_scatter_idx.scalar_type() == at::kInt &&
          owner_scatter_idx.is_contiguous(),
      "ring_build_routing: owner_scatter_idx must be 2D contiguous int32");
  TORCH_CHECK(
      rows_per_owner.dim() == 1 && rows_per_owner.size(0) == world_size &&
          rows_per_owner.scalar_type() == at::kInt,
      "ring_build_routing: rows_per_owner must be 1D int32 sized world_size");
  TORCH_CHECK(
      world_size > 0 && world_size <= MAX_OWNERS,
      "ring_build_routing: world_size must be in (0, ", MAX_OWNERS, "]");
  TORCH_CHECK(
      rank >= 0 && rank < world_size,
      "ring_build_routing: rank must be in [0, world_size)");
  TORCH_CHECK(
      topk_storage_stride >= topk,
      "ring_build_routing: topk_storage_stride must be >= topk");
  TORCH_CHECK(
      num_experts > 0, "ring_build_routing: num_experts must be > 0");

  const int64_t num_tokens = num_tokens_per_rank * world_size;
  TORCH_CHECK(
      global_topk_idx.size(0) == num_tokens && global_topk_idx.size(1) == topk,
      "ring_build_routing: global_topk_idx shape mismatch");
  TORCH_CHECK(
      owner_scatter_idx.size(0) == num_tokens &&
          owner_scatter_idx.size(1) == topk,
      "ring_build_routing: owner_scatter_idx shape mismatch");

  if (num_tokens == 0 || topk == 0) {
    global_topk_idx.zero_();
    owner_scatter_idx.zero_();
    rows_per_owner.zero_();
    return owner_scatter_idx;
  }

  c10::Device device(c10::DeviceType::XPU, owner_scatter_idx.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  // Zero rows_per_owner (used as global atomic counter).
  queue.memset(rows_per_owner.data_ptr<int32_t>(), 0,
               world_size * sizeof(int32_t));

  const int64_t total_pairs =
      static_cast<int64_t>(world_size) * num_tokens_per_rank * topk;

  // Optional weights gather pointers.
  const int64_t* w_rank_ptrs = nullptr;
  float* w_out_ptr = nullptr;
  if (weights_rank_ptrs.has_value() && global_topk_weights.has_value()) {
    TORCH_CHECK(
        weights_rank_ptrs->dim() == 1 &&
            weights_rank_ptrs->size(0) == world_size &&
            weights_rank_ptrs->scalar_type() == at::kLong,
        "ring_build_routing: weights_rank_ptrs must be 1D int64 sized world_size");
    TORCH_CHECK(
        global_topk_weights->dim() == 2 &&
            global_topk_weights->size(0) == num_tokens &&
            global_topk_weights->size(1) == topk &&
            global_topk_weights->scalar_type() == at::kFloat &&
            global_topk_weights->is_contiguous(),
        "ring_build_routing: global_topk_weights must be 2D contiguous float32");
    w_rank_ptrs = weights_rank_ptrs->data_ptr<int64_t>();
    w_out_ptr = global_topk_weights->data_ptr<float>();
  }

  const int64_t num_wgs = (total_pairs + WG_SIZE - 1) / WG_SIZE;
  const int64_t global_range = num_wgs * WG_SIZE;
  auto kfn = RingBuildRoutingKernel{
      {},
      topk_rank_ptrs.data_ptr<int64_t>(),
      w_rank_ptrs,
      global_topk_idx.data_ptr<int32_t>(),
      owner_scatter_idx.data_ptr<int32_t>(),
      w_out_ptr,
      rows_per_owner.data_ptr<int32_t>(),
      static_cast<int32_t>(num_tokens_per_rank),
      static_cast<int32_t>(topk),
      static_cast<int32_t>(topk_storage_stride),
      static_cast<int32_t>(num_experts),
      static_cast<int32_t>(world_size),
      total_pairs,
      {}};
  sycl_kernel_submit(
      sycl::range<1>(global_range), sycl::range<1>(WG_SIZE), queue, kfn);

  return owner_scatter_idx;
}

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "ring_build_routing(Tensor topk_rank_ptrs, Tensor(a!) global_topk_idx, "
      "Tensor(a!) owner_scatter_idx, Tensor(a!) rows_per_owner, "
      "int num_tokens_per_rank, int topk, int topk_storage_stride, "
      "int num_experts, int rank, int world_size, "
      "Tensor? weights_rank_ptrs=None, Tensor? global_topk_weights=None) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("ring_build_routing", ring_build_routing);
}
