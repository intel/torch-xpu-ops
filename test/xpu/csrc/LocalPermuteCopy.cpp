#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <climits>
#include <torch/library.h>
#include <comm/SYCLHelpers.h>

#ifndef AT_DISPATCH_FLOAT_AND_BFLOAT16
#define AT_DISPATCH_FLOAT_AND_BFLOAT16(scalar_type, name, ...)         \
  AT_DISPATCH_SWITCH(                                                  \
      scalar_type, name, AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__))
#endif

// Scalar fallback kernel for non-aligned hidden_size.
// Uses scatter_idx to write each (token, k) to its expert-sorted position.
template <typename T>
struct LocalPermuteCopyScalarKernel {
  const T* src_ptr;
  T* dst_ptr;
  const int32_t* scatter_idx_ptr;
  int32_t num_tokens_per_rank;
  int32_t hidden_size;
  int32_t topk;
  int32_t remote_token_offset;

  LocalPermuteCopyScalarKernel(
      const T* src_ptr_,
      T* dst_ptr_,
      const int32_t* scatter_idx_ptr_,
      int32_t num_tokens_per_rank_,
      int32_t hidden_size_,
      int32_t topk_,
      int32_t remote_token_offset_)
      : src_ptr(src_ptr_),
        dst_ptr(dst_ptr_),
        scatter_idx_ptr(scatter_idx_ptr_),
        num_tokens_per_rank(num_tokens_per_rank_),
        hidden_size(hidden_size_),
        topk(topk_),
        remote_token_offset(remote_token_offset_) {}

  void operator()(sycl::nd_item<1> item) const {
    const int32_t idx = static_cast<int32_t>(item.get_global_id(0));
    const int32_t total = num_tokens_per_rank * hidden_size;
    if (idx >= total) return;

    const int32_t h = idx % hidden_size;
    const int32_t local_token_idx = idx / hidden_size;
    const int32_t global_token_idx = remote_token_offset + local_token_idx;
    const T val = src_ptr[local_token_idx * hidden_size + h];
    const int32_t scatter_base = global_token_idx * topk;
    for (int32_t k = 0; k < topk; ++k) {
      int32_t dst_row = scatter_idx_ptr[scatter_base + k];
      dst_ptr[dst_row * hidden_size + h] = val;
    }
  }
};

// Vectorized kernel with scatter_idx-based write positioning.
// Each (token, k) is written to the position given by scatter_idx[global_token_idx * topk + k].
template <typename scalar_t, int VEC_SIZE>
struct LocalPermuteCopyVecKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const scalar_t* src_ptr;
  scalar_t* dst_ptr;
  const int32_t* scatter_idx_ptr;
  int32_t num_tokens_per_rank;
  int32_t hidden_size;
  int32_t topk;
  int32_t remote_token_offset;
  int32_t hidden_vecs;       // hidden_size / VEC_SIZE
  int32_t total;             // num_tokens_per_rank * hidden_vecs

  LocalPermuteCopyVecKernel(
      const scalar_t* src_ptr_,
      scalar_t* dst_ptr_,
      const int32_t* scatter_idx_ptr_,
      int32_t num_tokens_per_rank_,
      int32_t hidden_size_,
      int32_t topk_,
      int32_t remote_token_offset_,
      int32_t hidden_vecs_)
      : src_ptr(src_ptr_),
        dst_ptr(dst_ptr_),
        scatter_idx_ptr(scatter_idx_ptr_),
        num_tokens_per_rank(num_tokens_per_rank_),
        hidden_size(hidden_size_),
        topk(topk_),
        remote_token_offset(remote_token_offset_),
        hidden_vecs(hidden_vecs_),
        total(num_tokens_per_rank_ * hidden_vecs_) {}

  void operator()(sycl::nd_item<1> item) const {
    const int32_t idx = static_cast<int32_t>(item.get_global_id(0));
    if (idx >= total) return;

    const int32_t vec_h = idx % hidden_vecs;
    const int32_t local_token_idx = idx / hidden_vecs;
    const int32_t global_token_idx = remote_token_offset + local_token_idx;

    // Load source vector once
    auto src_vec =
        reinterpret_cast<const vec_t*>(src_ptr + local_token_idx * hidden_size);
    vec_t v = src_vec[vec_h];

    // Store to scattered destinations using scatter_idx
    const int32_t scatter_base = global_token_idx * topk;
    for (int32_t k = 0; k < topk; ++k) {
      int32_t dst_row = scatter_idx_ptr[scatter_base + k];
      auto dst_vec =
          reinterpret_cast<vec_t*>(dst_ptr + dst_row * hidden_size);
      dst_vec[vec_h] = v;
    }
  }
};

at::Tensor local_permute_copy_(
    const at::Tensor& src_hidden,
    const at::Tensor& scatter_idx,
    int64_t remote_token_offset,
    at::Tensor remap_hidden_states) {
  TORCH_CHECK(src_hidden.dim() == 2, "local_permute_copy_: src_hidden must be 2D");
  TORCH_CHECK(scatter_idx.dim() == 2, "local_permute_copy_: scatter_idx must be 2D [num_tokens, topk]");
  TORCH_CHECK(
      scatter_idx.scalar_type() == at::kInt,
      "local_permute_copy_: scatter_idx must be int32");
  TORCH_CHECK(scatter_idx.is_contiguous(), "local_permute_copy_: scatter_idx must be contiguous");
  TORCH_CHECK(
      src_hidden.scalar_type() == remap_hidden_states.scalar_type(),
      "local_permute_copy_: src and remap dtype must match");
  TORCH_CHECK(src_hidden.is_contiguous(), "local_permute_copy_: src_hidden must be contiguous");
  TORCH_CHECK(remap_hidden_states.is_contiguous(), "local_permute_copy_: remap_hidden_states must be contiguous");

  const int64_t num_tokens_per_rank = src_hidden.size(0);
  const int64_t hidden_size = src_hidden.size(1);
  const int64_t num_tokens = scatter_idx.size(0);
  const int64_t topk = scatter_idx.size(1);

  TORCH_CHECK(remote_token_offset >= 0, "local_permute_copy_: remote_token_offset must be >= 0");
  TORCH_CHECK(
      remote_token_offset + num_tokens_per_rank <= num_tokens,
      "local_permute_copy_: remote token range out of bounds");
  TORCH_CHECK(
      remap_hidden_states.size(1) == hidden_size,
      "local_permute_copy_: remap_hidden_states hidden size mismatch");

  if (num_tokens_per_rank == 0) {
    return remap_hidden_states;
  }

  // Runtime int32 overflow guard
  TORCH_CHECK(
      remap_hidden_states.size(0) * hidden_size <= INT32_MAX,
      "local_permute_copy_: total output elements exceed int32 range");

  c10::Device device(c10::DeviceType::XPU, src_hidden.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 128;

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      src_hidden.scalar_type(), "local_permute_copy_", [&]() {
        const int32_t n = static_cast<int32_t>(num_tokens_per_rank);
        const int32_t h = static_cast<int32_t>(hidden_size);
        const int32_t tk = static_cast<int32_t>(topk);
        const int32_t off = static_cast<int32_t>(remote_token_offset);

        if (hidden_size % VEC_SIZE == 0) {
          const int32_t hidden_vecs = h / VEC_SIZE;
          const int64_t total = static_cast<int64_t>(n) * hidden_vecs;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = LocalPermuteCopyVecKernel<scalar_t, VEC_SIZE>(
              src_hidden.data_ptr<scalar_t>(),
              remap_hidden_states.data_ptr<scalar_t>(),
              scatter_idx.data_ptr<int32_t>(),
              n, h, tk, off, hidden_vecs);
          sycl_kernel_submit(sycl::range<1>(blocks * threads), sycl::range<1>(threads), queue, kfn);
        } else {
          // Scalar fallback for non-aligned hidden_size.
          const int64_t total = static_cast<int64_t>(n) * h;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = LocalPermuteCopyScalarKernel<scalar_t>(
              src_hidden.data_ptr<scalar_t>(),
              remap_hidden_states.data_ptr<scalar_t>(),
              scatter_idx.data_ptr<int32_t>(),
              n, h, tk, off);
          sycl_kernel_submit(
              sycl::range<1>(blocks * threads),
              sycl::range<1>(threads),
              queue,
              kfn);
        }
      });

  return remap_hidden_states;
}

// Fused multi-rank kernel: processes all src_ranks in a single launch.
// Eliminates kernel launch overhead when all src data is pre-gathered.
// src_all: [world_size, num_tokens_per_rank, hidden_size]
// scatter_idx: [num_tokens, topk] int32 - expert-sorted output positions
template <typename scalar_t, int VEC_SIZE>
struct LocalPermuteCopyFusedKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const scalar_t* src_ptr;
  scalar_t* dst_ptr;
  const int32_t* scatter_idx_ptr;
  int32_t num_tokens_per_rank;
  int32_t hidden_size;
  int32_t topk;
  int32_t hidden_vecs;
  int32_t tokens_per_rank_mask;
  int32_t tokens_per_rank_shift;
  int32_t total;

  LocalPermuteCopyFusedKernel(
      const scalar_t* src_ptr_,
      scalar_t* dst_ptr_,
      const int32_t* scatter_idx_ptr_,
      int32_t num_tokens_per_rank_,
      int32_t hidden_size_,
      int32_t topk_,
      int32_t hidden_vecs_,
      int32_t tokens_per_rank_mask_,
      int32_t tokens_per_rank_shift_,
      int32_t total_)
      : src_ptr(src_ptr_),
        dst_ptr(dst_ptr_),
        scatter_idx_ptr(scatter_idx_ptr_),
        num_tokens_per_rank(num_tokens_per_rank_),
        hidden_size(hidden_size_),
        topk(topk_),
        hidden_vecs(hidden_vecs_),
        tokens_per_rank_mask(tokens_per_rank_mask_),
        tokens_per_rank_shift(tokens_per_rank_shift_),
        total(total_) {}

  void operator()(sycl::nd_item<1> item) const {
    const int32_t idx = static_cast<int32_t>(item.get_global_id(0));
    if (idx >= total) return;

    // Decompose: idx = (src_rank * num_tokens_per_rank + local_token_idx) * hidden_vecs + vec_h
    const int32_t vec_h = idx % hidden_vecs;
    const int32_t rank_and_token = idx / hidden_vecs;
    const int32_t local_token_idx = rank_and_token & tokens_per_rank_mask;
    const int32_t src_rank = rank_and_token >> tokens_per_rank_shift;

    const int32_t global_token_idx = src_rank * num_tokens_per_rank + local_token_idx;

    // Load source vector once (from src_rank's shard)
    const int32_t src_offset = (src_rank * num_tokens_per_rank + local_token_idx) * hidden_size;
    auto src_vec = reinterpret_cast<const vec_t*>(src_ptr + src_offset);
    vec_t v = src_vec[vec_h];

    // Store to scattered destinations using scatter_idx
    const int32_t scatter_base = global_token_idx * topk;
    for (int32_t k = 0; k < topk; ++k) {
      int32_t dst_row = scatter_idx_ptr[scatter_base + k];
      auto dst_vec =
          reinterpret_cast<vec_t*>(dst_ptr + dst_row * hidden_size);
      dst_vec[vec_h] = v;
    }
  }
};

// Fused variant: permute all ranks in a single kernel launch.
// src_all: [world_size, num_tokens_per_rank, hidden_size], contiguous
// scatter_idx: [num_tokens, topk] int32
at::Tensor local_permute_copy_fused_(
    const at::Tensor& src_all,
    const at::Tensor& scatter_idx,
    at::Tensor remap_hidden_states) {
  TORCH_CHECK(src_all.dim() == 3, "local_permute_copy_fused_: src_all must be 3D [world_size, tokens, hidden]");
  TORCH_CHECK(src_all.is_contiguous(), "local_permute_copy_fused_: src_all must be contiguous");
  TORCH_CHECK(remap_hidden_states.is_contiguous(), "local_permute_copy_fused_: remap must be contiguous");
  TORCH_CHECK(scatter_idx.dim() == 2, "local_permute_copy_fused_: scatter_idx must be 2D [num_tokens, topk]");
  TORCH_CHECK(scatter_idx.scalar_type() == at::kInt, "local_permute_copy_fused_: scatter_idx must be int32");
  TORCH_CHECK(scatter_idx.is_contiguous(), "local_permute_copy_fused_: scatter_idx must be contiguous");

  const int64_t world_size = src_all.size(0);
  const int64_t num_tokens_per_rank = src_all.size(1);
  const int64_t hidden_size = src_all.size(2);
  const int64_t num_tokens = scatter_idx.size(0);
  const int64_t topk = scatter_idx.size(1);

  TORCH_CHECK(num_tokens == world_size * num_tokens_per_rank);
  TORCH_CHECK(remap_hidden_states.size(1) == hidden_size);

  // Require power-of-2 for bitwise decomposition
  TORCH_CHECK((num_tokens_per_rank & (num_tokens_per_rank - 1)) == 0,
      "local_permute_copy_fused_: num_tokens_per_rank must be power of 2");
  TORCH_CHECK(remap_hidden_states.size(0) * hidden_size <= INT32_MAX);

  if (num_tokens_per_rank == 0) {
    return remap_hidden_states;
  }

  c10::Device device(c10::DeviceType::XPU, src_all.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;

  auto log2_po2 = [](int32_t v) -> int32_t {
    int32_t r = 0;
    while (v > 1) { v >>= 1; ++r; }
    return r;
  };

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      src_all.scalar_type(), "local_permute_copy_fused_", [&]() {
        const int32_t n = static_cast<int32_t>(num_tokens_per_rank);
        const int32_t h = static_cast<int32_t>(hidden_size);
        const int32_t tk = static_cast<int32_t>(topk);

        if (hidden_size % VEC_SIZE == 0) {
          const int32_t hidden_vecs = h / VEC_SIZE;
          const int32_t tpr_mask = n - 1;
          const int32_t tpr_shift = log2_po2(n);
          const int32_t total = static_cast<int32_t>(world_size) * n * hidden_vecs;
          const int64_t blocks = (static_cast<int64_t>(total) + threads - 1) / threads;

          auto kfn = LocalPermuteCopyFusedKernel<scalar_t, VEC_SIZE>(
              src_all.data_ptr<scalar_t>(),
              remap_hidden_states.data_ptr<scalar_t>(),
              scatter_idx.data_ptr<int32_t>(),
              n, h, tk, hidden_vecs,
              tpr_mask, tpr_shift, total);
          sycl_kernel_submit(
              sycl::range<1>(blocks * threads),
              sycl::range<1>(threads),
              queue,
              kfn);
        } else {
          // Fallback: call per-rank kernel in a loop
          for (int64_t r = 0; r < world_size; ++r) {
            local_permute_copy_(
                src_all[r], scatter_idx,
                r * num_tokens_per_rank, remap_hidden_states);
          }
        }
      });

  return remap_hidden_states;
}

// ---------------------------------------------------------------------------
// Fused allgather+permute kernel: single kernel reads from all ranks'
// symmetric memory via rank_buffers_ptr and writes ALL (token, k) positions.
//
// Same ring-ordered decomposition as EpDispatch, but without ownership
// filtering — every token contributes to all topk destinations.
//
// Decomposition: idx → (local_token_idx, step, vec_h)
//   vec_h = idx % hidden_vecs           (innermost: coalesced access)
//   step = (idx / hidden_vecs) % world_size  (interleaved: overlap reads)
//   local_token_idx = idx / (hidden_vecs * world_size)
//   src_rank = (rank + step + 1) % world_size
// ---------------------------------------------------------------------------

// Scalar fallback for non-aligned hidden_size.
template <typename T>
struct AllgatherPermuteRingScalarKernel {
  const int64_t* rank_ptrs;
  const int32_t* scatter_idx_ptr;
  T* remap_ptr;
  int64_t num_tokens_per_rank;
  int64_t hidden_size;
  int64_t topk;
  int64_t rank;
  int64_t world_size;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t idx = static_cast<int64_t>(item.get_global_id(0));
    const int64_t total = world_size * num_tokens_per_rank * hidden_size;
    if (idx >= total) return;

    const int64_t h = idx % hidden_size;
    const int64_t pair_idx = idx / hidden_size;
    const int64_t step = pair_idx % world_size;
    const int64_t local_token_idx = pair_idx / world_size;

    const int64_t src_rank = (rank + step + 1) % world_size;
    const int64_t global_token_idx = src_rank * num_tokens_per_rank + local_token_idx;

    const T* src = reinterpret_cast<const T*>(rank_ptrs[src_rank]);
    const T val = src[local_token_idx * hidden_size + h];

    for (int64_t k = 0; k < topk; ++k) {
      const int32_t dst_row = scatter_idx_ptr[global_token_idx * topk + k];
      remap_ptr[static_cast<int64_t>(dst_row) * hidden_size + h] = val;
    }
  }
};

// Vectorized ring-ordered kernel.
template <typename scalar_t, int VEC_SIZE>
struct AllgatherPermuteRingVecKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const int64_t* rank_ptrs;
  const int32_t* scatter_idx_ptr;
  scalar_t* remap_ptr;
  int32_t num_tokens_per_rank;
  int32_t hidden_size;
  int32_t topk;
  int32_t rank;
  int32_t world_size;
  int32_t hidden_vecs;  // hidden_size / VEC_SIZE

  void operator()(sycl::nd_item<1> item) const {
    const int32_t idx = static_cast<int32_t>(item.get_global_id(0));
    const int32_t total = num_tokens_per_rank * world_size * hidden_vecs;
    if (idx >= total) return;

    const int32_t vec_h = idx % hidden_vecs;
    const int32_t pair_idx = idx / hidden_vecs;
    const int32_t step = pair_idx % world_size;
    const int32_t local_token_idx = pair_idx / world_size;

    const int32_t src_rank = (rank + step + 1) % world_size;
    const int32_t global_token_idx = src_rank * num_tokens_per_rank + local_token_idx;

    // Coalesced vectorized read from source rank's memory
    const scalar_t* src = reinterpret_cast<const scalar_t*>(rank_ptrs[src_rank]);
    auto src_vec = reinterpret_cast<const vec_t*>(
        src + static_cast<int64_t>(local_token_idx) * hidden_size);
    vec_t v = src_vec[vec_h];

    // Write to ALL topk destinations
    const int64_t topk_base = static_cast<int64_t>(global_token_idx) * topk;
    for (int32_t k = 0; k < topk; ++k) {
      const int32_t dst_row = scatter_idx_ptr[topk_base + k];
      auto dst_vec = reinterpret_cast<vec_t*>(
          remap_ptr + static_cast<int64_t>(dst_row) * hidden_size);
      dst_vec[vec_h] = v;
    }
  }
};

at::Tensor allgather_permute(
    const at::Tensor& rank_buffers_ptr,
    const at::Tensor& scatter_idx,
    at::Tensor remap_hidden_states,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(
      rank_buffers_ptr.dim() == 1 && rank_buffers_ptr.size(0) == world_size,
      "allgather_permute: rank_buffers_ptr must be 1D with size == world_size");
  TORCH_CHECK(
      rank_buffers_ptr.scalar_type() == at::kLong,
      "allgather_permute: rank_buffers_ptr must be int64");
  TORCH_CHECK(
      scatter_idx.dim() == 2,
      "allgather_permute: scatter_idx must be 2D [num_tokens, topk]");
  TORCH_CHECK(
      scatter_idx.scalar_type() == at::kInt,
      "allgather_permute: scatter_idx must be int32");
  TORCH_CHECK(
      scatter_idx.is_contiguous(),
      "allgather_permute: scatter_idx must be contiguous");
  TORCH_CHECK(
      remap_hidden_states.dim() == 2,
      "allgather_permute: remap_hidden_states must be 2D");
  TORCH_CHECK(
      remap_hidden_states.is_contiguous(),
      "allgather_permute: remap_hidden_states must be contiguous");
  TORCH_CHECK(
      rank >= 0 && rank < world_size,
      "allgather_permute: rank must be in [0, world_size)");

  const int64_t num_tokens = scatter_idx.size(0);
  const int64_t topk = scatter_idx.size(1);
  const int64_t hidden_size = remap_hidden_states.size(1);

  TORCH_CHECK(
      num_tokens % world_size == 0,
      "allgather_permute: num_tokens must be divisible by world_size");
  const int64_t num_tokens_per_rank = num_tokens / world_size;

  TORCH_CHECK(
      remap_hidden_states.size(0) == num_tokens * topk,
      "allgather_permute: remap_hidden_states first dim must be num_tokens * topk");

  if (num_tokens == 0 || topk == 0 || hidden_size == 0) {
    return remap_hidden_states;
  }

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;

  c10::Device device(c10::DeviceType::XPU, remap_hidden_states.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      remap_hidden_states.scalar_type(), "allgather_permute", [&]() {
        if (hidden_size % VEC_SIZE == 0) {
          const int64_t hidden_vecs = hidden_size / VEC_SIZE;
          const int64_t total = world_size * num_tokens_per_rank * hidden_vecs;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = AllgatherPermuteRingVecKernel<scalar_t, VEC_SIZE>{
              rank_buffers_ptr.data_ptr<int64_t>(),
              scatter_idx.data_ptr<int32_t>(),
              remap_hidden_states.data_ptr<scalar_t>(),
              static_cast<int32_t>(num_tokens_per_rank),
              static_cast<int32_t>(hidden_size),
              static_cast<int32_t>(topk),
              static_cast<int32_t>(rank),
              static_cast<int32_t>(world_size),
              static_cast<int32_t>(hidden_vecs)};
          sycl_kernel_submit(
              sycl::range<1>(blocks * threads),
              sycl::range<1>(threads),
              queue,
              kfn);
        } else {
          const int64_t total = world_size * num_tokens_per_rank * hidden_size;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = AllgatherPermuteRingScalarKernel<scalar_t>{
              rank_buffers_ptr.data_ptr<int64_t>(),
              scatter_idx.data_ptr<int32_t>(),
              remap_hidden_states.data_ptr<scalar_t>(),
              num_tokens_per_rank,
              hidden_size,
              topk,
              rank,
              world_size};
          sycl_kernel_submit(
              sycl::range<1>(blocks * threads),
              sycl::range<1>(threads),
              queue,
              kfn);
        }
      });

  return remap_hidden_states;
}

// ---------------------------------------------------------------------------
// Pure allgather kernel: single kernel reads from all ranks' symmetric
// memory via rank_buffers_ptr and writes contiguously to the output buffer.
// No permutation — rank r's data lands at output[r * tokens_per_rank, :].
//
// Same ring-ordered decomposition as allgather_permute for PCIe overlap,
// but without scatter_idx / topk indirection.
//
// Decomposition: idx → (local_token_idx, step, vec_h)
//   vec_h = idx % hidden_vecs
//   step = (idx / hidden_vecs) % world_size
//   local_token_idx = idx / (hidden_vecs * world_size)
//   src_rank = (rank + step + 1) % world_size
//   dst_row  = src_rank * num_tokens_per_rank + local_token_idx
// ---------------------------------------------------------------------------

// Scalar fallback for non-aligned hidden_size.
template <typename T>
struct AllgatherRingScalarKernel {
  const int64_t* rank_ptrs;
  T* dst_ptr;
  int64_t num_tokens_per_rank;
  int64_t hidden_size;
  int64_t rank;
  int64_t world_size;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t idx = static_cast<int64_t>(item.get_global_id(0));
    const int64_t total = world_size * num_tokens_per_rank * hidden_size;
    if (idx >= total) return;

    const int64_t h = idx % hidden_size;
    const int64_t pair_idx = idx / hidden_size;
    const int64_t step = pair_idx % world_size;
    const int64_t local_token_idx = pair_idx / world_size;

    const int64_t src_rank = (rank + step + 1) % world_size;

    const T* src = reinterpret_cast<const T*>(rank_ptrs[src_rank]);
    const T val = src[local_token_idx * hidden_size + h];

    const int64_t dst_row = src_rank * num_tokens_per_rank + local_token_idx;
    dst_ptr[dst_row * hidden_size + h] = val;
  }
};

// Vectorized ring-ordered kernel.
template <typename scalar_t, int VEC_SIZE>
struct AllgatherRingVecKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const int64_t* rank_ptrs;
  scalar_t* dst_ptr;
  int32_t num_tokens_per_rank;
  int32_t hidden_size;
  int32_t rank;
  int32_t world_size;
  int32_t hidden_vecs;  // hidden_size / VEC_SIZE

  void operator()(sycl::nd_item<1> item) const {
    const int32_t idx = static_cast<int32_t>(item.get_global_id(0));
    const int32_t total = num_tokens_per_rank * world_size * hidden_vecs;
    if (idx >= total) return;

    const int32_t vec_h = idx % hidden_vecs;
    const int32_t pair_idx = idx / hidden_vecs;
    const int32_t step = pair_idx % world_size;
    const int32_t local_token_idx = pair_idx / world_size;

    const int32_t src_rank = (rank + step + 1) % world_size;

    // Coalesced vectorized read from source rank's symmetric memory
    const scalar_t* src = reinterpret_cast<const scalar_t*>(rank_ptrs[src_rank]);
    auto src_vec = reinterpret_cast<const vec_t*>(
        src + static_cast<int64_t>(local_token_idx) * hidden_size);
    vec_t v = src_vec[vec_h];

    // Contiguous write: rank r's tokens go to [r * tokens_per_rank, ...]
    const int32_t dst_row = src_rank * num_tokens_per_rank + local_token_idx;
    auto dst_vec = reinterpret_cast<vec_t*>(
        dst_ptr + static_cast<int64_t>(dst_row) * hidden_size);
    dst_vec[vec_h] = v;
  }
};

at::Tensor allgather(
    const at::Tensor& rank_buffers_ptr,
    at::Tensor output,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(
      rank_buffers_ptr.dim() == 1 && rank_buffers_ptr.size(0) == world_size,
      "allgather: rank_buffers_ptr must be 1D with size == world_size");
  TORCH_CHECK(
      rank_buffers_ptr.scalar_type() == at::kLong,
      "allgather: rank_buffers_ptr must be int64");
  TORCH_CHECK(
      output.dim() == 2,
      "allgather: output must be 2D [num_tokens, hidden_size]");
  TORCH_CHECK(
      output.is_contiguous(),
      "allgather: output must be contiguous");
  TORCH_CHECK(
      rank >= 0 && rank < world_size,
      "allgather: rank must be in [0, world_size)");

  const int64_t num_tokens = output.size(0);
  const int64_t hidden_size = output.size(1);

  TORCH_CHECK(
      num_tokens % world_size == 0,
      "allgather: num_tokens must be divisible by world_size");
  const int64_t num_tokens_per_rank = num_tokens / world_size;

  if (num_tokens == 0 || hidden_size == 0) {
    return output;
  }

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;

  c10::Device device(c10::DeviceType::XPU, output.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      output.scalar_type(), "allgather", [&]() {
        if (hidden_size % VEC_SIZE == 0) {
          const int64_t hidden_vecs = hidden_size / VEC_SIZE;
          const int64_t total = world_size * num_tokens_per_rank * hidden_vecs;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = AllgatherRingVecKernel<scalar_t, VEC_SIZE>{
              rank_buffers_ptr.data_ptr<int64_t>(),
              output.data_ptr<scalar_t>(),
              static_cast<int32_t>(num_tokens_per_rank),
              static_cast<int32_t>(hidden_size),
              static_cast<int32_t>(rank),
              static_cast<int32_t>(world_size),
              static_cast<int32_t>(hidden_vecs)};
          sycl_kernel_submit(
              sycl::range<1>(blocks * threads),
              sycl::range<1>(threads),
              queue,
              kfn);
        } else {
          const int64_t total = world_size * num_tokens_per_rank * hidden_size;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = AllgatherRingScalarKernel<scalar_t>{
              rank_buffers_ptr.data_ptr<int64_t>(),
              output.data_ptr<scalar_t>(),
              num_tokens_per_rank,
              hidden_size,
              rank,
              world_size};
          sycl_kernel_submit(
              sycl::range<1>(blocks * threads),
              sycl::range<1>(threads),
              queue,
              kfn);
        }
      });

  return output;
}

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "local_permute_copy_(Tensor src_hidden, Tensor scatter_idx, int remote_token_offset, Tensor(a!) remap_hidden_states) -> Tensor(a!)");
  m.def(
      "local_permute_copy_fused_(Tensor src_all, Tensor scatter_idx, Tensor(a!) remap_hidden_states) -> Tensor(a!)");
  m.def(
      "allgather_permute(Tensor rank_buffers_ptr, Tensor scatter_idx, "
      "Tensor(a!) remap_hidden_states, int rank, int world_size) -> Tensor(a!)");
  m.def(
      "allgather(Tensor rank_buffers_ptr, "
      "Tensor(a!) output, int rank, int world_size) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("local_permute_copy_", local_permute_copy_);
  m.impl("local_permute_copy_fused_", local_permute_copy_fused_);
  m.impl("allgather_permute", allgather_permute);
  m.impl("allgather", allgather);
}
