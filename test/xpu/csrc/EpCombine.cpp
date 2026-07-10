#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <comm/SYCLHelpers.h>
#include <cstdlib>

#ifndef AT_DISPATCH_FLOAT_AND_BFLOAT16
#define AT_DISPATCH_FLOAT_AND_BFLOAT16(scalar_type, name, ...)         \
  AT_DISPATCH_SWITCH(                                                  \
      scalar_type, name, AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__))
#endif

// The bandwidth-critical remote store pushes the aggregated partial into the
// target rank's receive buffer over the cross-GPU link (Xe-Link / PCIe). Use an
// Intel GPU LSC store with explicit cache control (L1WB_L3WB) so contiguous
// remote writes get combined through L3 into larger burst transactions instead
// of many small write-through transactions (same lever measured ~18% faster on
// the reduce-scatter push path). Enabled by default; define
// EPCOMBINE_NO_LSC_STORE to fall back to the plain vectorized store.
#if !defined(EPCOMBINE_LSC_STORE) && !defined(EPCOMBINE_NO_LSC_STORE)
#define EPCOMBINE_LSC_STORE 1
#endif

#if defined(EPCOMBINE_LSC_STORE) && defined(__SYCL_DEVICE_ONLY__) && \
    defined(__SPIR__)
typedef uint32_t ep_lsc_u4 __attribute__((ext_vector_type(4)));
enum EpLscStcc { EP_LSC_STCC_L1WB_L3WB = 7 };
SYCL_EXTERNAL extern "C" void __builtin_IB_lsc_store_global_uint4(
    __attribute__((opencl_global)) ep_lsc_u4* base,
    int off,
    ep_lsc_u4 val,
    enum EpLscStcc cc);
#endif

// Remote store of a POD vector to the peer buffer. Implemented inline at the
// kernel store site (see EpCombineRingWGKernel) via EP_VEC_STORE so the LSC
// builtin is called directly from the SYCL kernel body (a free function called
// straight from a kernel would require SYCL_EXTERNAL). When LSC is available
// the vector is split into 16-byte (uint4) chunks, each written with L1WB_L3WB
// cache control so the peer-directed writes coalesce through L3.
// `sizeof(Vec)` is a multiple of 16 for every enabled combination
// (bf16x8=16B, bf16x16=32B, fp32x8=32B, fp32x16=64B).
#if defined(EPCOMBINE_LSC_STORE) && defined(__SYCL_DEVICE_ONLY__) && \
    defined(__SPIR__)
#define EP_VEC_STORE(DST_PTR, VEC_VAL)                                        \
  do {                                                                        \
    using _EpVecT = std::remove_reference_t<decltype(VEC_VAL)>;               \
    if constexpr (sizeof(_EpVecT) % 16 == 0) {                                \
      const ep_lsc_u4* _ep_s = reinterpret_cast<const ep_lsc_u4*>(&(VEC_VAL));\
      _Pragma("unroll")                                                       \
      for (int _ep_i = 0; _ep_i < static_cast<int>(sizeof(_EpVecT) / 16);     \
           ++_ep_i) {                                                         \
        __builtin_IB_lsc_store_global_uint4(                                  \
            ((__attribute__((opencl_global)) ep_lsc_u4*)(DST_PTR)) + _ep_i,   \
            0, _ep_s[_ep_i], EP_LSC_STCC_L1WB_L3WB);                          \
      }                                                                       \
    } else {                                                                  \
      *(DST_PTR) = (VEC_VAL);                                                 \
    }                                                                         \
  } while (0)
#else
#define EP_VEC_STORE(DST_PTR, VEC_VAL) (*(DST_PTR) = (VEC_VAL))
#endif

// TP+EP combine kernel (reverse of ep_dispatch, ring-ordered push).
//
// ep_dispatch: ring-ordered PULL from remote hidden → ownership check → write to local remap
// ep_combine:  ring-ordered PUSH: local aggregate → write to remote output buffer
//
// Algorithm:
//   For each target_rank (ring-ordered):
//     For each token belonging to target_rank:
//       Aggregate: sum topk_weights[token,k] * local_expert_output[scatter_idx[token,k]]
//                  (only for k where this rank owns the expert)
//       Push: write partial result to target_rank's receive buffer at this rank's slot
//
//   After barrier: each rank sums received contributions from all EP ranks.
//
// Ring ordering ensures that at each step, adjacent work groups write to
// DIFFERENT target ranks, spreading PCIe write traffic evenly.
//
// Decomposition: idx → (local_token_idx, step, vec_h)
//   vec_h = idx % hidden_vecs           (innermost: coalesced write)
//   step = (idx / hidden_vecs) % world_size  (interleaved: spread writes)
//   local_token_idx = idx / (hidden_vecs * world_size)
//   target_rank = (rank + step + 1) % world_size
//
// Workspace layout on each rank: [world_size, num_tokens_per_rank, hidden]
//   slot[i] = partial contribution FROM rank i for this rank's tokens.
//   rank_output_ptrs[r] points to rank r's receive buffer base.
//   This rank writes to rank_output_ptrs[target_rank] at offset
//   rank * num_tokens_per_rank * hidden.

// Scalar fallback for non-aligned hidden_size.
template <typename T>
struct EpCombineRingScalarKernel {
  const T* expert_output_ptr;       // [num_tokens * topk, hidden] local expert results
  const int64_t* rank_output_ptrs;  // [world_size] pointers to each rank's receive buffer
  const int32_t* topk_idx_ptr;
  const int32_t* scatter_idx_ptr;
  const float* topk_weights_ptr;
  int64_t num_tokens_per_rank;
  int64_t hidden_size;
  int64_t topk;
  int64_t rank;
  int64_t world_size;
  int32_t base_experts;
  int32_t rem_experts;
  int32_t boundary;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t idx = static_cast<int64_t>(item.get_global_id(0));
    const int64_t total = world_size * num_tokens_per_rank * hidden_size;
    if (idx >= total) return;

    const int64_t h = idx % hidden_size;
    const int64_t pair_idx = idx / hidden_size;
    const int64_t step = pair_idx % world_size;
    const int64_t local_token_idx = pair_idx / world_size;

    // Ring ordering: spread writes across target ranks
    const int64_t target_rank = (rank + step + 1) % world_size;
    const int64_t global_token_idx = target_rank * num_tokens_per_rank + local_token_idx;

    // Ownership pre-check: skip if this rank doesn't own any expert for this token.
    // Caller must pre-zero the receive buffer so skipped slots remain zero.
    const int64_t topk_base = global_token_idx * topk;
    bool has_owned = false;
    for (int64_t k = 0; k < topk; ++k) {
      const int32_t expert = topk_idx_ptr[topk_base + k];
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      if (owner == static_cast<int32_t>(rank)) { has_owned = true; break; }
    }
    if (!has_owned) return;

    // Compute weighted partial sum from local expert_output (only owned experts)
    float acc = 0.0f;
    for (int64_t k = 0; k < topk; ++k) {
      const int32_t expert = topk_idx_ptr[topk_base + k];
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      if (owner == static_cast<int32_t>(rank)) {
        const float weight = topk_weights_ptr[topk_base + k];
        const int32_t src_row = scatter_idx_ptr[topk_base + k];
        acc += weight * static_cast<float>(
            expert_output_ptr[static_cast<int64_t>(src_row) * hidden_size + h]);
      }
    }

    // Push to target_rank's receive buffer at this rank's slot
    // Layout: [world_size, num_tokens_per_rank, hidden]
    T* target_buf = reinterpret_cast<T*>(rank_output_ptrs[target_rank]);
    const int64_t dst_offset =
        (rank * num_tokens_per_rank + local_token_idx) * hidden_size + h;
    target_buf[dst_offset] = static_cast<T>(acc);
  }
};

// Aligned POD vector so the compiler emits a single wide load/store transaction
// (16 bytes for bf16 x8, 32 bytes for fp32 x8) instead of per-element accesses.
// Wide, coalesced transactions are critical for remote (PCIe) write throughput.
template <typename scalar_t, int VEC_SIZE>
struct alignas(sizeof(scalar_t) * VEC_SIZE) EpVec {
  scalar_t data[VEC_SIZE];
};

constexpr int kEpCombineMaxTopK = 32;

// Work-group-per-token vectorized ring-ordered push kernel.
//
// One work-group handles one (target-token) row of the hidden dimension.
// Threads in the group stride over the hidden vectors, issuing wide vectorized
// loads from LOCAL expert_output and wide vectorized stores to the target rank's
// receive buffer.
//
// Key optimizations (mirror flashinfer moeA2ACombineKernel):
// 1. One WG per token: ownership is resolved ONCE per token (not once per
//    hidden element as in the old flat kernel), and the whole row's writes are
//    contiguous, maximizing PCIe burst efficiency.
// 2. Wide vectorized load/store (single 16B/32B transactions).
// 3. Ring ordering: adjacent work-groups target DIFFERENT ranks, spreading
//    PCIe write bandwidth across all interconnect links.
// 4. Ownership pre-check: skip the whole row if no expert is owned (caller
//    pre-zeroes the receive buffer so skipped slots stay zero).
// 5. Float accumulator for precision with bfloat16 data.
template <typename scalar_t, int VEC_SIZE>
struct EpCombineRingWGKernel {
  const scalar_t* expert_output_ptr;
  const int64_t* rank_output_ptrs;
  const int32_t* topk_idx_ptr;
  const int32_t* scatter_idx_ptr;
  const float* topk_weights_ptr;
  int32_t num_tokens_per_rank;
  int32_t hidden_size;
  int32_t topk;
  int32_t rank;
  int32_t world_size;
  int32_t hidden_vecs;  // hidden_size / VEC_SIZE
  int32_t base_experts;
  int32_t rem_experts;
  int32_t boundary;
  int32_t write_empty;  // if true, always write (zeros) so caller need not pre-zero
  int32_t no_compute;   // debug: skip local read/accumulate, write zeros (isolate write BW)

  void operator()(sycl::nd_item<1> item) const {
    using Vec = EpVec<scalar_t, VEC_SIZE>;

    const int32_t wg = static_cast<int32_t>(item.get_group(0));
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t nthreads = static_cast<int32_t>(item.get_local_range(0));

    // Ring ordering: consecutive work-groups map to consecutive steps, so
    // adjacent groups push to different target ranks.
    const int32_t step = wg % world_size;
    const int32_t local_token_idx = wg / world_size;
    if (local_token_idx >= num_tokens_per_rank) return;

    const int32_t target_rank = (rank + step + 1) % world_size;
    const int32_t global_token_idx = target_rank * num_tokens_per_rank + local_token_idx;
    const int64_t topk_base = static_cast<int64_t>(global_token_idx) * topk;

    // Resolve ownership ONCE for this token: collect owned (weight, src_row).
    float owned_w[kEpCombineMaxTopK];
    int32_t owned_row[kEpCombineMaxTopK];
    int32_t num_owned = 0;
    for (int32_t k = 0; k < topk; ++k) {
      const int32_t expert = topk_idx_ptr[topk_base + k];
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      if (owner == rank) {
        owned_w[num_owned] = topk_weights_ptr[topk_base + k];
        owned_row[num_owned] = scatter_idx_ptr[topk_base + k];
        ++num_owned;
      }
    }
    if (num_owned == 0 && !write_empty) return;  // receive buffer pre-zeroed by caller

    scalar_t* target_buf = reinterpret_cast<scalar_t*>(rank_output_ptrs[target_rank]);
    scalar_t* dst_row = target_buf +
        (static_cast<int64_t>(rank) * num_tokens_per_rank + local_token_idx) *
            hidden_size;

    for (int32_t vh = lid; vh < hidden_vecs; vh += nthreads) {
      const int32_t h_start = vh * VEC_SIZE;
      float acc[VEC_SIZE];
      #pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) acc[i] = 0.0f;

      for (int32_t j = 0; j < num_owned && !no_compute; ++j) {
        const float weight = owned_w[j];
        const Vec src = *reinterpret_cast<const Vec*>(
            expert_output_ptr +
            static_cast<int64_t>(owned_row[j]) * hidden_size + h_start);
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          acc[i] += weight * static_cast<float>(src.data[i]);
        }
      }

      Vec out;
      #pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        out.data[i] = static_cast<scalar_t>(acc[i]);
      }
      EP_VEC_STORE(reinterpret_cast<Vec*>(dst_row + h_start), out);
    }
  }
};

at::Tensor ep_combine(
    const at::Tensor& expert_output,
    const at::Tensor& rank_output_ptrs,
    const at::Tensor& topk_idx,
    const at::Tensor& scatter_idx,
    const at::Tensor& topk_weights,
    at::Tensor output,
    int64_t num_experts,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(
      rank_output_ptrs.dim() == 1 && rank_output_ptrs.size(0) == world_size,
      "ep_combine: rank_output_ptrs must be 1D with size == world_size");
  TORCH_CHECK(
      rank_output_ptrs.scalar_type() == at::kLong,
      "ep_combine: rank_output_ptrs must be int64");
  TORCH_CHECK(
      expert_output.dim() == 2,
      "ep_combine: expert_output must be 2D [num_tokens * topk, hidden]");
  TORCH_CHECK(expert_output.is_contiguous());
  TORCH_CHECK(topk_idx.dim() == 2, "ep_combine: topk_idx must be 2D");
  TORCH_CHECK(
      topk_idx.scalar_type() == at::kInt,
      "ep_combine: topk_idx must be int32");
  TORCH_CHECK(topk_idx.is_contiguous());
  TORCH_CHECK(
      scatter_idx.dim() == 2 && scatter_idx.size(0) == topk_idx.size(0) &&
          scatter_idx.size(1) == topk_idx.size(1),
      "ep_combine: scatter_idx must be 2D with same shape as topk_idx");
  TORCH_CHECK(
      scatter_idx.scalar_type() == at::kInt,
      "ep_combine: scatter_idx must be int32");
  TORCH_CHECK(scatter_idx.is_contiguous());
  TORCH_CHECK(
      topk_weights.dim() == 2 && topk_weights.size(0) == topk_idx.size(0) &&
          topk_weights.size(1) == topk_idx.size(1),
      "ep_combine: topk_weights must be 2D with same shape as topk_idx");
  TORCH_CHECK(
      topk_weights.scalar_type() == at::kFloat,
      "ep_combine: topk_weights must be float32");
  TORCH_CHECK(topk_weights.is_contiguous());
  TORCH_CHECK(output.dim() == 2, "ep_combine: output must be 2D");
  TORCH_CHECK(output.is_contiguous());
  TORCH_CHECK(rank >= 0 && rank < world_size);

  const int64_t num_tokens = topk_idx.size(0);
  const int64_t topk = topk_idx.size(1);
  const int64_t hidden_size = expert_output.size(1);

  TORCH_CHECK(
      topk <= kEpCombineMaxTopK,
      "ep_combine: topk exceeds kEpCombineMaxTopK");

  TORCH_CHECK(
      num_tokens % world_size == 0,
      "ep_combine: num_tokens must be divisible by world_size");
  const int64_t num_tokens_per_rank = num_tokens / world_size;

  TORCH_CHECK(
      output.size(0) == num_tokens_per_rank,
      "ep_combine: output first dim must be num_tokens_per_rank");
  TORCH_CHECK(
      output.size(1) == hidden_size,
      "ep_combine: output hidden size must match expert_output");

  if (num_tokens == 0 || topk == 0 || hidden_size == 0) {
    return output;
  }

  // Precompute ownership constants on host
  const int32_t base_experts = static_cast<int32_t>(num_experts / world_size);
  const int32_t rem_experts = static_cast<int32_t>(num_experts % world_size);
  const int32_t boundary = rem_experts * (base_experts + 1);

  // Tunable via env for sweeping: EPCOMBINE_VEC (8|16), EPCOMBINE_THREADS.
  int vec_sel = 8;
  if (const char* v = std::getenv("EPCOMBINE_VEC")) vec_sel = std::atoi(v);
  int64_t threads = 256;
  if (const char* t = std::getenv("EPCOMBINE_THREADS")) threads = std::atoi(t);
  int no_compute_flag = 0;
  if (const char* nc = std::getenv("EPCOMBINE_NOCOMPUTE")) no_compute_flag = std::atoi(nc);

  c10::Device device(c10::DeviceType::XPU, output.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      output.scalar_type(), "ep_combine", [&]() {
        const int64_t num_wgs = world_size * num_tokens_per_rank;
        auto launch_wg = [&](auto vec_tag) {
          constexpr int VEC_SIZE = decltype(vec_tag)::value;
          const int64_t hidden_vecs = hidden_size / VEC_SIZE;
          auto kfn = EpCombineRingWGKernel<scalar_t, VEC_SIZE>{
              expert_output.data_ptr<scalar_t>(),
              rank_output_ptrs.data_ptr<int64_t>(),
              topk_idx.data_ptr<int32_t>(),
              scatter_idx.data_ptr<int32_t>(),
              topk_weights.data_ptr<float>(),
              static_cast<int32_t>(num_tokens_per_rank),
              static_cast<int32_t>(hidden_size),
              static_cast<int32_t>(topk),
              static_cast<int32_t>(rank),
              static_cast<int32_t>(world_size),
              static_cast<int32_t>(hidden_vecs),
              base_experts,
              rem_experts,
              boundary,
              /*write_empty=*/1,
              /*no_compute=*/no_compute_flag};
          sycl_kernel_submit(
              sycl::range<1>(num_wgs * threads),
              sycl::range<1>(threads),
              queue,
              kfn);
        };

        if (vec_sel == 16 && hidden_size % 16 == 0) {
          launch_wg(std::integral_constant<int, 16>{});
        } else if (hidden_size % 8 == 0) {
          launch_wg(std::integral_constant<int, 8>{});
        } else {
          const int64_t total = world_size * num_tokens_per_rank * hidden_size;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = EpCombineRingScalarKernel<scalar_t>{
              expert_output.data_ptr<scalar_t>(),
              rank_output_ptrs.data_ptr<int64_t>(),
              topk_idx.data_ptr<int32_t>(),
              scatter_idx.data_ptr<int32_t>(),
              topk_weights.data_ptr<float>(),
              num_tokens_per_rank,
              hidden_size,
              topk,
              rank,
              world_size,
              base_experts,
              rem_experts,
              boundary};
          sycl_kernel_submit(
              sycl::range<1>(blocks * threads),
              sycl::range<1>(threads),
              queue,
              kfn);
        }
      });

  return output;
}

// Fused reduction kernel: output[t] = sum_r recv[r][t] over all world_size
// receive-buffer slots. One work-group per token, threads stride the hidden
// dimension issuing wide vectorized loads/stores. Replaces the per-slot Python
// add loop (which launched world_size+1 kernels and re-read `output` each time).
template <typename scalar_t, int VEC_SIZE>
struct EpCombineReduceKernel {
  const scalar_t* recv_ptr;  // [world_size, num_tokens_per_rank, hidden]
  scalar_t* output_ptr;      // [num_tokens_per_rank, hidden]
  int32_t num_tokens_per_rank;
  int32_t hidden_size;
  int32_t hidden_vecs;
  int32_t world_size;

  void operator()(sycl::nd_item<1> item) const {
    using Vec = EpVec<scalar_t, VEC_SIZE>;
    const int32_t token = static_cast<int32_t>(item.get_group(0));
    if (token >= num_tokens_per_rank) return;
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t nthreads = static_cast<int32_t>(item.get_local_range(0));

    const int64_t slot_stride =
        static_cast<int64_t>(num_tokens_per_rank) * hidden_size;
    const int64_t token_off = static_cast<int64_t>(token) * hidden_size;
    scalar_t* dst_row = output_ptr + token_off;

    for (int32_t vh = lid; vh < hidden_vecs; vh += nthreads) {
      const int32_t h = vh * VEC_SIZE;
      float acc[VEC_SIZE];
      #pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) acc[i] = 0.0f;

      const scalar_t* base = recv_ptr + token_off + h;
      for (int32_t r = 0; r < world_size; ++r) {
        const Vec v = *reinterpret_cast<const Vec*>(base + r * slot_stride);
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) acc[i] += static_cast<float>(v.data[i]);
      }

      Vec out;
      #pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) out.data[i] = static_cast<scalar_t>(acc[i]);
      *reinterpret_cast<Vec*>(dst_row + h) = out;
    }
  }
};

// Reduce the world_size receive-buffer slots of `recv` into `output`.
at::Tensor ep_combine_reduce(
    const at::Tensor& recv,
    at::Tensor output,
    int64_t world_size) {
  TORCH_CHECK(recv.is_contiguous() && output.is_contiguous());
  TORCH_CHECK(output.dim() == 2, "ep_combine_reduce: output must be 2D");
  const int64_t num_tokens_per_rank = output.size(0);
  const int64_t hidden_size = output.size(1);
  if (num_tokens_per_rank == 0 || hidden_size == 0) return output;

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;

  c10::Device device(c10::DeviceType::XPU, output.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      output.scalar_type(), "ep_combine_reduce", [&]() {
        TORCH_CHECK(hidden_size % VEC_SIZE == 0,
                    "ep_combine_reduce: hidden must be divisible by 8");
        const int32_t hidden_vecs = static_cast<int32_t>(hidden_size / VEC_SIZE);
        const int64_t num_wgs = num_tokens_per_rank;
        auto kfn = EpCombineReduceKernel<scalar_t, VEC_SIZE>{
            recv.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<int32_t>(num_tokens_per_rank),
            static_cast<int32_t>(hidden_size),
            hidden_vecs,
            static_cast<int32_t>(world_size)};
        sycl_kernel_submit(
            sycl::range<1>(num_wgs * threads),
            sycl::range<1>(threads),
            queue,
            kfn);
      });

  return output;
}

// Like local_unpermute_copy_ but only reads rows for experts owned by this rank.
// Saves ~75% HBM bandwidth compared to reading all topk rows.
// Output must be pre-zeroed by caller (skipped tokens remain zero).
template <typename scalar_t, int VEC_SIZE>
struct EpCombineLocalVecKernel {
  const scalar_t* expert_output_ptr;
  const int32_t* topk_idx_ptr;
  const int32_t* scatter_idx_ptr;
  const float* topk_weights_ptr;
  scalar_t* output_ptr;
  int32_t chunk_start;
  int32_t num_tokens_per_chunk;
  int32_t hidden_size;
  int32_t topk;
  int32_t hidden_vecs;
  int32_t rank;
  int32_t base_experts;
  int32_t rem_experts;
  int32_t boundary;

  void operator()(sycl::nd_item<1> item) const {
    const int32_t idx = static_cast<int32_t>(item.get_global_id(0));
    const int32_t total = num_tokens_per_chunk * hidden_vecs;
    if (idx >= total) return;

    const int32_t vec_h = idx % hidden_vecs;
    const int32_t local_token_idx = idx / hidden_vecs;
    const int32_t global_token_idx = chunk_start + local_token_idx;
    const int32_t h_start = vec_h * VEC_SIZE;

    // Ownership pre-check: skip if no owned experts for this token
    const int64_t topk_base = static_cast<int64_t>(global_token_idx) * topk;
    bool has_owned = false;
    for (int32_t k = 0; k < topk; ++k) {
      const int32_t expert = topk_idx_ptr[topk_base + k];
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      if (owner == rank) { has_owned = true; break; }
    }
    if (!has_owned) return;  // output pre-zeroed by caller

    // Accumulate only from owned experts
    float acc[VEC_SIZE] = {};
    for (int32_t k = 0; k < topk; ++k) {
      const int32_t expert = topk_idx_ptr[topk_base + k];
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      if (owner == rank) {
        const float weight = topk_weights_ptr[topk_base + k];
        const int32_t src_row = scatter_idx_ptr[topk_base + k];
        const scalar_t* src = expert_output_ptr +
            static_cast<int64_t>(src_row) * hidden_size + h_start;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          acc[i] += weight * static_cast<float>(src[i]);
        }
      }
    }

    scalar_t* dst = output_ptr +
        static_cast<int64_t>(local_token_idx) * hidden_size + h_start;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      dst[i] = static_cast<scalar_t>(acc[i]);
    }
  }
};

at::Tensor ep_combine_local_(
    const at::Tensor& expert_output,
    const at::Tensor& topk_idx,
    const at::Tensor& scatter_idx,
    const at::Tensor& topk_weights,
    at::Tensor output,
    int64_t num_experts,
    int64_t chunk_start,
    int64_t num_tokens_per_chunk,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(expert_output.dim() == 2);
  TORCH_CHECK(expert_output.is_contiguous());
  TORCH_CHECK(topk_idx.dim() == 2 && topk_idx.scalar_type() == at::kInt);
  TORCH_CHECK(topk_idx.is_contiguous());
  TORCH_CHECK(scatter_idx.dim() == 2 && scatter_idx.scalar_type() == at::kInt);
  TORCH_CHECK(scatter_idx.is_contiguous());
  TORCH_CHECK(topk_weights.dim() == 2 && topk_weights.scalar_type() == at::kFloat);
  TORCH_CHECK(topk_weights.is_contiguous());
  TORCH_CHECK(output.dim() == 2 && output.is_contiguous());
  TORCH_CHECK(output.size(0) == num_tokens_per_chunk);
  TORCH_CHECK(output.size(1) == expert_output.size(1));

  const int64_t hidden_size = expert_output.size(1);
  const int64_t topk = topk_idx.size(1);

  if (num_tokens_per_chunk == 0) return output;

  const int32_t base_experts_val = static_cast<int32_t>(num_experts / world_size);
  const int32_t rem_experts_val = static_cast<int32_t>(num_experts % world_size);
  const int32_t boundary_val = rem_experts_val * (base_experts_val + 1);

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;

  c10::Device device(c10::DeviceType::XPU, output.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      expert_output.scalar_type(), "ep_combine_local_", [&]() {
        if (hidden_size % VEC_SIZE == 0) {
          const int32_t hidden_vecs = static_cast<int32_t>(hidden_size / VEC_SIZE);
          const int64_t total = num_tokens_per_chunk * hidden_vecs;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = EpCombineLocalVecKernel<scalar_t, VEC_SIZE>{
              expert_output.data_ptr<scalar_t>(),
              topk_idx.data_ptr<int32_t>(),
              scatter_idx.data_ptr<int32_t>(),
              topk_weights.data_ptr<float>(),
              output.data_ptr<scalar_t>(),
              static_cast<int32_t>(chunk_start),
              static_cast<int32_t>(num_tokens_per_chunk),
              static_cast<int32_t>(hidden_size),
              static_cast<int32_t>(topk),
              hidden_vecs,
              static_cast<int32_t>(rank),
              base_experts_val,
              rem_experts_val,
              boundary_val};
          sycl_kernel_submit(
              sycl::range<1>(blocks * threads),
              sycl::range<1>(threads),
              queue,
              kfn);
        }
      });

  return output;
}

// ===========================================================================
// PULL / gather-based combine (two-phase, owner-side pre-aggregation).
//
// The naive "pull topk" gathers TOP_K scattered remote rows per output token
// (8 remote loads, gather via scatter_idx) -> far too many expensive remote
// loads. Instead we split combine into two phases so each puller reads AT MOST
// `world_size` DENSE remote rows per token:
//
//   Phase 1 (EpCombinePartialKernel, LOCAL): every rank aggregates the experts
//   IT OWNS for every global token into a single partial row, applying the
//   softmax weights locally:
//       partial_local[gtok] = sum_{k : owner(topk_idx[gtok,k])==rank}
//                                 topk_weights[gtok,k] * expert_output[scatter_idx[gtok,k]]
//   Reads are local (expert_output), the store is local into this rank's
//   symmetric `partial` buffer laid out as [num_global_tokens, hidden].
//
//   Phase 2 (EpCombineGatherKernel, PULL): every rank, for each of ITS OWN
//   output tokens, reads the pre-aggregated partial row from each contributing
//   owner rank and sums them:
//       output[t] = sum_{r in owners(t)} partial_of[r][global_token(t)]
//   Remote traffic is now <= world_size dense, coalesced reads per token (no
//   weights, no scatter gather) and the only writes are local -> no incast.
//
// `partial_rank_ptrs[r]` points to rank r's symmetric partial buffer base.
template <typename scalar_t, int VEC_SIZE>
struct EpCombinePartialKernel {
  const scalar_t* expert_output_ptr;  // local [num_tokens * topk, hidden]
  scalar_t* partial_ptr;              // local symmetric [num_global_tokens, hidden]
  const int32_t* topk_idx_ptr;
  const int32_t* scatter_idx_ptr;
  const float* topk_weights_ptr;
  int32_t num_global_tokens;
  int32_t hidden_size;
  int32_t topk;
  int32_t rank;
  int32_t hidden_vecs;
  int32_t base_experts;
  int32_t rem_experts;
  int32_t boundary;
  int32_t num_tokens_per_rank;  // own-shard tokens are fused into gather

  void operator()(sycl::nd_item<1> item) const {
    using Vec = EpVec<scalar_t, VEC_SIZE>;
    const int32_t wg = static_cast<int32_t>(item.get_group(0));  // global token
    if (wg >= num_global_tokens) return;
    // This rank's OWN-shard tokens are combined inline by the gather kernel
    // (fused local contribution), so we neither pre-aggregate nor serve them
    // here — only this rank ever reads its own partial rows for its own shard.
    if (wg / num_tokens_per_rank == rank) return;
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t nthreads = static_cast<int32_t>(item.get_local_range(0));
    const int64_t topk_base = static_cast<int64_t>(wg) * topk;

    // Resolve owned experts ONCE for this token.
    float owned_w[kEpCombineMaxTopK];
    int32_t owned_row[kEpCombineMaxTopK];
    int32_t num_owned = 0;
    for (int32_t k = 0; k < topk; ++k) {
      const int32_t expert = topk_idx_ptr[topk_base + k];
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      if (owner == rank) {
        owned_w[num_owned] = topk_weights_ptr[topk_base + k];
        owned_row[num_owned] = scatter_idx_ptr[topk_base + k];
        ++num_owned;
      }
    }

    // Sparse write: this rank contributes to global token `wg` only if it owns
    // at least one of its experts. Tokens with no owned expert are skipped
    // entirely (no dense zero-store) — the gather phase reads a rank's partial
    // row ONLY when that rank is in the token's owner_mask, i.e. exactly when
    // num_owned > 0 here, so the skipped rows are never read.
    if (num_owned == 0) return;

    scalar_t* out_row = partial_ptr + static_cast<int64_t>(wg) * hidden_size;
    for (int32_t vh = lid; vh < hidden_vecs; vh += nthreads) {
      const int32_t h_start = vh * VEC_SIZE;
      float acc[VEC_SIZE];
      #pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) acc[i] = 0.0f;

      for (int32_t j = 0; j < num_owned; ++j) {
        const float weight = owned_w[j];
        const Vec src = *reinterpret_cast<const Vec*>(
            expert_output_ptr +
            static_cast<int64_t>(owned_row[j]) * hidden_size + h_start);
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          acc[i] += weight * static_cast<float>(src.data[i]);
        }
      }

      Vec out;
      #pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) out.data[i] = static_cast<scalar_t>(acc[i]);
      // Local store into this rank's symmetric partial buffer. Only reached for
      // owned tokens (num_owned > 0); non-owned rows are left untouched and are
      // never read by the gather phase.
      *reinterpret_cast<Vec*>(out_row + h_start) = out;
    }
  }
};

template <typename scalar_t, int VEC_SIZE>
struct EpCombineGatherKernel {
  const int64_t* partial_ptrs;  // [world_size] symmetric partial bases
  const int32_t* topk_idx_ptr;
  scalar_t* output_ptr;  // local [num_tokens_per_rank, hidden]
  const scalar_t* expert_output_ptr;  // local [num_tokens * topk, hidden]
  const int32_t* scatter_idx_ptr;
  const float* topk_weights_ptr;
  int32_t num_tokens_per_rank;
  int32_t hidden_size;
  int32_t topk;
  int32_t rank;
  int32_t world_size;
  int32_t hidden_vecs;
  int32_t base_experts;
  int32_t rem_experts;
  int32_t boundary;

  void operator()(sycl::nd_item<1> item) const {
    using Vec = EpVec<scalar_t, VEC_SIZE>;
    const int32_t wg = static_cast<int32_t>(item.get_group(0));
    if (wg >= num_tokens_per_rank) return;
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t nthreads = static_cast<int32_t>(item.get_local_range(0));

    const int32_t global_token_idx = rank * num_tokens_per_rank + wg;
    const int64_t topk_base = static_cast<int64_t>(global_token_idx) * topk;

    // Build the set of contributing REMOTE owner ranks for this token (dedup)
    // and, in the same pass, collect this rank's own owned experts so their
    // contribution can be computed inline (fused from phase 1) — overlapping
    // the comm-bound remote reads below. world_size <= 32.
    uint32_t owner_mask = 0;
    float owned_w[kEpCombineMaxTopK];
    int32_t owned_row[kEpCombineMaxTopK];
    int32_t num_owned = 0;
    for (int32_t k = 0; k < topk; ++k) {
      const int32_t expert = topk_idx_ptr[topk_base + k];
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      owner_mask |= (1u << owner);
      if (owner == rank) {
        owned_w[num_owned] = topk_weights_ptr[topk_base + k];
        owned_row[num_owned] = scatter_idx_ptr[topk_base + k];
        ++num_owned;
      }
    }

    const int64_t row_off = static_cast<int64_t>(global_token_idx) * hidden_size;
    scalar_t* out_row = output_ptr + static_cast<int64_t>(wg) * hidden_size;

    for (int32_t vh = lid; vh < hidden_vecs; vh += nthreads) {
      const int32_t h_start = vh * VEC_SIZE;
      float acc[VEC_SIZE];
      #pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) acc[i] = 0.0f;

      // Inline local contribution (this rank's own owned experts). Local reads
      // hide under the remote partial reads that follow.
      for (int32_t j = 0; j < num_owned; ++j) {
        const float weight = owned_w[j];
        const Vec src = *reinterpret_cast<const Vec*>(
            expert_output_ptr +
            static_cast<int64_t>(owned_row[j]) * hidden_size + h_start);
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          acc[i] += weight * static_cast<float>(src.data[i]);
        }
      }

      for (int32_t i = 0; i < world_size; ++i) {
        // Ring order: each rank starts reading from a different owner so the
        // read traffic is spread across links instead of all ranks hammering
        // owner 0 first (read-side incast).
        const int32_t r = (rank + 1 + i) % world_size;
        if (r == rank) continue;  // local part already added inline
        if (!(owner_mask & (1u << r))) continue;
        const scalar_t* src =
            reinterpret_cast<const scalar_t*>(partial_ptrs[r]);
        const Vec v = *reinterpret_cast<const Vec*>(src + row_off + h_start);
        #pragma unroll
        for (int i2 = 0; i2 < VEC_SIZE; ++i2) {
          acc[i2] += static_cast<float>(v.data[i2]);
        }
      }

      Vec out;
      #pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) out.data[i] = static_cast<scalar_t>(acc[i]);
      *reinterpret_cast<Vec*>(out_row + h_start) = out;  // local store
    }
  }
};

// Two-phase pull combine. `partial_rank_ptrs[rank]` must point to THIS rank's
// symmetric partial buffer [num_global_tokens, hidden]; the other entries point
// to the peers' buffers. Caller is responsible for the cross-rank barrier
// between phase 1 and phase 2 (all ranks' partial buffers must be visible
// before any gather reads them).
at::Tensor ep_combine_pull(
    const at::Tensor& expert_output,
    const at::Tensor& partial_local,
    const at::Tensor& partial_rank_ptrs,
    const at::Tensor& topk_idx,
    const at::Tensor& scatter_idx,
    const at::Tensor& topk_weights,
    at::Tensor output,
    int64_t num_experts,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(
      partial_rank_ptrs.dim() == 1 && partial_rank_ptrs.size(0) == world_size,
      "ep_combine_pull: partial_rank_ptrs must be 1D with size == world_size");
  TORCH_CHECK(partial_rank_ptrs.scalar_type() == at::kLong,
              "ep_combine_pull: partial_rank_ptrs must be int64");
  TORCH_CHECK(partial_local.is_contiguous() &&
                  partial_local.scalar_type() == output.scalar_type(),
              "ep_combine_pull: partial_local must be contiguous, match output dtype");
  TORCH_CHECK(expert_output.dim() == 2 && expert_output.is_contiguous(),
              "ep_combine_pull: expert_output must be 2D contiguous");
  TORCH_CHECK(topk_idx.dim() == 2 && topk_idx.scalar_type() == at::kInt &&
                  topk_idx.is_contiguous(),
              "ep_combine_pull: topk_idx must be 2D contiguous int32");
  TORCH_CHECK(scatter_idx.dim() == 2 && scatter_idx.scalar_type() == at::kInt &&
                  scatter_idx.is_contiguous() &&
                  scatter_idx.sizes() == topk_idx.sizes(),
              "ep_combine_pull: scatter_idx must match topk_idx (int32)");
  TORCH_CHECK(topk_weights.dim() == 2 &&
                  topk_weights.scalar_type() == at::kFloat &&
                  topk_weights.is_contiguous() &&
                  topk_weights.sizes() == topk_idx.sizes(),
              "ep_combine_pull: topk_weights must match topk_idx (float32)");
  TORCH_CHECK(output.dim() == 2 && output.is_contiguous(),
              "ep_combine_pull: output must be 2D contiguous");
  TORCH_CHECK(world_size <= 32, "ep_combine_pull: world_size must be <= 32");
  TORCH_CHECK(rank >= 0 && rank < world_size);

  const int64_t num_tokens = topk_idx.size(0);
  const int64_t topk = topk_idx.size(1);
  const int64_t hidden_size = output.size(1);
  TORCH_CHECK(topk <= kEpCombineMaxTopK,
              "ep_combine_pull: topk exceeds kEpCombineMaxTopK");
  TORCH_CHECK(num_tokens % world_size == 0,
              "ep_combine_pull: num_tokens must be divisible by world_size");
  const int64_t num_tokens_per_rank = num_tokens / world_size;
  TORCH_CHECK(output.size(0) == num_tokens_per_rank,
              "ep_combine_pull: output first dim must be num_tokens_per_rank");
  TORCH_CHECK(expert_output.size(1) == hidden_size,
              "ep_combine_pull: expert_output hidden must match output");

  if (num_tokens_per_rank == 0 || hidden_size == 0 || topk == 0) return output;

  const int32_t base_experts = static_cast<int32_t>(num_experts / world_size);
  const int32_t rem_experts = static_cast<int32_t>(num_experts % world_size);
  const int32_t boundary = rem_experts * (base_experts + 1);

  int vec_sel = 8;
  if (const char* v = std::getenv("EPCOMBINE_VEC")) vec_sel = std::atoi(v);
  int64_t threads = 256;
  if (const char* t = std::getenv("EPCOMBINE_THREADS")) threads = std::atoi(t);

  c10::Device device(c10::DeviceType::XPU, output.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      output.scalar_type(), "ep_combine_pull", [&]() {
        auto* partial_local_ptr = partial_local.data_ptr<scalar_t>();
        auto launch = [&](auto vec_tag) {
          constexpr int VEC_SIZE = decltype(vec_tag)::value;
          const int64_t hidden_vecs = hidden_size / VEC_SIZE;

          // Phase 1: local owner-side pre-aggregation (one WG per global token).
          auto pfn = EpCombinePartialKernel<scalar_t, VEC_SIZE>{
              expert_output.data_ptr<scalar_t>(),
              partial_local_ptr,
              topk_idx.data_ptr<int32_t>(),
              scatter_idx.data_ptr<int32_t>(),
              topk_weights.data_ptr<float>(),
              static_cast<int32_t>(num_tokens),
              static_cast<int32_t>(hidden_size),
              static_cast<int32_t>(topk),
              static_cast<int32_t>(rank),
              static_cast<int32_t>(hidden_vecs),
              base_experts,
              rem_experts,
              boundary,
              static_cast<int32_t>(num_tokens_per_rank)};
          sycl_kernel_submit(
              sycl::range<1>(num_tokens * threads),
              sycl::range<1>(threads), queue, pfn);

          // Phase 2: gather pre-aggregated partials (one WG per output token).
          auto gfn = EpCombineGatherKernel<scalar_t, VEC_SIZE>{
              partial_rank_ptrs.data_ptr<int64_t>(),
              topk_idx.data_ptr<int32_t>(),
              output.data_ptr<scalar_t>(),
              expert_output.data_ptr<scalar_t>(),
              scatter_idx.data_ptr<int32_t>(),
              topk_weights.data_ptr<float>(),
              static_cast<int32_t>(num_tokens_per_rank),
              static_cast<int32_t>(hidden_size),
              static_cast<int32_t>(topk),
              static_cast<int32_t>(rank),
              static_cast<int32_t>(world_size),
              static_cast<int32_t>(hidden_vecs),
              base_experts,
              rem_experts,
              boundary};
          sycl_kernel_submit(
              sycl::range<1>(num_tokens_per_rank * threads),
              sycl::range<1>(threads), queue, gfn);
        };

        TORCH_CHECK(hidden_size % 8 == 0,
                    "ep_combine_pull: hidden must be divisible by 8");
        if (vec_sel == 16 && hidden_size % 16 == 0) {
          launch(std::integral_constant<int, 16>{});
        } else {
          launch(std::integral_constant<int, 8>{});
        }
      });

  return output;
}

// Phase-1-only: owner-side pre-aggregation into this rank's local partial
// buffer. Purely local (local reads + local store); no cross-rank traffic.
at::Tensor ep_combine_pull_partial(
    const at::Tensor& expert_output,
    at::Tensor partial_local,
    const at::Tensor& topk_idx,
    const at::Tensor& scatter_idx,
    const at::Tensor& topk_weights,
    int64_t num_experts,
    int64_t rank,
    int64_t world_size) {
  const int64_t num_tokens = topk_idx.size(0);
  const int64_t topk = topk_idx.size(1);
  const int64_t hidden_size = partial_local.size(1);
  const int64_t num_tokens_per_rank = num_tokens / world_size;
  const int32_t base_experts = static_cast<int32_t>(num_experts / world_size);
  const int32_t rem_experts = static_cast<int32_t>(num_experts % world_size);
  const int32_t boundary = rem_experts * (base_experts + 1);
  int vec_sel = 8;
  if (const char* v = std::getenv("EPCOMBINE_VEC")) vec_sel = std::atoi(v);
  int64_t threads = 256;
  if (const char* t = std::getenv("EPCOMBINE_THREADS")) threads = std::atoi(t);
  c10::Device device(c10::DeviceType::XPU, partial_local.device().index());
  c10::DeviceGuard guard(device);
  auto& queue = at::xpu::getCurrentXPUStream().queue();
  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      partial_local.scalar_type(), "ep_combine_pull_partial", [&]() {
        auto launch = [&](auto vec_tag) {
          constexpr int VEC_SIZE = decltype(vec_tag)::value;
          const int64_t hidden_vecs = hidden_size / VEC_SIZE;
          auto pfn = EpCombinePartialKernel<scalar_t, VEC_SIZE>{
              expert_output.data_ptr<scalar_t>(),
              partial_local.data_ptr<scalar_t>(),
              topk_idx.data_ptr<int32_t>(),
              scatter_idx.data_ptr<int32_t>(),
              topk_weights.data_ptr<float>(),
              static_cast<int32_t>(num_tokens),
              static_cast<int32_t>(hidden_size),
              static_cast<int32_t>(topk),
              static_cast<int32_t>(rank),
              static_cast<int32_t>(hidden_vecs),
              base_experts, rem_experts, boundary,
              static_cast<int32_t>(num_tokens_per_rank)};
          sycl_kernel_submit(sycl::range<1>(num_tokens * threads),
                             sycl::range<1>(threads), queue, pfn);
        };
        if (vec_sel == 16 && hidden_size % 16 == 0)
          launch(std::integral_constant<int, 16>{});
        else
          launch(std::integral_constant<int, 8>{});
      });
  return partial_local;
}

// Phase-2-only: gather pre-aggregated partial rows from the (already-populated)
// symmetric partial buffers of every owner rank into `output`. Remote reads.
at::Tensor ep_combine_pull_gather(
    const at::Tensor& partial_rank_ptrs,
    const at::Tensor& topk_idx,
    at::Tensor output,
    const at::Tensor& expert_output,
    const at::Tensor& scatter_idx,
    const at::Tensor& topk_weights,
    int64_t num_experts,
    int64_t rank,
    int64_t world_size) {
  const int64_t num_tokens = topk_idx.size(0);
  const int64_t topk = topk_idx.size(1);
  const int64_t hidden_size = output.size(1);
  const int64_t num_tokens_per_rank = num_tokens / world_size;
  const int32_t base_experts = static_cast<int32_t>(num_experts / world_size);
  const int32_t rem_experts = static_cast<int32_t>(num_experts % world_size);
  const int32_t boundary = rem_experts * (base_experts + 1);
  int vec_sel = 8;
  if (const char* v = std::getenv("EPCOMBINE_VEC")) vec_sel = std::atoi(v);
  int64_t threads = 256;
  if (const char* t = std::getenv("EPCOMBINE_THREADS")) threads = std::atoi(t);
  c10::Device device(c10::DeviceType::XPU, output.device().index());
  c10::DeviceGuard guard(device);
  auto& queue = at::xpu::getCurrentXPUStream().queue();
  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      output.scalar_type(), "ep_combine_pull_gather", [&]() {
        auto launch = [&](auto vec_tag) {
          constexpr int VEC_SIZE = decltype(vec_tag)::value;
          const int64_t hidden_vecs = hidden_size / VEC_SIZE;
          auto gfn = EpCombineGatherKernel<scalar_t, VEC_SIZE>{
              partial_rank_ptrs.data_ptr<int64_t>(),
              topk_idx.data_ptr<int32_t>(),
              output.data_ptr<scalar_t>(),
              expert_output.data_ptr<scalar_t>(),
              scatter_idx.data_ptr<int32_t>(),
              topk_weights.data_ptr<float>(),
              static_cast<int32_t>(num_tokens_per_rank),
              static_cast<int32_t>(hidden_size),
              static_cast<int32_t>(topk),
              static_cast<int32_t>(rank),
              static_cast<int32_t>(world_size),
              static_cast<int32_t>(hidden_vecs),
              base_experts, rem_experts, boundary};
          sycl_kernel_submit(sycl::range<1>(num_tokens_per_rank * threads),
                             sycl::range<1>(threads), queue, gfn);
        };
        if (vec_sel == 16 && hidden_size % 16 == 0)
          launch(std::integral_constant<int, 16>{});
        else
          launch(std::integral_constant<int, 8>{});
      });
  return output;
}

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "ep_combine(Tensor expert_output, Tensor rank_output_ptrs, "
      "Tensor topk_idx, Tensor scatter_idx, Tensor topk_weights, "
      "Tensor(a!) output, int num_experts, "
      "int rank, int world_size) -> Tensor(a!)");
  m.def(
      "ep_combine_pull(Tensor expert_output, Tensor partial_local, "
      "Tensor partial_rank_ptrs, Tensor topk_idx, Tensor scatter_idx, "
      "Tensor topk_weights, Tensor(a!) output, int num_experts, int rank, "
      "int world_size) -> Tensor(a!)");
  m.def(
      "ep_combine_pull_partial(Tensor expert_output, Tensor(a!) partial_local, "
      "Tensor topk_idx, Tensor scatter_idx, Tensor topk_weights, "
      "int num_experts, int rank, int world_size) -> Tensor(a!)");
  m.def(
      "ep_combine_pull_gather(Tensor partial_rank_ptrs, Tensor topk_idx, "
      "Tensor(a!) output, Tensor expert_output, Tensor scatter_idx, "
      "Tensor topk_weights, int num_experts, int rank, int world_size) "
      "-> Tensor(a!)");
  m.def(
      "ep_combine_reduce(Tensor recv, Tensor(a!) output, int world_size) "
      "-> Tensor(a!)");
  m.def(
      "ep_combine_local_(Tensor expert_output, Tensor topk_idx, "
      "Tensor scatter_idx, Tensor topk_weights, "
      "Tensor(a!) output, int num_experts, "
      "int chunk_start, int num_tokens_per_chunk, "
      "int rank, int world_size) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("ep_combine", ep_combine);
  m.impl("ep_combine_pull", ep_combine_pull);
  m.impl("ep_combine_pull_partial", ep_combine_pull_partial);
  m.impl("ep_combine_pull_gather", ep_combine_pull_gather);
  m.impl("ep_combine_reduce", ep_combine_reduce);
  m.impl("ep_combine_local_", ep_combine_local_);
}
