#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <comm/SYCLHelpers.h>

#ifndef AT_DISPATCH_FLOAT_AND_BFLOAT16
#define AT_DISPATCH_FLOAT_AND_BFLOAT16(scalar_type, name, ...)         \
  AT_DISPATCH_SWITCH(                                                  \
      scalar_type, name, AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__))
#endif

// Scalar fallback kernel for non-aligned numel_per_rank.
// Ring-ordered allgather with explicit input_shard for local rank.
// For src_rank == rank: reads from input_shard (fast local read)
// For src_rank != rank: reads from rank_buffers_ptr[src_rank] (remote symm_mem)
template <typename T>
struct AllgatherWithSymmMemScalarKernel {
  const T* input_shard_ptr;
  const int64_t* rank_buffers_ptr;
  T* output_ptr;
  int64_t numel_per_rank;
  int64_t rank;
  int64_t world_size;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t idx = static_cast<int64_t>(item.get_global_id(0));
    const int64_t total = world_size * numel_per_rank;
    if (idx >= total) return;

    const int64_t step = idx / numel_per_rank;
    const int64_t elem = idx % numel_per_rank;

    // Ring ordering: different ranks read different sources at the same step
    const int64_t src_rank = (rank + step + 1) % world_size;

    // Read source element (local or remote)
    T val;
    if (src_rank == rank) {
      // Fast local read from input_shard
      val = input_shard_ptr[elem];
    } else {
      // Remote read from symmetric memory via PCIe
      const T* src = reinterpret_cast<const T*>(rank_buffers_ptr[src_rank]);
      val = src[elem];
    }

    // Contiguous write: rank r's elements go to [r * numel_per_rank, ...]
    output_ptr[src_rank * numel_per_rank + elem] = val;
  }
};

// Vectorized ring-ordered kernel with VEC_SIZE=8.
// Decomposition: idx → (step, vec_elem)
//   vec_elem = idx % numel_vecs           (innermost: coalesced access)
//   step = idx / numel_vecs               (step through world_size)
template <typename scalar_t, int VEC_SIZE>
struct AllgatherWithSymmMemVecKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const scalar_t* input_shard_ptr;
  const int64_t* rank_buffers_ptr;
  scalar_t* output_ptr;
  int32_t numel_per_rank;
  int32_t rank;
  int32_t world_size;
  int32_t numel_vecs;  // numel_per_rank / VEC_SIZE

  void operator()(sycl::nd_item<1> item) const {
    const int32_t idx = static_cast<int32_t>(item.get_global_id(0));
    const int32_t total = world_size * numel_vecs;
    if (idx >= total) return;

    const int32_t vec_elem = idx % numel_vecs;
    const int32_t step = idx / numel_vecs;

    // Ring ordering avoids all ranks hitting the same source simultaneously
    const int32_t src_rank = (rank + step + 1) % world_size;

    // Coalesced vectorized read from source (local or remote)
    vec_t v;
    if (src_rank == rank) {
      // Fast local read from input_shard
      auto src_vec = reinterpret_cast<const vec_t*>(input_shard_ptr);
      v = src_vec[vec_elem];
    } else {
      // Remote read from symmetric memory via PCIe
      const scalar_t* src = reinterpret_cast<const scalar_t*>(rank_buffers_ptr[src_rank]);
      auto src_vec = reinterpret_cast<const vec_t*>(src);
      v = src_vec[vec_elem];
    }

    // Contiguous write: rank r's elements go to [r * numel_per_rank, ...]
    auto dst_vec = reinterpret_cast<vec_t*>(
        output_ptr + static_cast<int64_t>(src_rank) * numel_per_rank);
    dst_vec[vec_elem] = v;
  }
};

at::Tensor allgather_with_symm_mem(
    const at::Tensor& input_shard,
    const at::Tensor& rank_buffers_ptr,
    at::Tensor output,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(
      input_shard.dim() == 1,
      "allgather_with_symm_mem: input_shard must be 1D");
  TORCH_CHECK(
      input_shard.is_contiguous(),
      "allgather_with_symm_mem: input_shard must be contiguous");
  TORCH_CHECK(
      rank_buffers_ptr.dim() == 1 && rank_buffers_ptr.size(0) == world_size,
      "allgather_with_symm_mem: rank_buffers_ptr must be 1D with size == world_size");
  TORCH_CHECK(
      rank_buffers_ptr.scalar_type() == at::kLong,
      "allgather_with_symm_mem: rank_buffers_ptr must be int64");
  TORCH_CHECK(
      output.dim() == 1,
      "allgather_with_symm_mem: output must be 1D");
  TORCH_CHECK(
      output.is_contiguous(),
      "allgather_with_symm_mem: output must be contiguous");
  TORCH_CHECK(
      rank >= 0 && rank < world_size,
      "allgather_with_symm_mem: rank must be in [0, world_size)");
  TORCH_CHECK(
      input_shard.scalar_type() == output.scalar_type(),
      "allgather_with_symm_mem: input_shard and output must have same dtype");

  const int64_t numel_per_rank = input_shard.numel();
  const int64_t total_numel = output.numel();

  TORCH_CHECK(
      total_numel == numel_per_rank * world_size,
      "allgather_with_symm_mem: output.numel() must equal input_shard.numel() * world_size");

  if (numel_per_rank == 0) {
    return output;
  }

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;

  c10::Device device(c10::DeviceType::XPU, output.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      output.scalar_type(), "allgather_with_symm_mem", [&]() {
        if (numel_per_rank % VEC_SIZE == 0) {
          const int64_t numel_vecs = numel_per_rank / VEC_SIZE;
          const int64_t total = world_size * numel_vecs;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = AllgatherWithSymmMemVecKernel<scalar_t, VEC_SIZE>{
              input_shard.data_ptr<scalar_t>(),
              rank_buffers_ptr.data_ptr<int64_t>(),
              output.data_ptr<scalar_t>(),
              static_cast<int32_t>(numel_per_rank),
              static_cast<int32_t>(rank),
              static_cast<int32_t>(world_size),
              static_cast<int32_t>(numel_vecs)};
          sycl_kernel_submit(
              sycl::range<1>(blocks * threads),
              sycl::range<1>(threads),
              queue,
              kfn);
        } else {
          const int64_t total = world_size * numel_per_rank;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = AllgatherWithSymmMemScalarKernel<scalar_t>{
              input_shard.data_ptr<scalar_t>(),
              rank_buffers_ptr.data_ptr<int64_t>(),
              output.data_ptr<scalar_t>(),
              numel_per_rank,
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
      "allgather_with_symm_mem(Tensor input_shard, Tensor rank_buffers_ptr, "
      "Tensor(a!) output, int rank, int world_size) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("allgather_with_symm_mem", allgather_with_symm_mem);
}
