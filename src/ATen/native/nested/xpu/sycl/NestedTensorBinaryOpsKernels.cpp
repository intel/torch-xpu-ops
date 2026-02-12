
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/nested/NestedTensorBinaryOps.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/native/xpu/sycl/IndexUtils.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <ATen/xpu/XPUContext.h>
#include <comm/XPUMathCompat.h>
#include <type_traits>

#define GROUP_DIM 256

namespace at::native::xpu {

template <typename T, int size>
using vec_t = memory::aligned_vector<T, size>;

template <typename scalar_t>
struct AddFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a + b;
  }
};

template <typename scalar_t>
struct MulFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a * b;
  }
};

template <typename scalar_t, typename VEC_T, int vec_size, typename func_t>
struct OpDenseVectorizedFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  OpDenseVectorizedFunctor(
      const VEC_T* input_vec,
      const VEC_T* dense_vec,
      VEC_T* output_vec,
      int64_t batch_size,
      int64_t embedding_dim,
      const int64_t* offsets,
      int64_t chunks_per_batch,
      const func_t f)
      : input_vec(input_vec),
        dense_vec(dense_vec),
        output_vec(output_vec),
        batch_size(batch_size),
        embedding_dim(embedding_dim),
        offsets(offsets),
        chunks_per_batch(chunks_per_batch),
        f(f) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t global_group_id = item.get_group(0);
    const int64_t tid = item.get_local_id(0);
    const int64_t local_size = item.get_local_range(0);

    const int64_t batch_idx = global_group_id / chunks_per_batch;
    const int64_t chunk_idx = global_group_id % chunks_per_batch;

    if (batch_idx >= batch_size)
      return;

    const int64_t v_embedding_dim = embedding_dim / vec_size;
    const int64_t v_d_per_chunk =
        (v_embedding_dim + chunks_per_batch - 1) / chunks_per_batch;
    const int64_t v_d_start = chunk_idx * v_d_per_chunk;
    const int64_t v_d_end =
        std::min(v_d_start + v_d_per_chunk, v_embedding_dim);
    const int64_t cur_v_chunk_size = v_d_end - v_d_start;

    const int64_t batch_start_offset = offsets[batch_idx];
    const int64_t batch_end_offset = offsets[batch_idx + 1];
    const int64_t range = batch_end_offset - batch_start_offset;

    const bool use_slm = (embedding_dim > 1024);

    if (cur_v_chunk_size > 0) {
      if (use_slm) {
        for (int64_t i = tid; i < cur_v_chunk_size; i += local_size) {
          slm_dense[i] = dense_vec[batch_idx * v_embedding_dim + v_d_start + i];
        }
        item.barrier(sycl_local_fence);
      }

      for (int64_t i = tid; i < cur_v_chunk_size; i += local_size) {
        VEC_T v_dense_val = use_slm
            ? slm_dense[i]
            : dense_vec[batch_idx * v_embedding_dim + v_d_start + i];

        for (int64_t n_offset = (v_d_start + i) * vec_size; n_offset < range;
             n_offset += embedding_dim) {
          const int64_t target_idx = (batch_start_offset + n_offset) / vec_size;
          VEC_T v_input_val = input_vec[target_idx];
          VEC_T v_res;
#pragma unroll
          for (int v = 0; v < vec_size; ++v) {
            v_res[v] = f(v_input_val[v], v_dense_val[v]);
          }
          output_vec[target_idx] = v_res;
        }
      }
    }

    if (chunk_idx == chunks_per_batch - 1) {
      const int64_t tail_start_d = v_embedding_dim * vec_size;
      for (int64_t d_idx = tail_start_d + tid; d_idx < embedding_dim;
           d_idx += local_size) {
        const scalar_t s_dense_val = reinterpret_cast<const scalar_t*>(
            dense_vec)[batch_idx * embedding_dim + d_idx];
        for (int64_t n_offset = d_idx; n_offset < range;
             n_offset += embedding_dim) {
          const int64_t target_idx = batch_start_offset + n_offset;
          reinterpret_cast<scalar_t*>(output_vec)[target_idx] =
              f(reinterpret_cast<const scalar_t*>(input_vec)[target_idx],
                s_dense_val);
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    int64_t v_d_per_chunk =
        (embedding_dim / vec_size + chunks_per_batch - 1) / chunks_per_batch;
    slm_dense =
        sycl_local_acc_t<VEC_T>(v_d_per_chunk > 0 ? v_d_per_chunk : 1, cgh);
  }

  const VEC_T* input_vec;
  const VEC_T* dense_vec;
  VEC_T* output_vec;
  int64_t batch_size;
  int64_t embedding_dim;
  const int64_t* offsets;
  int64_t chunks_per_batch;
  const func_t f;
  sycl_local_acc_t<VEC_T> slm_dense;
};

#define LAUNCH_VEC_CASE(SIZE)                                           \
  case SIZE: {                                                          \
    using VEC_T = vec_t<scalar_t, SIZE>;                                \
    auto kfn = OpDenseVectorizedFunctor<scalar_t, VEC_T, SIZE, func_t>( \
        reinterpret_cast<const VEC_T*>(input),                          \
        reinterpret_cast<const VEC_T*>(dense),                          \
        reinterpret_cast<VEC_T*>(output),                               \
        batch_size,                                                     \
        embedding_dim,                                                  \
        input_offsets,                                                  \
        chunks_per_batch,                                               \
        f);                                                             \
    sycl_kernel_submit(                                                 \
        total_groups* GROUP_DIM,                                        \
        GROUP_DIM,                                                      \
        at::xpu::getCurrentSYCLQueue(),                                 \
        kfn);                                                           \
  } break;

template <typename scalar_t, typename func_t>
void nested_op_dense_kernelLauncher(
    const scalar_t* input,
    const scalar_t* dense,
    scalar_t* output,
    int64_t batch_size,
    int64_t embedding_dim,
    const int64_t* input_offsets,
    func_t f) {
  int max_vec_size = memory::can_vectorize_up_to<scalar_t>(
      reinterpret_cast<const char*>(input));
  int vec_size = 1;
  for (int v : {8, 4, 2}) {
    if (max_vec_size >= v && embedding_dim % v == 0) {
      vec_size = v;
      break;
    }
  }

  int64_t chunks_per_batch =
      (embedding_dim > 1024) ? (embedding_dim + 1023) / 1024 : 1;
  int64_t total_groups = batch_size * chunks_per_batch;

  switch (vec_size) {
    LAUNCH_VEC_CASE(8)
    LAUNCH_VEC_CASE(4)
    LAUNCH_VEC_CASE(2)
    default: {
      LAUNCH_VEC_CASE(1)
    }
  }
}

template <typename scalar_t, typename func_t>
void _nested_op_dense_esuhm_kernel(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    func_t f) {
  auto self_ptr = get_nested_tensor_impl(self);
  const auto self_buffer = self_ptr->get_buffer();
  const auto offsets = self_ptr->get_storage_offsets();
  const auto batch_size = other.size(0);
  const auto embedding_size = other.size(2);
  auto result_buffer = get_nested_tensor_impl(result)->get_buffer();

  auto numel_tensor = at::tensor({self_ptr->numel()}, offsets.options());
  auto result_offsets = at::cat({offsets, numel_tensor}).to(at::kXPU);

  nested_op_dense_kernelLauncher<scalar_t, func_t>(
      self_buffer.const_data_ptr<scalar_t>(),
      other.const_data_ptr<scalar_t>(),
      result_buffer.data_ptr<scalar_t>(),
      batch_size,
      embedding_size,
      result_offsets.data_ptr<int64_t>(),
      f);
}

void _nested_op_dense_esuhm_xpu(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    const NESTED_DENSE_OP& op) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "_nested_op_dense_esuhm",
      [&]() {
        if (op == NESTED_DENSE_OP::ADD) {
          _nested_op_dense_esuhm_kernel<scalar_t>(
              result, self, other, AddFunctor<scalar_t>{});
        } else {
          _nested_op_dense_esuhm_kernel<scalar_t>(
              result, self, other, MulFunctor<scalar_t>{});
        }
      });
}

} // namespace at::native::xpu
