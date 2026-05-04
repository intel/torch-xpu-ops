/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/ceil_div.h>
#include <ATen/native/nested/xpu/sycl/NestedTensorBinaryOpsKernels.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <comm/Runtime.h>
#include <comm/SYCLHelpers.h>

#include <algorithm>
#include <cstdint>

constexpr int64_t SLM_TILING_THRESHOLD{1024};

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
      const func_t func)
      : input_vec(input_vec),
        dense_vec(dense_vec),
        output_vec(output_vec),
        batch_size(batch_size),
        embedding_dim(embedding_dim),
        offsets(offsets),
        chunks_per_batch(chunks_per_batch),
        func(func) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t item_id = item.get_local_id(0);
    const int64_t global_group_id = item.get_group(0);
    const int64_t local_size = item.get_local_range(0);

    const int64_t batch_idx = global_group_id / chunks_per_batch;
    const int64_t chunk_idx = global_group_id % chunks_per_batch;

    if (batch_idx >= batch_size) {
      return;
    }

    const int64_t vec_embedding_dim = embedding_dim / vec_size;
    const int64_t vecs_per_chunk =
        ceil_div(vec_embedding_dim, chunks_per_batch);
    const int64_t vec_start = chunk_idx * vecs_per_chunk;
    const int64_t chunk_end = vec_start + vecs_per_chunk;
    const int64_t vec_end = std::min(chunk_end, vec_embedding_dim);
    const int64_t vecs_count = vec_end - vec_start;

    const int64_t batch_start_offset = offsets[batch_idx];
    const int64_t batch_end_offset = offsets[batch_idx + 1];
    const int64_t range = batch_end_offset - batch_start_offset;

    if (vecs_count > 0) {
      int64_t chunk_offset = batch_idx * vec_embedding_dim + vec_start;

      const bool use_slm = embedding_dim > SLM_TILING_THRESHOLD;
      if (use_slm) {
        for (int64_t i = item_id; i < vecs_count; i += local_size) {
          slm_dense[i] = dense_vec[chunk_offset + i];
        }
        sycl::group_barrier(item.get_group());
      }

      for (int64_t i = item_id; i < vecs_count; i += local_size) {
        VEC_T vec_output_val =
            use_slm ? slm_dense[i] : dense_vec[chunk_offset + i];
        int64_t n_offset = (vec_start + i) * vec_size;
        for (; n_offset < range; n_offset += embedding_dim) {
          const int64_t target_idx = (batch_start_offset + n_offset) / vec_size;
          VEC_T vec_input_val = input_vec[target_idx];
          VEC_T vec_result;
#pragma unroll
          for (int v = 0; v < vec_size; ++v) {
            vec_result[v] = func(vec_input_val[v], vec_output_val[v]);
          }
          output_vec[target_idx] = vec_result;
        }
      }
    }

    if (chunk_idx == chunks_per_batch - 1) {
      const int64_t tail_start = vec_embedding_dim * vec_size;
      int64_t dense_val_idx = tail_start + item_id;
      for (; dense_val_idx < embedding_dim; dense_val_idx += local_size) {
        const int64_t dense_val_offset =
            batch_idx * embedding_dim + dense_val_idx;
        const scalar_t dense_val =
            reinterpret_cast<const scalar_t*>(dense_vec)[dense_val_offset];
        int64_t n_offset = dense_val_idx;
        for (; n_offset < range; n_offset += embedding_dim) {
          const int64_t target_idx = batch_start_offset + n_offset;
          const scalar_t input_val =
              reinterpret_cast<const scalar_t*>(input_vec)[target_idx];
          reinterpret_cast<scalar_t*>(output_vec)[target_idx] =
              func(input_val, dense_val);
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    int64_t vecs_per_chunk =
        ceil_div(embedding_dim / vec_size, chunks_per_batch);
    slm_dense =
        sycl_local_acc_t<VEC_T>(vecs_per_chunk == 0 ? 1 : vecs_per_chunk, cgh);
  }

  const VEC_T* input_vec;
  const VEC_T* dense_vec;
  VEC_T* output_vec;
  int64_t batch_size;
  int64_t embedding_dim;
  const int64_t* offsets;
  int64_t chunks_per_batch;
  const func_t func;
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
        func);                                                          \
    sycl_kernel_submit(                                                 \
        total_groups* GROUP_DIM,                                        \
        GROUP_DIM,                                                      \
        at::xpu::getCurrentSYCLQueue(),                                 \
        kfn);                                                           \
  } break;

template <typename scalar_t, typename func_t>
void nested_op_dense_kernel_impl(
    const scalar_t* input,
    const scalar_t* dense,
    scalar_t* output,
    int64_t batch_size,
    int64_t embedding_dim,
    const int64_t* input_offsets,
    func_t func) {
  int max_input_vec_size = memory::can_vectorize_up_to<scalar_t>(
      reinterpret_cast<const char*>(input));
  int max_dense_vec_size = memory::can_vectorize_up_to<scalar_t>(
      reinterpret_cast<const char*>(dense));
  int max_output_vec_size = memory::can_vectorize_up_to<scalar_t>(
      reinterpret_cast<const char*>(output));
  int max_vec_size =
      std::min({max_input_vec_size, max_dense_vec_size, max_output_vec_size});
  int vec_size{1};
  for (int size : {8, 4, 2}) {
    if (max_vec_size >= size && embedding_dim % size == 0) {
      vec_size = size;
      break;
    }
  }

  int64_t chunks_per_batch = (embedding_dim > SLM_TILING_THRESHOLD)
      ? ceil_div(embedding_dim, SLM_TILING_THRESHOLD)
      : 1;
  int64_t total_groups = batch_size * chunks_per_batch;
  if (total_groups == 0) {
    return;
  }

  constexpr int64_t GROUP_DIM{256};

  switch (vec_size) {
    LAUNCH_VEC_CASE(8)
    LAUNCH_VEC_CASE(4)
    LAUNCH_VEC_CASE(2)
    LAUNCH_VEC_CASE(1)
  }
}

template <typename scalar_t, typename func_t>
void _nested_op_dense_esuhm_kernel(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    func_t func) {
  NestedTensorImpl* self_ptr = get_nested_tensor_impl(self);
  const Tensor self_buffer = self_ptr->get_buffer();
  const Tensor offsets = self_ptr->get_storage_offsets();
  const int64_t batch_size = other.size(0);
  const int64_t embedding_size = other.size(-1);
  Tensor result_buffer = get_nested_tensor_impl(result)->get_buffer();

  Tensor self_numel = at::tensor({self_ptr->numel()}, offsets.options());
  Tensor result_offsets = at::cat({offsets, self_numel}).to(at::kXPU);

  nested_op_dense_kernel_impl<scalar_t, func_t>(
      self_buffer.const_data_ptr<scalar_t>(),
      other.const_data_ptr<scalar_t>(),
      result_buffer.data_ptr<scalar_t>(),
      batch_size,
      embedding_size,
      result_offsets.const_data_ptr<int64_t>(),
      func);
}

void _nested_op_dense_esuhm_xpu(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    const NESTED_DENSE_OP& op) {
  AT_DISPATCH_V2(
      self.scalar_type(),
      "_nested_op_dense_esuhm",
      AT_WRAP([&] {
        if (op == NESTED_DENSE_OP::ADD) {
          _nested_op_dense_esuhm_kernel<scalar_t>(
              result, self, other, AddFunctor<scalar_t>{});
        } else if (op == NESTED_DENSE_OP::MUL) {
          _nested_op_dense_esuhm_kernel<scalar_t>(
              result, self, other, MulFunctor<scalar_t>{});
        } else {
          TORCH_CHECK(
              false,
              "Unsupported NESTED_DENSE_OP in _nested_op_dense_esuhm_xpu");
        }
      }),
      AT_EXPAND(AT_ALL_TYPES),
      kHalf,
      kBFloat16);
}

} // namespace at::native::xpu
