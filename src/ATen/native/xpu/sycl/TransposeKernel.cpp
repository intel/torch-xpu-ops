/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <comm/SYCLContext.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <ATen/native/xpu/sycl/TransposeKernel.h>

#include <algorithm>
#include <numeric>

namespace at::native::xpu {

// SLM-tiled batch transpose kernel.
// Performs: dst[b][j][i] = src[b][i][j]
//   for b in [0, batch), i in [0, rows), j in [0, cols)
//
// Any copy between two dense tensors whose physical dimension orders differ
// by a swap of two contiguous groups can be expressed as a batch transpose
// and handled by this kernel.
//
// Optimizations:
// 1. 3D nd_range with batch as a separate dimension eliminates expensive
//    integer division by non-power-of-2 tile counts on every thread.
// 2. Type-aware SLM padding minimizes bank conflicts for 2-byte types.
// 3. FULL_TILE template fast-path eliminates all bounds checks.

static constexpr int TILE_DIM = 32;

template <typename scalar_t, int VEC_SIZE, bool FULL_TILE>
struct BatchTransposeFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr int WG_SIZE = 256;
  static constexpr int SLM_PAD = (sizeof(scalar_t) <= 2) ? 2 : 1;
  static constexpr int SLM_STRIDE = TILE_DIM + SLM_PAD;
  static constexpr int ROWS_PER_ITER = WG_SIZE * VEC_SIZE / TILE_DIM;
  using vec_t = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;

  void operator()(sycl::nd_item<3> item) const {
    int tx = item.get_local_id(2);
    int ty = item.get_local_id(1);
    int batch = item.get_group(0);
    int tile_y = item.get_group(1);
    int tile_x = item.get_group(2);

    int batch_off = batch * rows_ * cols_;

#pragma unroll
    for (int i = 0; i < TILE_DIM; i += ROWS_PER_ITER) {
      int src_row = tile_y * TILE_DIM + ty + i;
      int src_col = tile_x * TILE_DIM + tx * VEC_SIZE;

      if constexpr (FULL_TILE) {
        vec_t v = *reinterpret_cast<const vec_t*>(
            src_ + batch_off + src_row * cols_ + src_col);
#pragma unroll
        for (int k = 0; k < VEC_SIZE; k++) {
          slm_[(ty + i) * SLM_STRIDE + tx * VEC_SIZE + k] = v.val[k];
        }
      } else {
        if (src_row < rows_) {
#pragma unroll
          for (int k = 0; k < VEC_SIZE; k++) {
            if (src_col + k < cols_) {
              slm_[(ty + i) * SLM_STRIDE + tx * VEC_SIZE + k] =
                  src_[batch_off + src_row * cols_ + src_col + k];
            }
          }
        }
      }
    }

    sycl::group_barrier(item.get_group());

#pragma unroll
    for (int i = 0; i < TILE_DIM; i += ROWS_PER_ITER) {
      int dst_row = tile_x * TILE_DIM + ty + i;
      int dst_col = tile_y * TILE_DIM + tx * VEC_SIZE;

      if constexpr (FULL_TILE) {
        vec_t v;
#pragma unroll
        for (int k = 0; k < VEC_SIZE; k++) {
          v.val[k] = slm_[(tx * VEC_SIZE + k) * SLM_STRIDE + (ty + i)];
        }
        *reinterpret_cast<vec_t*>(
            dst_ + batch_off + dst_row * rows_ + dst_col) = v;
      } else {
        if (dst_row < cols_) {
#pragma unroll
          for (int k = 0; k < VEC_SIZE; k++) {
            if (dst_col + k < rows_) {
              dst_[batch_off + dst_row * rows_ + dst_col + k] =
                  slm_[(tx * VEC_SIZE + k) * SLM_STRIDE + (ty + i)];
            }
          }
        }
      }
    }
  }

  BatchTransposeFunctor(
      const scalar_t* src,
      scalar_t* dst,
      int rows,
      int cols)
      : src_(src),
        dst_(dst),
        rows_(rows),
        cols_(cols),
        slm_() {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    slm_ = sycl::local_accessor<scalar_t, 1>(
        sycl::range<1>(TILE_DIM * SLM_STRIDE), cgh);
  }

 private:
  const scalar_t* src_;
  scalar_t* dst_;
  int rows_;
  int cols_;
  sycl::local_accessor<scalar_t, 1> slm_;
};

template <typename scalar_t, int VEC_SIZE, bool FULL_TILE>
static void launch_transpose_kernel(
    const scalar_t* src,
    scalar_t* dst,
    int batch_size,
    int rows,
    int cols) {
  constexpr int kROWS_PER_ITER = BatchTransposeFunctor<scalar_t, VEC_SIZE, FULL_TILE>::ROWS_PER_ITER;
  int num_tiles_x = (cols + TILE_DIM - 1) / TILE_DIM;
  int num_tiles_y = (rows + TILE_DIM - 1) / TILE_DIM;

  sycl::range<3> local_range(1, kROWS_PER_ITER, TILE_DIM / VEC_SIZE);
  sycl::range<3> global_range(
      static_cast<size_t>(batch_size),
      static_cast<size_t>(num_tiles_y) * kROWS_PER_ITER,
      static_cast<size_t>(num_tiles_x) * (TILE_DIM / VEC_SIZE));

  auto ker = BatchTransposeFunctor<scalar_t, VEC_SIZE, FULL_TILE>(
      src, dst, rows, cols);

  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), ker);
}

template <typename scalar_t>
static void dispatch_transpose(
    const scalar_t* src,
    scalar_t* dst,
    int batch_size,
    int rows,
    int cols) {
  bool full_tile = (rows % TILE_DIM == 0) && (cols % TILE_DIM == 0);

  constexpr int kMaxVec =
      sizeof(scalar_t) * 4 <= 16 ? 4 : (sizeof(scalar_t) * 2 <= 16 ? 2 : 1);

  int vec_size = 1;
  if constexpr (kMaxVec >= 4) {
    if (cols % 4 == 0 && rows % 4 == 0)
      vec_size = 4;
    else if (cols % 2 == 0 && rows % 2 == 0)
      vec_size = 2;
  } else if constexpr (kMaxVec >= 2) {
    if (cols % 2 == 0 && rows % 2 == 0)
      vec_size = 2;
  }

#define LAUNCH(VS, FT) \
  launch_transpose_kernel<scalar_t, VS, FT>(src, dst, batch_size, rows, cols)

  if (full_tile) {
    if constexpr (kMaxVec >= 4) {
      if (vec_size == 4) { LAUNCH(4, true); return; }
    }
    if constexpr (kMaxVec >= 2) {
      if (vec_size == 2) { LAUNCH(2, true); return; }
    }
    LAUNCH(1, true);
  } else {
    if constexpr (kMaxVec >= 4) {
      if (vec_size == 4) { LAUNCH(4, false); return; }
    }
    if constexpr (kMaxVec >= 2) {
      if (vec_size == 2) { LAUNCH(2, false); return; }
    }
    LAUNCH(1, false);
  }
#undef LAUNCH
}

// ============================================================
// Detection: determine if a copy src->dst is a batch transpose.
//
// Algorithm:
// 1. Check both tensors are non-overlapping and dense (using PyTorch's
//    built-in is_non_overlapping_and_dense()).
// 2. Compute physical dimension order by sorting dims by stride
//    (descending), skipping size-1 dims.
// 3. Find the longest common prefix (batch dims).
// 4. Check if the remaining dims in dst are a swap of two contiguous
//    groups from src: src=[...batch | groupA | groupB]
//                     dst=[...batch | groupB | groupA]
// 5. Compute batch = product(batch_dims), rows = product(groupA),
//    cols = product(groupB).
// ============================================================

static bool detect_batch_transpose(
    const at::Tensor& src,
    const at::Tensor& dst,
    int& batch_size,
    int& rows,
    int& cols) {
  int ndim = src.dim();
  if (ndim < 2)
    return false;
  if (src.scalar_type() != dst.scalar_type())
    return false;
  if (!src.sizes().equals(dst.sizes()))
    return false;

  // Both must be densely packed (no holes in memory).
  // This is equivalent to: sorting dims by stride and checking
  // stride[i] == size[i+1] * stride[i+1] for all adjacent pairs.
  if (!src.is_non_overlapping_and_dense())
    return false;
  if (!dst.is_non_overlapping_and_dense())
    return false;

  auto sizes = src.sizes();
  auto src_strides = src.strides();
  auto dst_strides = dst.strides();

  // Build physical dim order for each tensor (skip size-1 dims)
  c10::SmallVector<int, 8> src_order, dst_order;
  for (int i = 0; i < ndim; i++) {
    if (sizes[i] > 1) {
      src_order.push_back(i);
      dst_order.push_back(i);
    }
  }

  std::sort(src_order.begin(), src_order.end(), [&](int a, int b) {
    return src_strides[a] > src_strides[b];
  });
  std::sort(dst_order.begin(), dst_order.end(), [&](int a, int b) {
    return dst_strides[a] > dst_strides[b];
  });

  int n = static_cast<int>(src_order.size());
  if (n < 2)
    return false;

  // Find longest common prefix in physical order (batch dimensions)
  int batch_end = 0;
  while (batch_end < n && src_order[batch_end] == dst_order[batch_end])
    batch_end++;

  int remaining = n - batch_end;
  if (remaining < 2)
    return false;

  // Check if dst_remaining is src_remaining with two groups swapped:
  //   src: [groupA (split items) | groupB (remaining-split items)]
  //   dst: [groupB | groupA]
  for (int split = 1; split < remaining; split++) {
    int groupB_len = remaining - split;
    bool match = true;
    // dst[batch_end .. batch_end+groupB_len) == src[batch_end+split .. n)
    for (int i = 0; i < groupB_len && match; i++) {
      if (dst_order[batch_end + i] != src_order[batch_end + split + i])
        match = false;
    }
    // dst[batch_end+groupB_len .. n) == src[batch_end .. batch_end+split)
    for (int i = 0; i < split && match; i++) {
      if (dst_order[batch_end + groupB_len + i] != src_order[batch_end + i])
        match = false;
    }
    if (match) {
      // Compute batch, rows (= product of groupA sizes), cols (= product of groupB sizes)
      int64_t b = 1, r = 1, c = 1;
      for (int i = 0; i < batch_end; i++)
        b *= sizes[src_order[i]];
      for (int i = 0; i < split; i++)
        r *= sizes[src_order[batch_end + i]];
      for (int i = 0; i < groupB_len; i++)
        c *= sizes[src_order[batch_end + split + i]];

      if (r < TILE_DIM || c < TILE_DIM)
        return false;

      // Occupancy check: ensure enough parallelism and SLM fits.
      constexpr int WG_SIZE = 256;
      int64_t num_tiles_x = (c + TILE_DIM - 1) / TILE_DIM;
      int64_t num_tiles_y = (r + TILE_DIM - 1) / TILE_DIM;
      int64_t total_wgs = b * num_tiles_y * num_tiles_x;

      int64_t simd = syclMaxSubGroupSize();
      int64_t sgs_per_wg = WG_SIZE / simd;
      int64_t total_sgs = total_wgs * sgs_per_wg;

      // 1) Total subgroups must fill all GPU thread slots
      int64_t thread_slots = syclGpuEuCount() * syclGpuHWThreadsPerEU();
      if (total_sgs < thread_slots)
        return false;

      // 2) SLM per WG must not exceed per-subslice budget at max
      //    thread-slot-limited concurrency
      int64_t elem_size = src.element_size();
      int64_t slm_pad = (elem_size <= 2) ? 2 : 1;
      int64_t slm_per_wg = TILE_DIM * (TILE_DIM + slm_pad) * elem_size;

      int64_t eu_per_xc = syclGpuEUCountPerSubslice();
      int64_t hw_thr = syclGpuHWThreadsPerEU();
      int64_t slots_per_xc = eu_per_xc * hw_thr;
      int64_t concurrent_wgs = slots_per_xc / sgs_per_wg;
      int64_t slm_per_wg_upbound = syclLocalMemSize() / concurrent_wgs;
      if (slm_per_wg > slm_per_wg_upbound)
        return false;

      batch_size = static_cast<int>(b);
      rows = static_cast<int>(r);
      cols = static_cast<int>(c);
      return true;
    }
  }

  return false;
}

bool can_use_channels_last_transpose_kernel(TensorIteratorBase& iter) {
  if (iter.ntensors() != 2)
    return false;
  const auto& dst = iter.tensor(0);
  const auto& src = iter.tensor(1);
  int batch_size, rows, cols;
  return detect_batch_transpose(src, dst, batch_size, rows, cols);
}

void channels_last_transpose_kernel(TensorIteratorBase& iter) {
  const auto& dst = iter.tensor(0);
  const auto& src = iter.tensor(1);

  int batch_size, rows, cols;
  detect_batch_transpose(src, dst, batch_size, rows, cols);

  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf, kBFloat16, src.scalar_type(), "batch_transpose_xpu", [&] {
        const scalar_t* src_data = src.const_data_ptr<scalar_t>();
        scalar_t* dst_data = dst.mutable_data_ptr<scalar_t>();
        dispatch_transpose(src_data, dst_data, batch_size, rows, cols);
      });
}

} // namespace at::native::xpu
