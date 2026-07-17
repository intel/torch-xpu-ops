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

namespace at::native::xpu {

// Tiled transpose kernel for channels_last <-> contiguous conversion.
// Treats each batch element as a 2D matrix transpose:
//   NHWC -> NCHW: transpose (H*W, C) -> (C, H*W)
//   NCHW -> NHWC: transpose (C, H*W) -> (H*W, C)
//
// Uses shared local memory (SLM) with padding for bank-conflict-free
// access, vectorized global loads/stores, and a FULL_TILE fast path
// that eliminates bounds checks when dimensions are divisible by TILE_DIM.

static constexpr int TILE_DIM = 32;

// Work-group keeps 256 threads regardless of VEC_SIZE:
//   local(dim0) = BRV = 256 * VEC_SIZE / TILE_DIM
//   local(dim1) = TILE_DIM / VEC_SIZE
// Each thread processes TILE_DIM / BRV rows with VEC_SIZE cols per load.
template <int VEC_SIZE>
static constexpr int BRV = 256 * VEC_SIZE / TILE_DIM;

template <typename scalar_t, int VEC_SIZE, bool FULL_TILE>
struct ChannelsLastTransposeFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr int kBRV = BRV<VEC_SIZE>;
  using vec_t = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;

  void operator()(sycl::nd_item<2> item) const {
    int tx = item.get_local_id(1);
    int ty = item.get_local_id(0);
    int tile_x = item.get_group(1);
    int tile_y_and_batch = item.get_group(0);

    int batch = tile_y_and_batch / num_tiles_y_;
    int tile_y = tile_y_and_batch % num_tiles_y_;

    int batch_off_src = batch * rows_ * cols_;
    int batch_off_dst = batch * cols_ * rows_;

    // READ: coalesced vec loads along cols (fast dim of source)
#pragma unroll
    for (int i = 0; i < TILE_DIM; i += kBRV) {
      int src_row = tile_y * TILE_DIM + ty + i;
      int src_col = tile_x * TILE_DIM + tx * VEC_SIZE;

      if constexpr (FULL_TILE) {
        vec_t v = *reinterpret_cast<const vec_t*>(
            src_ + batch_off_src + src_row * cols_ + src_col);
#pragma unroll
        for (int k = 0; k < VEC_SIZE; k++) {
          slm_[(ty + i) * (TILE_DIM + 1) + tx * VEC_SIZE + k] = v.val[k];
        }
      } else {
        if (src_row < rows_) {
#pragma unroll
          for (int k = 0; k < VEC_SIZE; k++) {
            if (src_col + k < cols_) {
              slm_[(ty + i) * (TILE_DIM + 1) + tx * VEC_SIZE + k] =
                  src_[batch_off_src + src_row * cols_ + src_col + k];
            }
          }
        }
      }
    }

    sycl::group_barrier(item.get_group());

    // WRITE: gather from transposed SLM, coalesced vec stores along rows
#pragma unroll
    for (int i = 0; i < TILE_DIM; i += kBRV) {
      int dst_row = tile_x * TILE_DIM + ty + i;
      int dst_col = tile_y * TILE_DIM + tx * VEC_SIZE;

      if constexpr (FULL_TILE) {
        vec_t v;
#pragma unroll
        for (int k = 0; k < VEC_SIZE; k++) {
          v.val[k] = slm_[(tx * VEC_SIZE + k) * (TILE_DIM + 1) + (ty + i)];
        }
        *reinterpret_cast<vec_t*>(
            dst_ + batch_off_dst + dst_row * rows_ + dst_col) = v;
      } else {
        if (dst_row < cols_) {
#pragma unroll
          for (int k = 0; k < VEC_SIZE; k++) {
            if (dst_col + k < rows_) {
              dst_[batch_off_dst + dst_row * rows_ + dst_col + k] =
                  slm_[(tx * VEC_SIZE + k) * (TILE_DIM + 1) + (ty + i)];
            }
          }
        }
      }
    }
  }

  ChannelsLastTransposeFunctor(
      const scalar_t* src,
      scalar_t* dst,
      int batch_size,
      int rows,
      int cols,
      int num_tiles_y)
      : src_(src),
        dst_(dst),
        batch_size_(batch_size),
        rows_(rows),
        cols_(cols),
        num_tiles_y_(num_tiles_y),
        slm_() {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    slm_ = sycl::local_accessor<scalar_t, 1>(
        sycl::range<1>(TILE_DIM * (TILE_DIM + 1)), cgh);
  }

 private:
  const scalar_t* src_;
  scalar_t* dst_;
  int batch_size_;
  int rows_;
  int cols_;
  int num_tiles_y_;
  sycl::local_accessor<scalar_t, 1> slm_;
};

template <typename scalar_t, int VEC_SIZE, bool FULL_TILE>
static void launch_transpose_kernel(
    const scalar_t* src,
    scalar_t* dst,
    int batch_size,
    int rows,
    int cols) {
  constexpr int kBRV = BRV<VEC_SIZE>;
  int num_tiles_x = (cols + TILE_DIM - 1) / TILE_DIM;
  int num_tiles_y = (rows + TILE_DIM - 1) / TILE_DIM;

  sycl::range<2> local_range(kBRV, TILE_DIM / VEC_SIZE);
  sycl::range<2> global_range(
      static_cast<size_t>(batch_size) * num_tiles_y * kBRV,
      static_cast<size_t>(num_tiles_x) * (TILE_DIM / VEC_SIZE));

  auto ker = ChannelsLastTransposeFunctor<scalar_t, VEC_SIZE, FULL_TILE>(
      src, dst, batch_size, rows, cols, num_tiles_y);

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

  // Max vector width: 16 bytes, capped at 4 elements
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
      if (vec_size == 4) {
        LAUNCH(4, true);
        return;
      }
    }
    if constexpr (kMaxVec >= 2) {
      if (vec_size == 2) {
        LAUNCH(2, true);
        return;
      }
    }
    LAUNCH(1, true);
  } else {
    if constexpr (kMaxVec >= 4) {
      if (vec_size == 4) {
        LAUNCH(4, false);
        return;
      }
    }
    if constexpr (kMaxVec >= 2) {
      if (vec_size == 2) {
        LAUNCH(2, false);
        return;
      }
    }
    LAUNCH(1, false);
  }
#undef LAUNCH
}

bool can_use_channels_last_transpose_kernel(TensorIteratorBase& iter) {
  if (iter.ntensors() != 2)
    return false;

  const auto& dst = iter.tensor(0);
  const auto& src = iter.tensor(1);

  if (dst.dim() != 4)
    return false;
  if (dst.scalar_type() != src.scalar_type())
    return false;
  if (!dst.sizes().equals(src.sizes()))
    return false;

  int C = src.size(1);
  int HW = src.size(2) * src.size(3);

  // Skip if either dimension is too small for effective tiling
  if (HW < TILE_DIM || C < TILE_DIM)
    return false;

  // NHWC -> NCHW
  if (src.is_contiguous(at::MemoryFormat::ChannelsLast) &&
      dst.is_contiguous(at::MemoryFormat::Contiguous)) {
    return true;
  }
  // NCHW -> NHWC
  if (src.is_contiguous(at::MemoryFormat::Contiguous) &&
      dst.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    return true;
  }
  return false;
}

void channels_last_transpose_kernel(TensorIteratorBase& iter) {
  const auto& dst = iter.tensor(0);
  const auto& src = iter.tensor(1);

  int N = src.size(0);
  int C = src.size(1);
  int H = src.size(2);
  int W = src.size(3);
  int HW = H * W;

  bool nhwc_to_nchw =
      src.is_contiguous(at::MemoryFormat::ChannelsLast) &&
      dst.is_contiguous(at::MemoryFormat::Contiguous);

  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf, kBFloat16, src.scalar_type(), "channels_last_transpose_xpu", [&] {
        const scalar_t* src_data = src.const_data_ptr<scalar_t>();
        scalar_t* dst_data = dst.mutable_data_ptr<scalar_t>();

        if (nhwc_to_nchw) {
          dispatch_transpose(src_data, dst_data, N, HW, C);
        } else {
          dispatch_transpose(src_data, dst_data, N, C, HW);
        }
      });
}

} // namespace at::native::xpu
