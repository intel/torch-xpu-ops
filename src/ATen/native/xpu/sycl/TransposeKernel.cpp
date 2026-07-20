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

// SLM-tiled batch transpose kernel.
// Performs: dst[b][j][i] = src[b][i][j] for b in [0, batch), i in [0, rows), j in [0, cols)
//
// Any copy that can be viewed as a 3D tensor with the last two dims
// transposed can use this kernel (channels_last conversion is a special case).
//
// Optimizations:
// 1. 3D nd_range with batch as a separate dimension eliminates expensive
//    integer division by non-power-of-2 tile counts on every thread.
// 2. Type-aware SLM padding minimizes bank conflicts for 2-byte types.
// 3. FULL_TILE template fast-path eliminates all bounds checks.

static constexpr int TILE_DIM = 32;

// SLM padding: +2 for 2-byte types avoids paired-element bank conflicts,
// +1 for 4-byte types is the classic conflict-free padding.
template <typename scalar_t>
static constexpr int SLM_PAD = (sizeof(scalar_t) <= 2) ? 2 : 1;

template <typename scalar_t>
static constexpr int SLM_STRIDE = TILE_DIM + SLM_PAD<scalar_t>;

// Work-group thread count = 256, regardless of VEC_SIZE:
//   local(dim1) = BRV rows, local(dim2) = TILE_DIM/VEC_SIZE cols
template <int VEC_SIZE>
static constexpr int BRV = 256 * VEC_SIZE / TILE_DIM;

template <typename scalar_t, int VEC_SIZE, bool FULL_TILE>
struct BatchTransposeFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr int kBRV = BRV<VEC_SIZE>;
  static constexpr int kSLM_STRIDE = SLM_STRIDE<scalar_t>;
  using vec_t = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;

  void operator()(sycl::nd_item<3> item) const {
    int tx = item.get_local_id(2);   // col within tile
    int ty = item.get_local_id(1);   // row within tile
    int batch = item.get_group(0);   // batch index (no division!)
    int tile_y = item.get_group(1);  // row-tile index
    int tile_x = item.get_group(2);  // col-tile index

    int batch_off = batch * rows_ * cols_;

    // READ: coalesced vec loads along cols (source fast dim)
#pragma unroll
    for (int i = 0; i < TILE_DIM; i += kBRV) {
      int src_row = tile_y * TILE_DIM + ty + i;
      int src_col = tile_x * TILE_DIM + tx * VEC_SIZE;

      if constexpr (FULL_TILE) {
        vec_t v = *reinterpret_cast<const vec_t*>(
            src_ + batch_off + src_row * cols_ + src_col);
#pragma unroll
        for (int k = 0; k < VEC_SIZE; k++) {
          slm_[(ty + i) * kSLM_STRIDE + tx * VEC_SIZE + k] = v.val[k];
        }
      } else {
        if (src_row < rows_) {
#pragma unroll
          for (int k = 0; k < VEC_SIZE; k++) {
            if (src_col + k < cols_) {
              slm_[(ty + i) * kSLM_STRIDE + tx * VEC_SIZE + k] =
                  src_[batch_off + src_row * cols_ + src_col + k];
            }
          }
        }
      }
    }

    sycl::group_barrier(item.get_group());

    // WRITE: transposed gather from SLM, coalesced vec stores
#pragma unroll
    for (int i = 0; i < TILE_DIM; i += kBRV) {
      int dst_row = tile_x * TILE_DIM + ty + i;
      int dst_col = tile_y * TILE_DIM + tx * VEC_SIZE;

      if constexpr (FULL_TILE) {
        vec_t v;
#pragma unroll
        for (int k = 0; k < VEC_SIZE; k++) {
          v.val[k] = slm_[(tx * VEC_SIZE + k) * kSLM_STRIDE + (ty + i)];
        }
        *reinterpret_cast<vec_t*>(
            dst_ + batch_off + dst_row * rows_ + dst_col) = v;
      } else {
        if (dst_row < cols_) {
#pragma unroll
          for (int k = 0; k < VEC_SIZE; k++) {
            if (dst_col + k < rows_) {
              dst_[batch_off + dst_row * rows_ + dst_col + k] =
                  slm_[(tx * VEC_SIZE + k) * kSLM_STRIDE + (ty + i)];
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
        sycl::range<1>(TILE_DIM * kSLM_STRIDE), cgh);
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
  constexpr int kBRV = BRV<VEC_SIZE>;
  int num_tiles_x = (cols + TILE_DIM - 1) / TILE_DIM;
  int num_tiles_y = (rows + TILE_DIM - 1) / TILE_DIM;

  // 3D: (batch, row-tiles * BRV, col-tiles * TILE_DIM/VEC_SIZE)
  sycl::range<3> local_range(1, kBRV, TILE_DIM / VEC_SIZE);
  sycl::range<3> global_range(
      static_cast<size_t>(batch_size),
      static_cast<size_t>(num_tiles_y) * kBRV,
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

// ============================================================
// Detection: identify if a copy is a batch transpose of last 2 dims.
//
// A copy src -> dst is a batch transpose if the tensors can be viewed as
// 3D [batch, rows, cols] where:
//   - src is contiguous in [batch, rows, cols] order
//   - dst is contiguous in [batch, cols, rows] order (last 2 dims swapped)
// or vice versa.
//
// This covers:
//   - 4D channels_last <-> contiguous (NHWC <-> NCHW)
//   - Any ndim>=2 tensor with last 2 dims transposed
// ============================================================

// Try to decompose a copy into batch_transpose(batch, rows, cols).
// Returns true and writes batch_size/rows/cols on success.
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

  // --- Special case: 4D channels_last <-> contiguous ---
  if (ndim == 4) {
    int N = src.size(0);
    int C = src.size(1);
    int HW = src.size(2) * src.size(3);

    if (HW >= TILE_DIM && C >= TILE_DIM) {
      // NHWC -> NCHW
      if (src.is_contiguous(at::MemoryFormat::ChannelsLast) &&
          dst.is_contiguous(at::MemoryFormat::Contiguous)) {
        batch_size = N;
        rows = HW;
        cols = C;
        return true;
      }
      // NCHW -> NHWC
      if (src.is_contiguous(at::MemoryFormat::Contiguous) &&
          dst.is_contiguous(at::MemoryFormat::ChannelsLast)) {
        batch_size = N;
        rows = C;
        cols = HW;
        return true;
      }
    }
  }

  // --- General case: ndim >= 2, last two dims transposed ---
  // Case A: src is contiguous, dst has last 2 strides swapped
  // src strides: [..., size(-1), 1]
  // dst strides: [..., 1, size(-2)]  (transposed last 2)
  auto try_detect = [&](const at::Tensor& contig,
                        const at::Tensor& transposed) -> bool {
    if (!contig.is_contiguous())
      return false;

    int M = contig.size(ndim - 2);
    int N = contig.size(ndim - 1);
    if (M < TILE_DIM || N < TILE_DIM)
      return false;

    auto t_strides = transposed.strides();
    // Last two strides must be [1, M] (transposed)
    if (t_strides[ndim - 1] != M || t_strides[ndim - 2] != 1)
      return false;

    // Leading dims must be contiguous batch: stride[i] = product of sizes below
    int64_t expected_stride = static_cast<int64_t>(M) * N;
    for (int i = ndim - 3; i >= 0; i--) {
      if (t_strides[i] != expected_stride)
        return false;
      expected_stride *= contig.size(i);
    }

    // Compute batch
    int batch = 1;
    for (int i = 0; i < ndim - 2; i++)
      batch *= contig.size(i);

    batch_size = batch;
    rows = M;
    cols = N;
    return true;
  };

  // src contiguous, dst transposed
  if (try_detect(src, dst))
    return true;
  // dst contiguous, src transposed
  if (try_detect(dst, src)) {
    // swap rows/cols: src is [batch, cols, rows] contiguous, dst is [batch, rows, cols] transposed
    // kernel reads src as [batch, cols, rows], so rows_param=cols, cols_param=rows
    std::swap(rows, cols);
    return true;
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
