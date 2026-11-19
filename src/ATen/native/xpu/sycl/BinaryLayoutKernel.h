/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <ATen/native/xpu/sycl/OffsetCalculator.h>
#include <c10/util/Load.h>
#include <c10/util/llvmMathExtras.h>
#include <comm/SYCLContext.h>
#include <algorithm>
#include <initializer_list>
#include <type_traits>

namespace at::native::xpu {

struct BinaryLayoutDescriptor {
  uint32_t rows = 0, cols = 0, leading = 0, batches = 1;
  int batch_dims = 0, direct = 0;
  at::detail::IntDivider<uint32_t> batch_div[MAX_DIMS];
  uint32_t batch_stride[3][MAX_DIMS] = {};
  int tile_rows = 0, tile_cols = 0, log_rows = 0, log_cols = 0;
  int swizzle_source = 0, swizzle_destination = 0;
  uint32_t swizzle_mask = 0;
};

// Linear nb x tr x tc tile for short transposed rows or short output rows.
// wt / wo: vector width (8, 4 or 2) of full 8-element chunks on the
// transposed side and on the direct-input/output side, or 0 for scalar access.
struct BinaryLayoutTile {
  uint32_t tr = 0, tc = 0, nb = 1, trp = 0, slm_mat = 0;
  at::detail::IntDivider<uint32_t> div_tr, div_tc, div_col_tiles, div_tiles,
      div_batch_groups;
  int wt = 0, wo = 0;
  bool batch_fast = false;
};

// Recognizes O = f(D, T) or f(T, D) where the output O and the direct input D
// are row-major rows x cols matrices (C axes contiguous, then R axes), the
// transposed input T is column-major with leading dimension `leading`, and
// the remaining slower axes are independently strided batches. TI axes are
// ordered fastest first; size-1 axes are dropped.
static inline bool make_binary_layout_descriptor(
    const TensorIteratorBase& iter,
    BinaryLayoutDescriptor& result) {
  // 32-bit indexing also bounds every element offset below INT32_MAX.
  if (iter.noutputs() != 1 || iter.ninputs() != 2 ||
      !iter.can_use_32bit_indexing() || iter.ndim() > MAX_DIMS ||
      iter.numel() <= 0) {
    return false;
  }
  const int64_t element = iter.element_size(0);
  if (iter.element_size(1) != element || iter.element_size(2) != element) {
    return false;
  }
  uint32_t size[MAX_DIMS], stride[3][MAX_DIMS];
  int dims = 0;
  int64_t dense = 1;
  for (int dim = 0; dim < iter.ndim(); ++dim) {
    if (iter.shape()[dim] == 1) {
      continue;
    }
    if (iter.strides(0)[dim] != dense * element) {
      return false;
    }
    for (int arg = 0; arg < 3; ++arg) {
      const int64_t bytes = iter.strides(arg)[dim];
      if (bytes < 0 || bytes % element) {
        return false;
      }
      stride[arg][dims] = bytes / element;
    }
    size[dims++] = iter.shape()[dim];
    dense *= iter.shape()[dim];
  }
  if (dims < 2) {
    return false;
  }
  // T is the input whose fastest axis is strided. C axes: stride cols in D
  // (and O), leading * cols in T; R axes: cols * rows in D, rows in T, with
  // each T column within `leading`. Later axes are batches.
  const int direct = stride[1][0] == 1 ? 1 : 2, transposed = 3 - direct;
  const uint64_t leading = stride[transposed][0];
  uint64_t cols = 1, rows = 1;
  int dim = 0;
  for (; dim < dims && stride[direct][dim] == cols &&
       stride[transposed][dim] == leading * cols;
       ++dim) {
    cols *= size[dim];
  }
  for (; dim < dims && stride[direct][dim] == cols * rows &&
       stride[transposed][dim] == rows && rows * size[dim] <= leading;
       ++dim) {
    rows *= size[dim];
  }
  if (rows < 2 || cols < 2) {
    return false;
  }
  // An input may share the output's memory only as the output itself (an
  // in-place direct operand). Compare storage byte ranges, not Storage
  // identity: separate Storages (from_blob, DLPack) can alias one allocation.
  // TI subiterators keep the full tensors, so this also excludes RAW hazards
  // across 32-bit splits.
  const auto& out = iter.tensor(0);
  const auto out_begin = reinterpret_cast<uintptr_t>(out.storage().data());
  for (int arg = 1; arg < 3; ++arg) {
    const auto& in = iter.tensor(arg);
    const auto in_begin = reinterpret_cast<uintptr_t>(in.storage().data());
    if (in_begin < out_begin + out.storage().nbytes() &&
        out_begin < in_begin + in.storage().nbytes() &&
        !(arg == direct && iter.data_ptr(arg) == iter.data_ptr(0) &&
          out.const_data_ptr() == in.const_data_ptr() &&
          out.sizes() == in.sizes() && out.strides() == in.strides())) {
      return false;
    }
  }
  result.rows = rows;
  result.cols = cols;
  result.leading = leading;
  result.direct = direct;
  result.batch_dims = dims - dim;
  for (int b = 0; b < result.batch_dims; ++b) {
    result.batch_div[b] = at::detail::IntDivider<uint32_t>(size[dim + b]);
    result.batches *= size[dim + b];
    for (int arg = 0; arg < 3; ++arg) {
      result.batch_stride[arg][b] = stride[arg][dim + b];
    }
  }
  int small = 2, log_small = 1;
  while (small < 32 && small < std::min(rows, cols)) {
    small *= 2;
    ++log_small;
  }
  result.tile_rows = rows <= cols ? small : 1024 / small;
  result.tile_cols = 1024 / result.tile_rows;
  result.log_rows = rows <= cols ? log_small : 10 - log_small;
  result.log_cols = 10 - result.log_rows;
  const int a = result.log_rows - 1;
  result.swizzle_source = a <= 3 ? 5 : a + 1;
  result.swizzle_destination = a <= 3 ? 0 : a - 3;
  result.swizzle_mask = (1u << (a <= 3 ? a + 1 : 8 - a)) - 1;
  return true;
}

// Chooses the tiled kernel (returns true) or keeps the 2x2 ownership kernel.
// The tiled kernel stages one T tile of at most 4096 elements; it pays off
// when the transposed rows are short and not a multiple of 16, or when output
// rows are short and not a multiple of 32 (measured on the add prototypes).
static inline bool make_binary_layout_tile(
    const TensorIteratorBase& iter,
    const BinaryLayoutDescriptor& d,
    BinaryLayoutTile& t) {
  constexpr uint32_t budget = 4096;
  auto pow2_floor = [](uint32_t x) { return 1u << c10::llvm::Log2_32(x); };
  const bool short_rows = d.rows <= 64;
  uint32_t tr, tc, nb = 1;
  if (short_rows) {
    if (d.rows % 16 == 0 || d.rows < 3 || d.rows == 4) {
      return false;
    }
    tr = d.rows;
    tc = std::min(d.cols, pow2_floor(budget / d.rows));
    // Whole small matrices: pack several batches into one tile.
    if (tc == d.cols && d.batch_dims == 1) {
      nb = std::min(d.batches, budget / (d.rows * d.cols));
    }
  } else if (d.cols <= 256 && d.cols % 32 != 0) {
    tc = d.cols;
    tr = std::min(d.rows, pow2_floor(budget / d.cols));
  } else {
    return false;
  }
  const uint64_t row_tiles = (d.rows + tr - 1) / tr;
  const uint64_t col_tiles = (d.cols + tc - 1) / tc;
  const uint64_t batch_groups = (d.batches + nb - 1) / nb;
  const uint64_t groups = row_tiles * col_tiles * batch_groups;
  // Too few long-row tiles cannot fill the device; the 2x2 kernel wins there.
  if (!short_rows && groups < 2048) {
    return false;
  }
  const int transposed = d.direct == 1 ? 2 : 1;
  bool broadcast = d.batch_dims > 0;
  for (int b = 0; b < d.batch_dims; ++b) {
    broadcast &= d.batch_stride[transposed][b] == 0;
  }
  // A chunk is contiguous when the tile's inner extent is a multiple of 8
  // (the chunk stays in one line) or when the whole tile is one contiguous
  // run. Vector width v also needs every tile start, line stride and batch
  // offset to be a multiple of v.
  auto width = [&](std::initializer_list<int> args,
                   uint32_t inner,
                   uint32_t stride,
                   bool run,
                   uint64_t tile_step) {
    for (uint32_t v : {8u, 4u, 2u}) {
      bool ok =
          (inner % 8 == 0 && stride % v == 0) || (run && tile_step % v == 0);
      for (int arg : args) {
        ok &= reinterpret_cast<uintptr_t>(iter.data_ptr(arg)) %
                (v * iter.element_size(arg)) ==
            0;
        for (int b = 0; b < d.batch_dims; ++b) {
          ok &= d.batch_stride[arg][b] % v == 0;
        }
      }
      if (ok) {
        return int(v);
      }
    }
    return 0;
  };
  const bool t_run = tr == d.leading &&
      (nb == 1 || d.batch_stride[transposed][0] == d.leading * d.cols);
  const bool o_run = tc == d.cols &&
      (nb == 1 ||
       (d.batch_stride[0][0] == d.rows * d.cols &&
        d.batch_stride[d.direct][0] == d.rows * d.cols));
  t.wt = width({transposed}, tr, d.leading, t_run, uint64_t(tc) * d.leading);
  t.wo = width({0, d.direct}, tc, d.cols, o_run, uint64_t(tr) * d.cols);
  if (!short_rows && (t.wt != 8 || t.wo != 8)) {
    return false;
  }
  t.tr = tr;
  t.tc = tc;
  t.nb = nb;
  // Odd 32-bit word stride between staged columns avoids SLM bank conflicts.
  t.trp = (tr % 2 || tr % 4 == 2) ? tr : tr + 2;
  t.slm_mat = t.trp * tc;
  t.div_tr = at::detail::IntDivider<uint32_t>(tr);
  t.div_tc = at::detail::IntDivider<uint32_t>(tc);
  t.div_col_tiles = at::detail::IntDivider<uint32_t>(col_tiles);
  t.div_tiles = at::detail::IntDivider<uint32_t>(row_tiles * col_tiles);
  t.div_batch_groups = at::detail::IntDivider<uint32_t>(batch_groups);
  // Few broadcast batches: sweep them fastest to reuse each T tile in cache.
  t.batch_fast = broadcast && d.batches <= 64;
  return true;
}

template <typename func_t>
struct LayoutBinaryKernel : __SYCL_KER_CONFIG_CONVENTION__ {
  using scalar_t = typename function_traits<func_t>::result_type;
  BinaryLayoutDescriptor d;
  scalar_t* out;
  const scalar_t* first;
  const scalar_t* second;
  func_t f;
  sycl_local_acc_t<uint32_t> slm;

  LayoutBinaryKernel(
      BinaryLayoutDescriptor descriptor,
      scalar_t* output,
      const scalar_t* a,
      const scalar_t* b,
      func_t func)
      : d(descriptor), out(output), first(a), second(b), f(func) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    slm = sycl_local_acc_t<uint32_t>(512, cgh);
  }

  uint32_t slm_index(uint32_t q) const {
    return (q & ~31u) |
        ((q & 31u) ^
         (((q >> d.swizzle_source) & d.swizzle_mask) << d.swizzle_destination));
  }

  [[sycl::reqd_sub_group_size(32)]] void operator()(
      sycl::nd_item<3> item) const {
    uint32_t batch = item.get_group(0);
    uint32_t out_base = 0, first_base = 0, second_base = 0;
    // Constant indices let the descriptor stay in registers.
#pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim < d.batch_dims) {
        const auto coordinate = d.batch_div[dim].divmod(batch);
        batch = coordinate.div;
        out_base += coordinate.mod * d.batch_stride[0][dim];
        first_base += coordinate.mod * d.batch_stride[1][dim];
        second_base += coordinate.mod * d.batch_stride[2][dim];
      }
    }
    const uint32_t direct_base = d.direct == 1 ? first_base : second_base;
    const uint32_t transposed_base = d.direct == 1 ? second_base : first_base;
    const uint32_t row_start = item.get_group(1) * d.tile_rows;
    const uint32_t col_start = item.get_group(2) * d.tile_cols;
    const int lid = item.get_local_id(2);
    const scalar_t* transposed = d.direct == 1 ? second : first;
    const scalar_t* direct = d.direct == 1 ? first : second;
    using vec_t = memory::aligned_vector<scalar_t, 2>;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const uint32_t q = lid + 256 * k;
      const uint32_t rp = q & (d.tile_rows / 2 - 1);
      const uint32_t lc = q >> (d.log_rows - 1);
      const uint32_t r = row_start + 2 * rp, c = col_start + lc;
      uint32_t word = 0;
      if (r < d.rows && c < d.cols) {
        const scalar_t* input =
            transposed + transposed_base + r + c * d.leading;
        if (r + 1 < d.rows &&
            reinterpret_cast<uintptr_t>(input) % alignof(vec_t) == 0) {
          const auto values = memory::load_vector<2>(input, 0);
          word = uint32_t(values[0].x) | (uint32_t(values[1].x) << 16);
        } else {
          word = c10::load(input).x;
          if (r + 1 < d.rows) {
            word |= uint32_t(c10::load(input + 1).x) << 16;
          }
        }
      }
      slm[slm_index(q)] = word;
    }
    // Every SLM word is initialized before any work-item reads its two words.
    sycl::group_barrier(item.get_group());
    const uint32_t lc = 2 * (lid & (d.tile_cols / 2 - 1));
    const uint32_t rp = lid >> (d.log_cols - 1);
    const uint32_t first_word = slm[slm_index((lc << (d.log_rows - 1)) | rp)];
    const uint32_t second_word =
        slm[slm_index(((lc + 1) << (d.log_rows - 1)) | rp)];
    const uint32_t c = col_start + lc;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint32_t r = row_start + 2 * rp + i;
      if (r < d.rows && c < d.cols) {
        const scalar_t* input = direct + direct_base + r * d.cols + c;
        scalar_t* output = out + out_base + r * d.cols + c;
        if (c + 1 < d.cols &&
            ((reinterpret_cast<uintptr_t>(input) |
              reinterpret_cast<uintptr_t>(output)) %
             alignof(vec_t)) == 0) {
          const auto values = memory::load_vector<2>(input, 0);
          vec_t results;
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const uint32_t word = j == 0 ? first_word : second_word;
            const scalar_t b(uint16_t(word >> (16 * i)), scalar_t::from_bits());
            results[j] = d.direct == 1 ? f(values[j], b) : f(b, values[j]);
          }
          *reinterpret_cast<vec_t*>(output) = results;
        } else {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            if (c + j < d.cols) {
              const scalar_t a = c10::load(input + j);
              const uint32_t word = j == 0 ? first_word : second_word;
              const scalar_t b(
                  uint16_t(word >> (16 * i)), scalar_t::from_bits());
              output[j] = d.direct == 1 ? f(a, b) : f(b, a);
            }
          }
        }
      }
    }
  }
};

// Stages one nb x tr x tc tile of the transposed operand T through SLM, so
// that T, the direct input D and the output O are all accessed along their
// contiguous axes. Work-item i owns tile elements [8i, 8i + 8) in each side's
// own order: (b, c, r) for T and (b, r, c) for D and O.
template <typename func_t>
struct LayoutBinaryKernelTiled : __SYCL_KER_CONFIG_CONVENTION__ {
  using scalar_t = typename function_traits<func_t>::result_type;
  // One chunk per work-item (up to 512 x 8 = 4096 elements) keeps heavy
  // functors free of register spills; smaller tiles launch fewer work-items.
  static constexpr int chunk = 8, max_group_size = 512;
  using chunk_t = memory::aligned_vector<scalar_t, chunk>;
  // One side of the tile: element (b, outer, inner) is at
  // p[b * batch + outer * stride + inner]; outer/inner are the valid extents.
  struct Side {
    uint32_t batch, stride, outer, inner;
    int width;
  };
  BinaryLayoutDescriptor d;
  BinaryLayoutTile t;
  scalar_t* out;
  const scalar_t* first;
  const scalar_t* second;
  func_t f;
  sycl_local_acc_t<scalar_t> slm;

  LayoutBinaryKernelTiled(
      BinaryLayoutDescriptor descriptor,
      BinaryLayoutTile tile,
      scalar_t* output,
      const scalar_t* a,
      const scalar_t* b,
      func_t func)
      : d(descriptor), t(tile), out(output), first(a), second(b), f(func) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    slm = sycl_local_acc_t<scalar_t>(t.nb * t.slm_mat, cgh);
  }

  // Splits tile element e into (b, outer, inner) and returns inner.
  template <bool transposed>
  uint32_t split(uint32_t e, uint32_t& b, uint32_t& outer) const {
    const auto q = (transposed ? t.div_tr : t.div_tc).divmod(e);
    b = 0;
    outer = q.div;
    if (t.nb > 1) {
      const auto bo = (transposed ? t.div_tc : t.div_tr).divmod(q.div);
      b = bo.div;
      outer = bo.mod;
    }
    return q.mod;
  }

  // Advances (b, outer, inner) to the next tile element.
  template <bool transposed>
  void next(uint32_t& b, uint32_t& outer, uint32_t& inner) const {
    if (++inner == (transposed ? t.tr : t.tc)) {
      inner = 0;
      if (++outer == (transposed ? t.tc : t.tr)) {
        outer = 0;
        ++b;
      }
    }
  }

  // A chunk is read or written as vectors only when all 8 elements are in
  // bounds (checking just its first element over-reads tails): within one
  // line when the inner extent is a multiple of 8, else below the valid
  // prefix of a contiguous tile. Other chunks take the masked scalar path.
  template <bool transposed>
  bool full(
      const Side& s,
      uint32_t e,
      uint32_t b,
      uint32_t outer,
      uint32_t inner,
      uint32_t mats,
      uint32_t valid) const {
    return s.width &&
        ((transposed ? t.tr : t.tc) % chunk == 0
             ? b < mats && outer < s.outer && inner + chunk <= s.inner
             : e + chunk <= valid);
  }

  template <int w, bool to_memory, typename ptr_t>
  static void move_vectors(ptr_t* p, chunk_t& v) {
    using vec_t = memory::aligned_vector<scalar_t, w>;
#pragma unroll
    for (int k = 0; k < chunk; k += w) {
      if constexpr (to_memory) {
        vec_t x;
#pragma unroll
        for (int j = 0; j < w; ++j) {
          x[j] = v[k + j];
        }
        *reinterpret_cast<vec_t*>(p + k) = x;
      } else {
        const auto x = memory::load_vector<w>(p + k, 0);
#pragma unroll
        for (int j = 0; j < w; ++j) {
          v[k + j] = x[j];
        }
      }
    }
  }

  // Loads chunk [e, e + 8) (elements outside the tensor read as zero) or,
  // if to_memory, stores its valid elements.
  template <bool transposed, bool to_memory, typename ptr_t>
  void move(
      ptr_t* p,
      const Side& s,
      uint32_t e,
      uint32_t mats,
      uint32_t valid,
      chunk_t& v) const {
    uint32_t b, o;
    uint32_t i = split<transposed>(e, b, o);
    if (full<transposed>(s, e, b, o, i, mats, valid)) {
      p += b * s.batch + o * s.stride + i;
      s.width == 8 ? move_vectors<8, to_memory>(p, v)
                   : (s.width == 4 ? move_vectors<4, to_memory>(p, v)
                                   : move_vectors<2, to_memory>(p, v));
      return;
    }
#pragma unroll
    for (int j = 0; j < chunk; ++j) {
      if (j > 0) {
        next<transposed>(b, o, i);
      }
      if constexpr (to_memory) {
        if (b < mats && o < s.outer && i < s.inner) {
          p[b * s.batch + o * s.stride + i] = v[j];
        }
      } else {
        v[j] = b < mats && o < s.outer && i < s.inner
            ? c10::load(p + b * s.batch + o * s.stride + i)
            : scalar_t(0);
      }
    }
  }

  [[sycl::reqd_sub_group_size(32)]] void operator()(
      sycl::nd_item<1> item) const {
    const uint32_t g = item.get_group(0);
    uint32_t tile, group;
    if (t.batch_fast) {
      const auto q = t.div_batch_groups.divmod(g);
      tile = q.div;
      group = q.mod;
    } else {
      const auto q = t.div_tiles.divmod(g);
      group = q.div;
      tile = q.mod;
    }
    const auto rc = t.div_col_tiles.divmod(tile);
    const uint32_t r0 = rc.div * t.tr, c0 = rc.mod * t.tc, b0 = group * t.nb;
    const uint32_t rows = sycl::min(t.tr, d.rows - r0);
    const uint32_t cols = sycl::min(t.tc, d.cols - c0);
    const uint32_t mats = sycl::min(t.nb, d.batches - b0);
    uint32_t batch = b0;
    uint32_t out_base = 0, first_base = 0, second_base = 0;
    // Constant indices let the descriptor stay in registers.
#pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim < d.batch_dims) {
        const auto coordinate = d.batch_div[dim].divmod(batch);
        batch = coordinate.div;
        out_base += coordinate.mod * d.batch_stride[0][dim];
        first_base += coordinate.mod * d.batch_stride[1][dim];
        second_base += coordinate.mod * d.batch_stride[2][dim];
      }
    }
    const bool first_direct = d.direct == 1;
    const scalar_t* transposed =
        (first_direct ? second + second_base : first + first_base) +
        c0 * d.leading + r0;
    const scalar_t* direct =
        (first_direct ? first + first_base : second + second_base) +
        r0 * d.cols + c0;
    scalar_t* output = out + out_base + r0 * d.cols + c0;
    // Only nb > 1 tiles (one batch dimension) use the batch strides.
    const Side ts{
        d.batch_stride[first_direct ? 2 : 1][0], d.leading, cols, rows, t.wt};
    const Side ds{
        d.batch_stride[first_direct ? 1 : 2][0], d.cols, rows, cols, t.wo};
    const Side os{d.batch_stride[0][0], d.cols, rows, cols, t.wo};
    const uint32_t e = item.get_local_id(0) * chunk;
    const uint32_t total = t.nb * t.tr * t.tc;
    const uint32_t valid = mats * rows * cols;
    const bool active = e < total;
    chunk_t dv;
    if (active) {
      // Issue both global loads before the barrier.
      chunk_t tv;
      move<false, false>(direct, ds, e, mats, valid, dv);
      move<true, false>(transposed, ts, e, mats, valid, tv);
      // Stage every tile element, including invalid ones, as [b][c][r] with
      // column stride trp; a chunk stays in one column when tr % 8 == 0.
      const uint32_t pad = t.trp - t.tr;
      if (t.tr % chunk == 0) {
        const uint32_t s = e + t.div_tr.div(e) * pad;
#pragma unroll
        for (int j = 0; j < chunk; ++j) {
          slm[s + j] = tv[j];
        }
      } else {
#pragma unroll
        for (int j = 0; j < chunk; ++j) {
          if (e + j < total) {
            slm[e + j + (pad ? t.div_tr.div(e + j) * pad : 0)] = tv[j];
          }
        }
      }
    }
    sycl::group_barrier(item.get_group());
    if (!active) {
      return;
    }
    uint32_t b, r;
    uint32_t c = split<false>(e, b, r);
    chunk_t y;
#pragma unroll
    for (int j = 0; j < chunk; ++j) {
      if (j > 0) {
        next<false>(b, r, c);
      }
      // b == nb only past the last tile element; read a dummy there.
      const scalar_t v = slm[b < t.nb ? b * t.slm_mat + c * t.trp + r : 0];
      // One functor call site per element keeps heavy functors small.
      y[j] = f(first_direct ? dv[j] : v, first_direct ? v : dv[j]);
    }
    move<false, true>(output, os, e, mats, valid, y);
  }
};

template <typename func_t>
static inline bool try_launch_layout_conflict_binary_kernel(
    TensorIteratorBase& iter,
    const func_t& f) {
  using traits = function_traits<func_t>;
  if constexpr (traits::arity != 2) {
    return false;
  } else {
    using scalar_t = typename traits::result_type;
    using first_t = std::decay_t<typename traits::template arg<0>::type>;
    using second_t = std::decay_t<typename traits::template arg<1>::type>;
    if constexpr (
        (std::is_same_v<scalar_t, c10::Half> ||
         std::is_same_v<scalar_t, c10::BFloat16>)&&std::
            is_same_v<scalar_t, first_t> &&
        std::is_same_v<scalar_t, second_t>) {
      BinaryLayoutDescriptor d;
      if (!make_binary_layout_descriptor(iter, d)) {
        return false;
      }
      auto& queue = getCurrentSYCLQueue();
      const auto device = queue.get_device();
      const auto subgroups =
          device.get_info<sycl::info::device::sub_group_sizes>();
      const auto max_group =
          device.get_info<sycl::info::device::max_work_group_size>();
      const auto local_mem =
          device.get_info<sycl::info::device::local_mem_size>();
      if (max_group < 256 || local_mem < 512 * sizeof(uint32_t) ||
          std::find(subgroups.begin(), subgroups.end(), 32) ==
              subgroups.end()) {
        return false;
      }
      auto* output = static_cast<scalar_t*>(iter.data_ptr(0));
      const auto* a = static_cast<const scalar_t*>(iter.data_ptr(1));
      const auto* b = static_cast<const scalar_t*>(iter.data_ptr(2));
      BinaryLayoutTile t;
      if (make_binary_layout_tile(iter, d, t) &&
          local_mem >= uint64_t(t.nb) * t.slm_mat * sizeof(scalar_t) &&
          max_group >= LayoutBinaryKernelTiled<func_t>::max_group_size) {
        using kernel_t = LayoutBinaryKernelTiled<func_t>;
        const int64_t groups =
            int64_t(t.div_tiles.divisor) * t.div_batch_groups.divisor;
        const int chunks =
            (t.nb * t.tr * t.tc + kernel_t::chunk - 1) / kernel_t::chunk;
        const int group_size = (chunks + 31) / 32 * 32;
        sycl_kernel_submit(
            groups * group_size,
            group_size,
            queue,
            kernel_t(d, t, output, a, b, f));
        return true;
      }
      sycl_kernel_submit(
          sycl::range<3>(
              d.batches,
              (uint64_t(d.rows) + d.tile_rows - 1) / d.tile_rows,
              ((uint64_t(d.cols) + d.tile_cols - 1) / d.tile_cols) * 256),
          sycl::range<3>(1, 1, 256),
          queue,
          LayoutBinaryKernel<func_t>(d, output, a, b, f));
      return true;
    } else {
      return false;
    }
  }
}

} // namespace at::native::xpu
