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
#include <ATen/xpu/XPUContext.h>
#include <c10/util/Load.h>
#include <c10/util/llvmMathExtras.h>
#include <comm/SYCLContext.h>

#include <algorithm>
#include <type_traits>

namespace at::native::xpu {

template <typename scalar_t>
struct BinaryLayoutPointers {
  scalar_t* output;
  const scalar_t* direct;
  const scalar_t* transposed;
};

struct BinaryLayoutDescriptor {
  uint32_t rows = 0, cols = 0, leading = 0, batches = 1;
  int batch_dims = 0;
  bool direct_first = false;
  at::detail::IntDivider<uint32_t> batch_div[MAX_DIMS];
  uint32_t output_stride[MAX_DIMS] = {}, direct_stride[MAX_DIMS] = {},
           transposed_stride[MAX_DIMS] = {};

  template <typename scalar_t>
  C10_ALWAYS_INLINE BinaryLayoutPointers<scalar_t> tile_pointers(
      BinaryLayoutPointers<scalar_t> data,
      uint32_t batch,
      uint32_t row,
      uint32_t col) const {
    uint32_t output = row * cols + col;
    uint32_t direct = output;
    uint32_t transposed = col * leading + row;
#pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim < batch_dims) {
        const auto coordinate = batch_div[dim].divmod(batch);
        batch = coordinate.div;
        output += coordinate.mod * output_stride[dim];
        direct += coordinate.mod * direct_stride[dim];
        transposed += coordinate.mod * transposed_stride[dim];
      }
    }
    return {
        data.output + output,
        data.direct + direct,
        data.transposed + transposed};
  }
};

// The tiled kernel stages the transposed input in column-major order. Each
// matrix has a whole number of 8-element chunks, including any final tail.
struct BinaryLayoutTile {
  static constexpr uint32_t chunk = 8, max_elements = 4096;
  uint32_t rows = 0, cols = 0, slm_pitch = 0, chunks = 0;
  uint32_t row_tiles = 0, col_tiles = 0;
  at::detail::IntDivider<uint32_t> div_rows, div_cols;
  int transposed_width = 0, output_width = 0;
  bool batch_fast = false;
};

// Recognizes O = f(D, T) or f(T, D) where the output O and the direct input D
// are row-major rows x cols matrices (C axes contiguous, then R axes), the
// transposed input T is column-major with leading dimension `leading`, and
// the remaining slower axes are independently strided batches. TI axes are
// ordered fastest first; size-1 axes are dropped. The caller establishes that
// the output is dense in this order.
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
  uint32_t size[MAX_DIMS], stride[2][MAX_DIMS];
  int dims = 0;
  for (int dim = 0; dim < iter.ndim(); ++dim) {
    if (iter.shape()[dim] == 1) {
      continue;
    }
    for (int input = 0; input < 2; ++input) {
      const int64_t bytes = iter.strides(input + 1)[dim];
      if (bytes < 0 || bytes % element) {
        return false;
      }
      stride[input][dims] = bytes / element;
    }
    size[dims++] = iter.shape()[dim];
  }
  if (dims < 2) {
    return false;
  }
  // T is the input whose fastest axis is strided. C axes: stride cols in D
  // (and O), leading * cols in T; R axes: cols * rows in D, rows in T, with
  // each T column within `leading`. Later axes are batches.
  const int direct = stride[0][0] == 1 ? 0 : 1, transposed = 1 - direct;
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
  const auto out_end = out_begin + out.storage().nbytes();
  for (int arg = 1; arg < 3; ++arg) {
    const auto& in = iter.tensor(arg);
    const auto in_begin = reinterpret_cast<uintptr_t>(in.storage().data());
    if (in_begin < out_end && out_begin < in_begin + in.storage().nbytes() &&
        !(arg == direct + 1 && iter.data_ptr(arg) == iter.data_ptr(0) &&
          out.const_data_ptr() == in.const_data_ptr() &&
          out.sizes() == in.sizes() && out.strides() == in.strides())) {
      return false;
    }
  }
  result.rows = rows;
  result.cols = cols;
  result.leading = leading;
  result.direct_first = direct == 0;
  result.batch_dims = dims - dim;
  uint32_t output_stride = rows * cols;
  for (int b = 0; b < result.batch_dims; ++b) {
    const uint32_t extent = size[dim + b];
    result.batch_div[b] = at::detail::IntDivider<uint32_t>(extent);
    result.batches *= extent;
    result.output_stride[b] = output_stride;
    result.direct_stride[b] = stride[direct][dim + b];
    result.transposed_stride[b] = stride[transposed][dim + b];
    output_stride *= extent;
  }
  return true;
}

static inline BinaryLayoutTile make_binary_layout_tile(
    const TensorIteratorBase& iter,
    const BinaryLayoutDescriptor& d) {
  constexpr auto chunk = BinaryLayoutTile::chunk;
  constexpr auto budget = BinaryLayoutTile::max_elements;
  const bool short_rows = d.rows <= 64;
  const int direct = d.direct_first ? 1 : 2;
  const auto element = iter.element_size(0);
  auto alignment = [&](int arg, const uint32_t* batch_strides) {
    auto address = reinterpret_cast<uintptr_t>(iter.data_ptr(arg));
    for (int b = 0; b < d.batch_dims; ++b) {
      address |= batch_strides[b] * element;
    }
    return address;
  };
  const auto transposed_alignment = alignment(3 - direct, d.transposed_stride);
  const auto output_alignment =
      alignment(0, d.output_stride) | alignment(direct, d.direct_stride);
  // An aligned chunk stays within one line or spans contiguous full lines.
  auto width = [&](uintptr_t alignment,
                   uint32_t inner,
                   uint32_t stride,
                   uint32_t outer) {
    for (int v = chunk; v > 1; v /= 2) {
      const bool lines = inner % chunk == 0 && stride % v == 0;
      const bool run = inner == stride && (outer * stride) % v == 0;
      if (alignment % (v * element) == 0 && (lines || run)) {
        return v;
      }
    }
    return 0;
  };
  // Sweep a few broadcast batches fastest to reuse each transposed tile.
  const bool batch_fast = d.batch_dims > 0 && d.batches <= 64 &&
      std::all_of(d.transposed_stride,
                  d.transposed_stride + d.batch_dims,
                  [](uint32_t stride) { return stride == 0; });
  auto make_tile = [&](uint32_t rows, uint32_t cols) {
    BinaryLayoutTile t;
    t.rows = rows;
    t.cols = cols;
    t.row_tiles = (d.rows + rows - 1) / rows;
    t.col_tiles = (d.cols + cols - 1) / cols;
    t.chunks = (rows * cols + chunk - 1) / chunk;
    t.transposed_width = width(transposed_alignment, rows, d.leading, cols);
    t.output_width = width(output_alignment, cols, d.cols, rows);
    // Add a word to even 32-bit strides to reduce SLM bank conflicts.
    t.slm_pitch = rows % 4 == 0 ? rows + 2 : rows;
    t.div_rows = at::detail::IntDivider<uint32_t>(rows);
    t.div_cols = at::detail::IntDivider<uint32_t>(cols);
    t.batch_fast = batch_fast;
    return t;
  };

  // Cover the short axis when it benefits from staging a whole extent.
  auto pow2_floor = [](uint32_t x) { return 1u << c10::llvm::Log2_32(x); };
  if (short_rows && d.rows >= 3) {
    return make_tile(d.rows, std::min(d.cols, pow2_floor(budget / d.rows)));
  }
  if (!short_rows && d.cols <= 256 && d.cols % 32 != 0) {
    const auto t =
        make_tile(std::min(d.rows, pow2_floor(budget / d.cols)), d.cols);
    // Narrow output tiles need enough groups and full-width accesses.
    if (t.row_tiles * t.col_tiles * d.batches >= 2048 &&
        t.transposed_width == chunk && t.output_width == chunk) {
      return t;
    }
  }
  // Keep enough columns to fill 256 chunks even for short transposed rows.
  constexpr uint32_t regular_elements = 2048;
  const uint32_t rows = std::min(d.rows, 32u);
  return make_tile(rows, std::min(d.cols, pow2_floor(regular_elements / rows)));
}

// Each work-item stages a chunk along the transposed input's contiguous axis,
// then reads a chunk in output order after the work-group barrier.
template <typename func_t>
struct LayoutBinaryKernelTiled : __SYCL_KER_CONFIG_CONVENTION__ {
  using scalar_t = typename function_traits<func_t>::result_type;
  static constexpr int chunk = BinaryLayoutTile::chunk;
  using chunk_t = memory::aligned_vector<scalar_t, chunk>;

  struct Side {
    uint32_t stride, outer, inner, tile_inner;
    int width;
  };

  struct Position {
    uint32_t outer, inner;

    void advance(uint32_t tile_inner) {
      if (++inner == tile_inner) {
        inner = 0;
        ++outer;
      }
    }
  };

  BinaryLayoutDescriptor d;
  BinaryLayoutTile t;
  BinaryLayoutPointers<scalar_t> data;
  func_t f;
  sycl_local_acc_t<scalar_t> slm;

  LayoutBinaryKernelTiled(
      BinaryLayoutDescriptor descriptor,
      BinaryLayoutTile tile,
      BinaryLayoutPointers<scalar_t> pointers,
      func_t func)
      : d(descriptor), t(tile), data(pointers), f(func) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    slm = sycl_local_acc_t<scalar_t>(t.slm_pitch * t.cols, cgh);
  }

  template <int width, bool store, typename ptr_t>
  static void move_vectors(ptr_t* p, chunk_t& values) {
    using vec_t = memory::aligned_vector<scalar_t, width>;
#pragma unroll
    for (int k = 0; k < chunk; k += width) {
      if constexpr (store) {
        vec_t v;
#pragma unroll
        for (int j = 0; j < width; ++j) {
          v[j] = values[k + j];
        }
        *reinterpret_cast<vec_t*>(p + k) = v;
      } else {
        const auto v = memory::load_vector<width>(p + k, 0);
#pragma unroll
        for (int j = 0; j < width; ++j) {
          values[k + j] = v[j];
        }
      }
    }
  }

  template <bool store, typename ptr_t>
  static void move_chunk(
      ptr_t* p,
      const Side& side,
      Position pos,
      uint32_t start,
      chunk_t& values) {
    // A chunk must stay within one valid line or within the valid prefix
    // of a tile made of contiguous full lines.
    const bool full = side.tile_inner % chunk == 0
        ? pos.outer < side.outer && pos.inner + chunk <= side.inner
        : start + chunk <= side.outer * side.inner;
    if (side.width && full) {
      p += pos.outer * side.stride + pos.inner;
      if (side.width == 8) {
        move_vectors<8, store>(p, values);
      } else if (side.width == 4) {
        move_vectors<4, store>(p, values);
      } else {
        move_vectors<2, store>(p, values);
      }
      return;
    }
#pragma unroll
    for (int j = 0; j < chunk; ++j) {
      if (j > 0) {
        pos.advance(side.tile_inner);
      }
      const bool valid = pos.outer < side.outer && pos.inner < side.inner;
      const uint32_t offset = pos.outer * side.stride + pos.inner;
      if constexpr (store) {
        if (valid) {
          p[offset] = values[j];
        }
      } else {
        values[j] = valid ? c10::load(p + offset) : scalar_t(0);
      }
    }
  }

  [[sycl::reqd_sub_group_size(32)]] void operator()(
      sycl::nd_item<3> item) const {
    const uint32_t g0 = item.get_group(0);
    const uint32_t g1 = item.get_group(1);
    const uint32_t g2 = item.get_group(2);
    const uint32_t row_start = (t.batch_fast ? g0 : g1) * t.rows;
    const uint32_t col_start = (t.batch_fast ? g1 : g2) * t.cols;
    const uint32_t batch = t.batch_fast ? g2 : g0;
    const uint32_t rows = sycl::min(t.rows, d.rows - row_start);
    const uint32_t cols = sycl::min(t.cols, d.cols - col_start);
    const uint32_t lid = item.get_local_id(2);
    const bool active = lid < t.chunks;
    const uint32_t start = lid * chunk;
    const uint32_t area = t.rows * t.cols;
    const Side direct{d.cols, rows, cols, t.cols, t.output_width};
    const Side transposed{d.leading, cols, rows, t.rows, t.transposed_width};
    const auto rc = t.div_cols.divmod(start);
    const auto cr = t.div_rows.divmod(start);
    const Position direct_pos{rc.div, rc.mod};
    const Position transposed_pos{cr.div, cr.mod};

    const auto tile = d.tile_pointers(data, batch, row_start, col_start);
    chunk_t dv;
    if (active) {
      chunk_t tv;
      move_chunk<false>(tile.direct, direct, direct_pos, start, dv);
      move_chunk<false>(tile.transposed, transposed, transposed_pos, start, tv);
      const uint32_t pad = t.slm_pitch - t.rows;
      if (t.rows % chunk == 0) {
        const uint32_t s = start + cr.div * pad;
#pragma unroll
        for (int j = 0; j < chunk; ++j) {
          slm[s + j] = tv[j];
        }
      } else {
#pragma unroll
        for (int j = 0; j < chunk; ++j) {
          if (start + j < area) {
            slm[start + j + (pad ? t.div_rows.div(start + j) * pad : 0)] =
                tv[j];
          }
        }
      }
    }
    sycl::group_barrier(item.get_group());
    if (!active) {
      return;
    }

    auto pos = direct_pos;
    chunk_t results;
#pragma unroll
    for (int j = 0; j < chunk; ++j) {
      if (j > 0) {
        pos.advance(t.cols);
      }
      if (pos.outer < t.rows) {
        const scalar_t v = slm[pos.inner * t.slm_pitch + pos.outer];
        results[j] = f(d.direct_first ? dv[j] : v, d.direct_first ? v : dv[j]);
      }
    }
    move_chunk<true>(tile.output, direct, direct_pos, start, results);
  }
};

template <typename func_t>
static inline bool try_launch_layout_conflict_binary_kernel(
    const TensorIteratorBase& iter,
    const func_t& f) {
  using traits = function_traits<func_t>;
  static_assert(traits::arity == 2);
  using scalar_t = typename traits::result_type;
  using first_t = std::decay_t<typename traits::template arg<0>::type>;
  using second_t = std::decay_t<typename traits::template arg<1>::type>;
  constexpr bool is_low_precision = std::is_same_v<scalar_t, c10::Half> ||
      std::is_same_v<scalar_t, c10::BFloat16>;
  if constexpr (
      is_low_precision && std::is_same_v<scalar_t, first_t> &&
      std::is_same_v<scalar_t, second_t>) {
    BinaryLayoutDescriptor d;
    if (!make_binary_layout_descriptor(iter, d)) {
      return false;
    }
    const auto stream = c10::xpu::getCurrentXPUStream();
    auto& queue = stream.queue();
    const auto* prop = at::xpu::getDeviceProperties(stream.device_index());
    const auto& subgroups = prop->sub_group_sizes;
    if (std::find(subgroups.begin(), subgroups.end(), 32) == subgroups.end()) {
      return false;
    }
    const int direct = d.direct_first ? 1 : 2;
    const BinaryLayoutPointers<scalar_t> data{
        static_cast<scalar_t*>(iter.data_ptr(0)),
        static_cast<const scalar_t*>(iter.data_ptr(direct)),
        static_cast<const scalar_t*>(iter.data_ptr(3 - direct))};
    const BinaryLayoutTile t = make_binary_layout_tile(iter, d);
    const size_t group_size = (t.chunks + 31) / 32 * 32;
    // Resource limits leave these tensors on the existing generic path.
    if (group_size > prop->max_work_group_size ||
        t.slm_pitch * t.cols * sizeof(scalar_t) > prop->local_mem_size) {
      return false;
    }
    const auto groups = t.batch_fast
        ? sycl::range<3>(t.row_tiles, t.col_tiles, d.batches * group_size)
        : sycl::range<3>(d.batches, t.row_tiles, t.col_tiles * group_size);
    sycl_kernel_submit(
        groups,
        sycl::range<3>(1, 1, group_size),
        queue,
        LayoutBinaryKernelTiled<func_t>(d, t, data, f));
    return true;
  }
  return false;
}

} // namespace at::native::xpu
