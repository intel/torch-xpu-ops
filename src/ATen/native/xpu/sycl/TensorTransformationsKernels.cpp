/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch_v2.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/ceil_div.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/xpu/sycl/IntegerDivider.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <ATen/native/xpu/sycl/OffsetCalculator.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/util/irange.h>
#include <comm/SYCLContext.h>
#include <comm/xpu_aten.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>

#include <ATen/native/xpu/sycl/TensorTransformationsKernels.h>

namespace at::native::xpu {

template <int N>
struct alignas(N) OpaqueType {
  char data[N];
};

template <typename func_t>
struct ElementwiseKernelFunctor {
  void operator()(sycl::nd_item<1> itemId) const {
    int idx = itemId.get_global_linear_id();

    for (int i = 0; i < loops; ++i) {
      if (idx < total_n_elems) {
        f(idx);
        idx += total_work_items;
      }
    }
  }
  ElementwiseKernelFunctor(
      int loops_,
      int total_n_elems_,
      func_t f_,
      int total_work_items_)
      : loops(loops_),
        total_n_elems(total_n_elems_),
        f(f_),
        total_work_items(total_work_items_) {}

 private:
  int loops;
  int total_n_elems;
  func_t f;
  int total_work_items;
};

template <typename func_t>
void elementwise_kernel(int total_n_elems, func_t f) {
  using KernelClass = ElementwiseKernelFunctor<func_t>;

  auto& queue = getCurrentSYCLQueue();
  int64_t max_wg_size = at::xpu::getKernelMaxWorkGroupSize<KernelClass>();
  const int64_t target_global_size = at::xpu::getDeviceMaxWorkItems();
  int work_group_size =
      total_n_elems > max_wg_size ? max_wg_size : total_n_elems;
  const int max_work_group_num = target_global_size / work_group_size;
  int total_group_num = (total_n_elems + work_group_size - 1) / work_group_size;
  int work_group_num = total_group_num < max_work_group_num
      ? total_group_num
      : max_work_group_num;
  // work item in each work group calculates loops' elements
  int loops = total_group_num / work_group_num + 1;

  int total_work_items = work_group_size * work_group_num;

  KernelClass kfn(loops, total_n_elems, f, total_work_items);

  sycl_kernel_submit(
      sycl::range<1>(total_work_items),
      sycl::range<1>(work_group_size),
      queue,
      kfn);
}

template <typename func_t>
static void launch_kernel(int total_n_elems, func_t f) {
  TORCH_INTERNAL_ASSERT(
      total_n_elems >= 0 &&
      total_n_elems <= std::numeric_limits<int32_t>::max());
  elementwise_kernel<func_t>(total_n_elems, f);
}

template <typename scalar_t, typename offset_calc_t>
struct FlipKernelImplLoopFunctor {
  void operator()(const int i) const {
    const auto offsets = offset_calc.get(i);
    // offsets can be negative here, but it's fine
    scalar_t* const RESTRICT out_data =
        reinterpret_cast<scalar_t*>(out_ptr + offsets[0]);
    const scalar_t* const RESTRICT in_data =
        reinterpret_cast<const scalar_t*>(in_ptr + offsets[1]);
    *out_data = *in_data;
  }

  FlipKernelImplLoopFunctor(
      char* const RESTRICT out_ptr,
      const char* const RESTRICT in_ptr,
      const offset_calc_t offset_calc)
      : out_ptr(out_ptr), in_ptr(in_ptr), offset_calc(offset_calc) {}

 private:
  char* const RESTRICT out_ptr;
  const char* const RESTRICT in_ptr;
  const offset_calc_t offset_calc;
};

template <typename scalar_t>
void flip_kernel_impl(TensorIterator& iter) {
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      flip_kernel_impl<scalar_t>(sub_iter);
    }
    return;
  }

  char* const RESTRICT out_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  const char* const RESTRICT in_ptr =
      reinterpret_cast<const char*>(iter.data_ptr(1));

  const auto offset_calc =
      make_offset_calculator<2, /*signed_strides=*/true>(iter);

  FlipKernelImplLoopFunctor<scalar_t, decltype(offset_calc)> loop(
      out_ptr, in_ptr, offset_calc);
  launch_kernel(iter.numel(), loop);
}

void flip_kernel(TensorIterator& iter, bool quantized) {
  if (quantized) {
    TORCH_CHECK(false, "XPU current does not flip for quantized tensor");
  }
  AT_DISPATCH_V2(
      iter.dtype(),
      "flip_xpu",
      AT_WRAP([&] {
        using dtype = OpaqueType<sizeof(scalar_t)>;
        flip_kernel_impl<dtype>(iter);
      }),
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
      AT_EXPAND(AT_FLOAT8_TYPES),
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES),
      kComplexHalf,
      kBComplex32,
      kHalf,
      kBool,
      kBFloat16);
}

namespace {

constexpr int kShiftedCopyMaxDims = 6;
constexpr int kShiftedCopyGroupSize = 256;

// Output dims in memory order (dim 0 innermost) after merging neighbours;
// sizes, strides and shifts are in units of U bytes for dim 0.
struct ShiftedCopyParams {
  int ndim = 0;
  uint32_t nvec = 0;
  at::detail::IntDivider<uint32_t> size[kShiftedCopyMaxDims];
  int32_t shift[kShiftedCopyMaxDims] = {};
  int32_t isize[kShiftedCopyMaxDims] = {}; // input sizes
  int32_t stride[kShiftedCopyMaxDims] = {}; // input strides
};

// out[c] = in[(c + shift) mod size] along every dim, or the fill value where
// c + shift lies outside the input (pad: shift = -left padding, which never
// wraps; roll: the input has the output's sizes). Each work-item writes V
// consecutive output units of U bytes as one aligned store and gathers them
// from the input, so sub-group stores cover whole cache lines and all dims
// are handled in a single pass. kWrap: a vector may run past the end of an
// output row (V does not divide the row length).
template <int U, int V, bool kWrap>
struct ShiftedCopyFunctor {
  using T = OpaqueType<U>;
  using VecT = memory::aligned_vector<T, V>;

  // Splits unit index e into outer coordinates c[1..]; returns the dim-0 one.
  // Loops have static bounds so that c stays in registers.
  uint32_t decompose(uint32_t e, uint32_t* c) const {
    const auto qr = p_.size[0].divmod(e);
    uint32_t r = qr.div;
#pragma unroll
    for (int d = 1; d < kShiftedCopyMaxDims; ++d) {
      if (d == p_.ndim - 1) {
        c[d] = r;
      } else if (d < p_.ndim) {
        const auto q = p_.size[d].divmod(r);
        c[d] = q.mod;
        r = q.div;
      }
    }
    return qr.mod;
  }

  // Moves c to the next dim-0 row, whose first unit is e.
  void next_row(uint32_t e, uint32_t* c) const {
    if (p_.ndim > 1 && c[1] + 1 < p_.size[1].divisor) {
      ++c[1];
    } else {
      decompose(e, c);
    }
  }

  int32_t src(int d, uint32_t x) const {
    const int32_t s = static_cast<int32_t>(x) + p_.shift[d];
    const int32_t n = p_.size[d].divisor;
    return s >= n ? s - n : s;
  }

  bool inside(int d, int32_t s) const {
    return static_cast<uint32_t>(s) < static_cast<uint32_t>(p_.isize[d]);
  }

  // Input offset of the outer coordinates; `valid`: they are inside it.
  int32_t row_offset(const uint32_t* c, bool& valid) const {
    int32_t off = 0;
    valid = true;
#pragma unroll
    for (int d = 1; d < kShiftedCopyMaxDims; ++d) {
      if (d < p_.ndim) {
        const int32_t s = src(d, c[d]);
        valid &= inside(d, s);
        off += s * p_.stride[d];
      }
    }
    return off;
  }

  void operator()(sycl::nd_item<1> item) const {
    const uint32_t vid = item.get_global_linear_id();
    if (vid >= p_.nvec) {
      return;
    }
    uint32_t c[kShiftedCopyMaxDims];
    uint32_t x = decompose(vid * V, c);
    bool valid;
    int32_t off = row_offset(c, valid);
    // U is the input's alignment, so with V > 1 the units are loaded one by
    // one; the store is still one aligned vector.
    VecT v;
#pragma unroll
    for (int i = 0; i < V; ++i) {
      if (kWrap && x == p_.size[0].divisor) {
        x = 0;
        next_row(vid * V + i, c);
        off = row_offset(c, valid);
      }
      const int32_t s = src(0, x);
      if (valid && inside(0, s)) {
        v[i] = in_[off + s * p_.stride[0]];
      } else {
        v[i] = fill_;
      }
      ++x;
    }
    reinterpret_cast<VecT*>(out_)[vid] = v;
  }

  ShiftedCopyFunctor(
      const void* in,
      void* out,
      const ShiftedCopyParams& p,
      const void* fill)
      : in_(static_cast<const T*>(in)), out_(static_cast<T*>(out)), p_(p) {
    std::memcpy(&fill_, fill, sizeof(T));
  }

 private:
  const T* in_;
  T* out_;
  ShiftedCopyParams p_;
  T fill_;
};

template <int U, int V, bool kWrap>
void launch_shifted_copy(
    const Tensor& input,
    Tensor& output,
    const ShiftedCopyParams& p,
    const void* fill) {
  ShiftedCopyFunctor<U, V, kWrap> kfn(
      input.const_data_ptr(), output.mutable_data_ptr(), p, fill);
  const int64_t groups = ceil_div<int64_t>(p.nvec, kShiftedCopyGroupSize);
  sycl_kernel_submit(
      sycl::range<1>(groups * kShiftedCopyGroupSize),
      sycl::range<1>(kShiftedCopyGroupSize),
      getCurrentSYCLQueue(),
      kfn);
}

// Launches ShiftedCopyFunctor for a dense `output`. Roll: `shift[d]` in
// [0, size(d)) and the input has the output's sizes. Pad: `shift[d]` in
// (-output size, 0]. `fill` holds the fill value repeated over 16 bytes.
// Returns false if unsupported.
bool shifted_copy(
    const Tensor& input,
    Tensor& output,
    IntArrayRef shift,
    const std::array<char, 16>& fill) {
  const int64_t ndim = output.dim();
  if (ndim == 0 || !output.is_non_overlapping_and_dense() ||
      !canUse32BitIndexMath(input) || !canUse32BitIndexMath(output)) {
    return false;
  }
  c10::SmallVector<int64_t, kShiftedCopyMaxDims> order(ndim);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    return output.stride(a) < output.stride(b);
  });

  // Walk dims innermost first. A dim is merged into the inner one when that
  // is neither shifted nor padded and the input is contiguous across both;
  // shifting the merged dim by shift * inner size is the same.
  int64_t size[kShiftedCopyMaxDims], isize[kShiftedCopyMaxDims],
      stride[kShiftedCopyMaxDims], sh[kShiftedCopyMaxDims];
  int nd = 0;
  for (const int64_t d : order) {
    const int64_t n = output.size(d);
    if (n == 1) {
      continue;
    }
    if (nd > 0 && sh[nd - 1] == 0 && isize[nd - 1] == size[nd - 1] &&
        input.stride(d) == stride[nd - 1] * size[nd - 1]) {
      sh[nd - 1] = shift[d] * size[nd - 1];
      isize[nd - 1] *= input.size(d);
      size[nd - 1] *= n;
      continue;
    }
    if (nd == kShiftedCopyMaxDims) {
      return false;
    }
    size[nd] = n;
    isize[nd] = input.size(d);
    stride[nd] = input.stride(d);
    sh[nd] = shift[d];
    ++nd;
  }
  if (nd == 0) {
    size[0] = isize[0] = stride[0] = 1;
    sh[0] = 0;
    nd = 1;
  }

  // Unit: the largest power of two (<= 16 bytes) dividing every byte offset
  // the kernel forms, so it moves U-byte units instead of elements. Needs a
  // contiguous inner dim.
  const int64_t es = output.element_size();
  const auto in_addr =
      static_cast<int64_t>(reinterpret_cast<uintptr_t>(input.const_data_ptr()));
  const auto out_addr = static_cast<int64_t>(
      reinterpret_cast<uintptr_t>(output.const_data_ptr()));
  int64_t u = es;
  if (stride[0] == 1) {
    int64_t bits =
        16 | in_addr | out_addr | size[0] * es | isize[0] * es | sh[0] * es;
    for (int d = 1; d < nd; ++d) {
      bits |= stride[d] * es;
    }
    u = bits & -bits;
    const int64_t k = u / es;
    size[0] /= k;
    isize[0] /= k;
    sh[0] /= k;
    for (int d = 1; d < nd; ++d) {
      stride[d] /= k;
    }
  }
  // Store width: one unit if it is at least 8 bytes, else 16 or 8 bytes that
  // end on output row boundaries, else 16 bytes running across rows.
  const int64_t total_bytes = output.numel() * es;
  int64_t vec_bytes = u;
  bool wrap = false;
  if (u < 8) {
    for (const int64_t b : {16, 8}) {
      if (size[0] * u % b == 0 && out_addr % b == 0) {
        vec_bytes = b;
        break;
      }
    }
    if (vec_bytes == u && total_bytes % 16 == 0 && out_addr % 16 == 0) {
      vec_bytes = 16;
      wrap = true;
    }
  }
  const int64_t v = vec_bytes / u;

  ShiftedCopyParams p;
  p.ndim = nd;
  p.nvec = static_cast<uint32_t>(total_bytes / vec_bytes);
  for (int d = 0; d < nd; ++d) {
    p.size[d] = at::detail::IntDivider<uint32_t>(size[d]);
    p.shift[d] = static_cast<int32_t>(sh[d]);
    p.isize[d] = static_cast<int32_t>(isize[d]);
    p.stride[d] = static_cast<int32_t>(stride[d]);
  }

#define SHIFTED_COPY_CASE(U, V)                                         \
  case U * 32 + V:                                                      \
    if constexpr (V > 1) {                                              \
      if (wrap) {                                                       \
        launch_shifted_copy<U, V, true>(input, output, p, fill.data()); \
        return true;                                                    \
      }                                                                 \
    }                                                                   \
    launch_shifted_copy<U, V, false>(input, output, p, fill.data());    \
    return true;
  switch (u * 32 + v) {
    SHIFTED_COPY_CASE(1, 1)
    SHIFTED_COPY_CASE(1, 8)
    SHIFTED_COPY_CASE(1, 16)
    SHIFTED_COPY_CASE(2, 1)
    SHIFTED_COPY_CASE(2, 4)
    SHIFTED_COPY_CASE(2, 8)
    SHIFTED_COPY_CASE(4, 1)
    SHIFTED_COPY_CASE(4, 2)
    SHIFTED_COPY_CASE(4, 4)
    SHIFTED_COPY_CASE(8, 1)
    SHIFTED_COPY_CASE(16, 1)
  }
#undef SHIFTED_COPY_CASE
  return false;
}

} // namespace

bool roll_kernel(
    const Tensor& input,
    Tensor& output,
    IntArrayRef shifts,
    IntArrayRef dims) {
  // Rolling dim d by `shift` reads input coordinate (i - shift) mod size.
  c10::SmallVector<int64_t, kShiftedCopyMaxDims> start(input.dim(), 0);
  for (const auto i : c10::irange(dims.size())) {
    const int64_t d = maybe_wrap_dim(dims[i], input.dim());
    const int64_t n = input.size(d);
    start[d] = ((start[d] - shifts[i]) % n + n) % n;
  }
  return shifted_copy(input, output, start, {});
}

bool constant_pad_nd_kernel(
    const Tensor& input,
    Tensor& output,
    IntArrayRef lo,
    const Scalar& value) {
  std::array<char, 16> fill{};
  AT_DISPATCH_V2(
      output.scalar_type(),
      "constant_pad_nd_xpu",
      AT_WRAP([&] {
        const scalar_t v = value.to<scalar_t>();
        for (size_t i = 0; i < fill.size(); i += sizeof(v)) {
          std::memcpy(fill.data() + i, &v, sizeof(v));
        }
      }),
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
      AT_EXPAND(AT_FLOAT8_TYPES),
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES),
      kComplexHalf,
      kHalf,
      kBool,
      kBFloat16);
  c10::SmallVector<int64_t, kShiftedCopyMaxDims> shift(lo.size());
  for (const auto d : c10::irange(lo.size())) {
    shift[d] = -lo[d];
  }
  return shifted_copy(input, output, shift, fill);
}

} // namespace at::native::xpu
