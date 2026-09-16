/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch_v2.h>
#include <ATen/ceil_div.h>
#include <ATen/core/Tensor.h>
#include <ATen/xpu/XPUContext.h>

#include <ATen/native/xpu/sycl/pstl/PSTLFunctions.h>
#include <comm/Memory.h>
#include <comm/SYCLHelpers.h>
#include <comm/TensorInfo.h>

#include <ATen/native/xpu/sycl/NonzeroKernel.h>
#include <ATen/xpu/EmptyTensor.h>

namespace at::native::xpu {

// 0/1 int64 mask: global_mask[i] = 1 iff data[i] != 0.
// For bool, use volatile int to prevent the compiler from eliminating the load.
template <typename scalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void is_nonzero_kernel_impl(
    const scalar_t* data_ptr,
    int64_t* global_mask_ptr) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  const auto item_id = item.get_global_linear_id();

  if constexpr (std::is_same_v<scalar_t, bool>) {
    volatile int in = static_cast<int>(data_ptr[item_id]);
    global_mask_ptr[item_id] = static_cast<int64_t>(in != 0);
  } else {
    global_mask_ptr[item_id] =
        static_cast<int64_t>(data_ptr[item_id] != scalar_t(0));
  }
}

// Work-group-level reduction: counts nonzeros in [data_, data_+N_).
// Each work-group writes its partial count to partial_sums_[group_id].
template <typename scalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void count_nonzeros_kernel_impl(
    const scalar_t* data,
    int64_t N,
    int64_t* partial_sums,
    int64_t wg_size) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  const auto local_id = item.get_local_linear_id();
  const auto global_id = item.get_global_linear_id();

  int64_t* local_buf = (int64_t*)syclexp::get_work_group_scratch_memory();

  if constexpr (std::is_same_v<scalar_t, bool>) {
    int64_t val = 0;
    if (global_id < static_cast<size_t>(N)) {
      volatile int in = static_cast<int>(data[global_id]);
      val = static_cast<int64_t>(in != 0);
    }
    local_buf[local_id] = val;
  } else {
    local_buf[local_id] = static_cast<int64_t>(
        global_id < static_cast<size_t>(N) && data[global_id] != scalar_t(0));
  }
  sycl::group_barrier(item.get_group());

  for (int64_t stride = wg_size / 2; stride > 0; stride >>= 1) {
    if (local_id < static_cast<size_t>(stride))
      local_buf[local_id] += local_buf[local_id + stride];
    sycl::group_barrier(item.get_group());
  }

  if (local_id == 0)
    partial_sums[item.get_group_linear_id()] = local_buf[0];
}

struct DivisorSizes {
  int64_t divisor[XPU_MAX_TENSORINFO_DIMS];
  int64_t sizes[XPU_MAX_TENSORINFO_DIMS];
};

// For each nonzero element, converts its flat index to per-dimension indices
// and writes them directly into the output buffer (layout: dim-major, i.e.
// out_ptr[d * num_nonzeros + slot]).
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void scatter_to_out_kernel_impl(
    const int64_t* global_mask_ptr,
    const int64_t* target_pos_ptr,
    int64_t* out_ptr,
    int64_t chunk_start,
    int64_t global_offset,
    int64_t num_nonzeros,
    int64_t num_dim,
    DivisorSizes divisor_sizes) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  const auto item_id = item.get_global_linear_id();
  if (global_mask_ptr[item_id] != 0) {
    // target_pos is the inclusive prefix sum of global_mask, so
    // target_pos[i]-1 is this element's rank among nonzeros in the chunk.
    // global_offset shifts it to the correct position in the full output.
    const int64_t slot = global_offset + target_pos_ptr[item_id] - 1;
    const int64_t flat_idx = chunk_start + static_cast<int64_t>(item_id);
    for (int64_t d = 0; d < num_dim; d++) {
      out_ptr[d * num_nonzeros + slot] =
          flat_idx / divisor_sizes.divisor[d] % divisor_sizes.sizes[d];
    }
  }
}

// Predicate for pstl::copy_if: returns true if self_begin_[x] != 0.
template <typename scalar_t>
struct CopyIfFunc {
  bool operator()(int64_t x) const {
    if constexpr (std::is_same_v<scalar_t, bool>) {
      volatile int in = static_cast<int>(self_begin_[x]);
      return in != 0;
    } else {
      return self_begin_[x] != scalar_t(0);
    }
  }
  CopyIfFunc(const scalar_t* self_begin) : self_begin_(self_begin) {}

 private:
  const scalar_t* self_begin_;
};

// Converts flat nonzero indices to per-dimension coordinates.
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void flatten_idx_to_real_idx_kernel_impl(
    int64_t N,
    const int64_t num_dim,
    const int64_t num_nonzeros,
    int64_t* out_begin,
    int64_t* idx_flat_begin,
    DivisorSizes divisor_sizes) {
  sycl::nd_item<1> item_id = syclext::this_work_item::get_nd_item<1>();
  auto global_id = item_id.get_global_linear_id();

  if (global_id < N) {
    auto* divisor = divisor_sizes.divisor;
    auto* sizes = divisor_sizes.sizes;
    auto dim = global_id / num_nonzeros;
    auto index = global_id % num_nonzeros;

    out_begin[global_id] = (idx_flat_begin[index] / divisor[dim]) % sizes[dim];
  }
}
template <typename scalar_t>
void nonzero_template(const Tensor& self_, Tensor& out) {
  Tensor self = self_.contiguous();

  const int64_t num_dim = self.dim();
  const int64_t N = self.numel();
  const scalar_t* self_data = self.const_data_ptr<scalar_t>();
  auto& queue = getCurrentSYCLQueue();
  auto long_options = out.options().dtype(at::kLong).memory_format(
      LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  // Threshold below which the single-pass copy_if approach is used.
  // For N <= scatter_chunk_size the O(N) scratch allocation is acceptable
  // and avoids the overhead of the two-pass chunked algorithm.
  const int64_t scatter_chunk_size = int64_t(1) << 25;

  if (N <= scatter_chunk_size) {
    // ---- Fast path: single-pass via pstl::copy_if ----
    Tensor idx_flat = at::empty({N}, long_options);
    int64_t* idx_flat_begin = idx_flat.data_ptr<int64_t>();
    int64_t* range_begin = nullptr;

    CopyIfFunc<scalar_t> f(self_data);
    auto idx_flat_end =
        pstl::copy_if<int64_t>(range_begin, range_begin + N, idx_flat_begin, f);
    auto num_nonzeros = std::distance(idx_flat_begin, idx_flat_end);

    bool need_to_copy = out.dim() == 2 && out.sizes()[0] == num_nonzeros &&
        out.sizes()[1] == num_dim && !out.t().is_contiguous();
    Tensor out_ = need_to_copy
        ? Tensor(at::detail::empty_xpu({num_dim, num_nonzeros}, out.options()))
        : out.resize_({num_dim, num_nonzeros});

    if (num_nonzeros > 0 && num_dim > 0) {
      int64_t* out_begin = out_.data_ptr<int64_t>();

      // preload sizes tensor for index calculation
      struct DivisorSizes divisor_sizes;
      divisor_sizes.sizes[num_dim - 1] = self.size(num_dim - 1);
      divisor_sizes.divisor[num_dim - 1] = 1;
      for (auto dim = num_dim - 2; dim >= 0; dim--) {
        divisor_sizes.sizes[dim] = self.size(dim);
        divisor_sizes.divisor[dim] =
            divisor_sizes.sizes[dim + 1] * divisor_sizes.divisor[dim + 1];
      }

      const int64_t total = num_nonzeros * num_dim;

      const auto wg_sz = std::min(
          syclMaxWorkGroupSize<flatten_idx_to_real_idx_kernel_impl>(), total);
      const auto num_wg = at::ceil_div(total, wg_sz);

      sycl_kernel_submit<flatten_idx_to_real_idx_kernel_impl>(
          wg_sz * num_wg,
          wg_sz,
          getCurrentSYCLQueue(),
          0,
          total,
          num_dim,
          num_nonzeros,
          out_begin,
          idx_flat_begin,
          divisor_sizes);
    }

    if (need_to_copy) {
      out.copy_(out_.t());
    } else {
      out.set_(out_.t());
    }
    return;
  }

  // ---- Memory-efficient path: chunked two-pass (for large tensors) ----
  const int64_t num_chunks = at::ceil_div(N, scatter_chunk_size);

  // ---- Pass 1: count nonzeros per chunk via work-group reduction ----
  const int64_t count_wg_size =
      syclMaxWorkGroupSize<count_nonzeros_kernel_impl<scalar_t>>();

  // Pre-allocate a single device buffer wide enough to hold every WG's partial
  // sum for every chunk. All count kernels are enqueued without blocking so
  // only one device to host transfer is needed at the end.
  const int64_t max_wgs_per_chunk =
      at::ceil_div(scatter_chunk_size, count_wg_size);
  Tensor all_partial_sums =
      at::empty({num_chunks * max_wgs_per_chunk}, long_options);
  int64_t* all_partial_sums_ptr = all_partial_sums.data_ptr<int64_t>();

  std::vector<int64_t> chunk_wgs(num_chunks);

  for (int64_t ci = 0; ci < num_chunks; ci++) {
    const int64_t start = ci * scatter_chunk_size;
    const int64_t this_chunk = std::min(scatter_chunk_size, N - start);
    const int64_t num_wgs = at::ceil_div(this_chunk, count_wg_size);
    chunk_wgs[ci] = num_wgs;

    sycl_kernel_submit<count_nonzeros_kernel_impl<scalar_t>>(
        num_wgs * count_wg_size,
        count_wg_size,
        queue,
        0,
        self_data + start,
        this_chunk,
        all_partial_sums_ptr + ci * max_wgs_per_chunk,
        count_wg_size);
  }

  // Single device to host sync: retrieve all partial sums at once.
  const int64_t total_partial_sums = num_chunks * max_wgs_per_chunk;
  std::vector<int64_t> psums(total_partial_sums);
  memcpyDeviceToHost(
      psums.data(),
      all_partial_sums_ptr,
      total_partial_sums * sizeof(int64_t),
      /* async */ false,
      /* hctx */ nullptr);

  std::vector<int64_t> chunk_counts(num_chunks);
  std::vector<int64_t> chunk_offsets(num_chunks);
  int64_t num_nonzeros = 0;

  for (int64_t ci = 0; ci < num_chunks; ci++) {
    chunk_offsets[ci] = num_nonzeros;
    const int64_t* row = psums.data() + ci * max_wgs_per_chunk;
    int64_t count = 0;
    for (int64_t wi = 0; wi < chunk_wgs[ci]; wi++)
      count += row[wi];
    chunk_counts[ci] = count;
    num_nonzeros += count;
  }

  // ---- Allocate output (dim-major layout: {num_dim, num_nonzeros}) ----
  bool need_to_copy = out.dim() == 2 && out.sizes()[0] == num_nonzeros &&
      out.sizes()[1] == num_dim && !out.t().is_contiguous();
  Tensor out_ = need_to_copy
      ? Tensor(at::detail::empty_xpu({num_dim, num_nonzeros}, out.options()))
      : out.resize_({num_dim, num_nonzeros});

  // Precompute per-dimension sizes and divisors for flatâ†’multi-dim
  // conversion.
  struct DivisorSizes divisor_sizes;
  if (num_dim > 0) {
    divisor_sizes.sizes[num_dim - 1] = self.size(num_dim - 1);
    divisor_sizes.divisor[num_dim - 1] = 1;
    for (auto d = num_dim - 2; d >= 0; d--) {
      divisor_sizes.sizes[d] = self.size(d);
      divisor_sizes.divisor[d] =
          divisor_sizes.sizes[d + 1] * divisor_sizes.divisor[d + 1];
    }
  }

  // ---- Pass 2: scatter per-dim indices into the output tensor ----
  // global_mask and target_pos are reusable scratch buffers capped at
  // actual_chunk_size so small tensors don't pay the full 256 MB each.
  const int64_t actual_chunk_size = std::min(scatter_chunk_size, N);
  Tensor global_mask = at::empty({actual_chunk_size}, long_options);
  Tensor target_pos = at::empty({actual_chunk_size}, long_options);
  int64_t* global_mask_ptr = global_mask.data_ptr<int64_t>();
  int64_t* target_pos_ptr = target_pos.data_ptr<int64_t>();
  int64_t* out_ptr =
      (num_nonzeros > 0 && num_dim > 0) ? out_.data_ptr<int64_t>() : nullptr;

  if (out_ptr != nullptr) {
    for (int64_t ci = 0; ci < num_chunks; ci++) {
      if (chunk_counts[ci] == 0)
        continue;

      const int64_t start = ci * scatter_chunk_size;
      const int64_t this_chunk = std::min(scatter_chunk_size, N - start);

      // Fill global_mask[0..this_chunk): 1 where element is nonzero, 0
      // elsewhere.
      const int64_t count_wg_size =
          syclMaxWorkGroupSize<is_nonzero_kernel_impl<scalar_t>>();
      const int64_t num_wgs = at::ceil_div(this_chunk, count_wg_size);

      sycl_kernel_submit<is_nonzero_kernel_impl<scalar_t>>(
          num_wgs * count_wg_size,
          count_wg_size,
          queue,
          0,
          self_data + start,
          global_mask_ptr);

      // Inclusive prefix sum of global_mask â†’ target_pos[i] = number of
      // nonzeros in [0..i] of this chunk. Used by
      // scatter_to_out_kernel_impl to compute each nonzero's output slot:
      // slot = global_offset + target_pos[i] - 1.
      pstl::inclusive_scan<int64_t>(
          global_mask_ptr,
          global_mask_ptr + this_chunk,
          target_pos_ptr,
          int64_t(0));

      const int64_t count_wg_size1 =
          syclMaxWorkGroupSize<scatter_to_out_kernel_impl>();
      const int64_t num_wgs1 = at::ceil_div(this_chunk, count_wg_size);

      sycl_kernel_submit<scatter_to_out_kernel_impl>(
          num_wgs1 * count_wg_size1,
          count_wg_size1,
          queue,
          0,
          global_mask_ptr,
          target_pos_ptr,
          out_ptr,
          start,
          chunk_offsets[ci],
          num_nonzeros,
          num_dim,
          divisor_sizes);
    }
  }

  if (need_to_copy) {
    out.copy_(out_.t());
  } else {
    out.set_(out_.t());
  }
}

void nonzero_kernel(const Tensor& self, Tensor& out) {
  AT_DISPATCH_V2(
      self.scalar_type(),
      "nonzero_xpu",
      AT_WRAP([&] { nonzero_template<scalar_t>(self, out); }),
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
      kComplexHalf,
      kBComplex32,
      kBool,
      kBFloat16,
      kHalf);
}
} // namespace at::native::xpu