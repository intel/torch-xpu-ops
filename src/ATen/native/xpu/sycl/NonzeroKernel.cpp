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
#include <ATen/ceil_div.h>
#include <ATen/core/Tensor.h>

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
struct IsNonzeroKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    if constexpr (std::is_same_v<scalar_t, bool>) {
      volatile int in = static_cast<int>(data_ptr_[item_id]);
      global_mask_ptr_[item_id] = static_cast<int64_t>(in != 0);
    } else {
      global_mask_ptr_[item_id] =
          static_cast<int64_t>(data_ptr_[item_id] != scalar_t(0));
    }
  }
  IsNonzeroKernelFunctor(const scalar_t* data_ptr, int64_t* global_mask_ptr)
      : data_ptr_(data_ptr), global_mask_ptr_(global_mask_ptr) {}

 private:
  const scalar_t* data_ptr_;
  int64_t* global_mask_ptr_;
};

// Work-group-level reduction: counts nonzeros in [data_, data_+N_).
// Each work-group writes its partial count to partial_sums_[group_id].
template <typename scalar_t>
struct CountNonzerosKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item) const {
    const auto local_id = item.get_local_linear_id();
    const auto global_id = item.get_global_linear_id();

    if constexpr (std::is_same_v<scalar_t, bool>) {
      int64_t val = 0;
      if (global_id < static_cast<size_t>(N_)) {
        volatile int in = static_cast<int>(data_[global_id]);
        val = static_cast<int64_t>(in != 0);
      }
      local_buf_[local_id] = val;
    } else {
      local_buf_[local_id] = static_cast<int64_t>(
          global_id < static_cast<size_t>(N_) &&
          data_[global_id] != scalar_t(0));
    }
    sycl::group_barrier(item.get_group());

    for (int64_t stride = wg_size_ / 2; stride > 0; stride >>= 1) {
      if (local_id < static_cast<size_t>(stride))
        local_buf_[local_id] += local_buf_[local_id + stride];
      sycl::group_barrier(item.get_group());
    }

    if (local_id == 0)
      partial_sums_[item.get_group_linear_id()] = local_buf_[0];
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    local_buf_ = sycl_local_acc_t<int64_t>(wg_size_, cgh);
  }

  CountNonzerosKernelFunctor(
      const scalar_t* data,
      int64_t N,
      int64_t* partial_sums,
      int64_t wg_size)
      : data_(data), N_(N), partial_sums_(partial_sums), wg_size_(wg_size) {}

 private:
  const scalar_t* data_;
  int64_t N_;
  int64_t* partial_sums_;
  int64_t wg_size_;
  sycl_local_acc_t<int64_t> local_buf_;
};

// For each nonzero element, converts its flat index to per-dimension indices
// and writes them directly into the output buffer (layout: dim-major, i.e.
// out_ptr[d * num_nonzeros + slot]).
struct ScatterToOutKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    if (global_mask_ptr_[item_id] != 0) {
      // target_pos is the inclusive prefix sum of global_mask, so
      // target_pos[i]-1 is this element's rank among nonzeros in the chunk.
      // global_offset shifts it to the correct position in the full output.
      const int64_t slot = global_offset_ + target_pos_ptr_[item_id] - 1;
      const int64_t flat_idx =
          chunk_start_ + static_cast<int64_t>(item_id.get_linear_id());
      for (int64_t d = 0; d < num_dim_; d++) {
        out_ptr_[d * num_nonzeros_ + slot] = flat_idx / divisor_[d] % sizes_[d];
      }
    }
  }
  ScatterToOutKernelFunctor(
      const int64_t* global_mask_ptr,
      const int64_t* target_pos_ptr,
      int64_t* out_ptr,
      int64_t chunk_start,
      int64_t global_offset,
      int64_t num_nonzeros,
      int64_t num_dim,
      int64_t* divisor,
      int64_t* sizes)
      : global_mask_ptr_(global_mask_ptr),
        target_pos_ptr_(target_pos_ptr),
        out_ptr_(out_ptr),
        chunk_start_(chunk_start),
        global_offset_(global_offset),
        num_nonzeros_(num_nonzeros),
        num_dim_(num_dim) {
    for (int64_t d = 0; d < num_dim; d++) {
      divisor_[d] = divisor[d];
      sizes_[d] = sizes[d];
    }
  }

 private:
  const int64_t* global_mask_ptr_;
  const int64_t* target_pos_ptr_;
  int64_t* out_ptr_;
  int64_t chunk_start_;
  int64_t global_offset_;
  int64_t num_nonzeros_;
  int64_t num_dim_;
  int64_t divisor_[XPU_MAX_TENSORINFO_DIMS];
  int64_t sizes_[XPU_MAX_TENSORINFO_DIMS];
};

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
struct FlattenIdxtoRealIdxKernelFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    auto global_id = item_id.get_global_linear_id();

    if (global_id < N_) {
      auto dim = global_id / num_nonzeros_;
      auto index = global_id % num_nonzeros_;
      out_begin_[global_id] =
          idx_flat_begin_[index] / divisor_[dim] % sizes_[dim];
    }
  }
  FlattenIdxtoRealIdxKernelFunctor(
      int64_t N,
      const int64_t num_dim,
      const int64_t num_nonzeros,
      int64_t* out_begin,
      int64_t* idx_flat_begin,
      int64_t* divisor,
      int64_t* sizes)
      : N_(N),
        num_dim_(num_dim),
        num_nonzeros_(num_nonzeros),
        out_begin_(out_begin),
        idx_flat_begin_(idx_flat_begin) {
    for (auto dim = num_dim - 1; dim >= 0; dim--) {
      sizes_[dim] = sizes[dim];
      divisor_[dim] = divisor[dim];
    }
  }

 private:
  int64_t N_;
  const int64_t num_dim_;
  const int64_t num_nonzeros_;
  int64_t* out_begin_;
  int64_t* idx_flat_begin_;
  int64_t divisor_[XPU_MAX_TENSORINFO_DIMS];
  int64_t sizes_[XPU_MAX_TENSORINFO_DIMS];
};

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

      int64_t sizes[XPU_MAX_TENSORINFO_DIMS];
      int64_t divisor[XPU_MAX_TENSORINFO_DIMS];
      sizes[num_dim - 1] = self.size(num_dim - 1);
      divisor[num_dim - 1] = 1;
      for (auto d = num_dim - 2; d >= 0; d--) {
        sizes[d] = self.size(d);
        divisor[d] = sizes[d + 1] * divisor[d + 1];
      }

      const int64_t total = num_nonzeros * num_dim;
      FlattenIdxtoRealIdxKernelFunctor kfn(
          total,
          num_dim,
          num_nonzeros,
          out_begin,
          idx_flat_begin,
          divisor,
          sizes);
      const auto wg_sz = std::min(syclMaxWorkGroupSize(kfn), total);
      const auto num_wg = at::ceil_div(total, wg_sz);
      sycl_kernel_submit(wg_sz * num_wg, wg_sz, queue, kfn);
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
  using CountFunctor = CountNonzerosKernelFunctor<scalar_t>;
  const auto count_wg_size = syclMaxWorkGroupSize<CountFunctor>();

  // Pre-allocate a single device buffer wide enough to hold every WG's partial
  // sum for every chunk. All count kernels are enqueued without blocking so
  // only one device→host transfer is needed at the end.
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

    CountFunctor count_kfn(
        self_data + start,
        this_chunk,
        all_partial_sums_ptr + ci * max_wgs_per_chunk,
        count_wg_size);
    sycl_kernel_submit(
        num_wgs * count_wg_size, count_wg_size, queue, count_kfn);
  }

  // Single device→host sync: retrieve all partial sums at once.
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

  // Precompute per-dimension sizes and divisors for flat→multi-dim conversion.
  int64_t sizes[XPU_MAX_TENSORINFO_DIMS];
  int64_t divisor[XPU_MAX_TENSORINFO_DIMS];
  if (num_dim > 0) {
    sizes[num_dim - 1] = self.size(num_dim - 1);
    divisor[num_dim - 1] = 1;
    for (auto d = num_dim - 2; d >= 0; d--) {
      sizes[d] = self.size(d);
      divisor[d] = sizes[d + 1] * divisor[d + 1];
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
      IsNonzeroKernelFunctor<scalar_t> mask_kfn(
          self_data + start, global_mask_ptr);
      sycl_kernel_submit(sycl::range<1>(this_chunk), queue, mask_kfn);

      // Inclusive prefix sum of global_mask → target_pos[i] = number of
      // nonzeros in [0..i] of this chunk. Used by ScatterToOutKernelFunctor to
      // compute each nonzero's output slot: slot = global_offset +
      // target_pos[i] - 1.
      pstl::inclusive_scan<int64_t>(
          global_mask_ptr,
          global_mask_ptr + this_chunk,
          target_pos_ptr,
          int64_t(0));

      ScatterToOutKernelFunctor scatter_kfn(
          global_mask_ptr,
          target_pos_ptr,
          out_ptr,
          start,
          chunk_offsets[ci],
          num_nonzeros,
          num_dim,
          divisor,
          sizes);
      sycl_kernel_submit(sycl::range<1>(this_chunk), queue, scatter_kfn);
    }
  }

  if (need_to_copy) {
    out.copy_(out_.t());
  } else {
    out.set_(out_.t());
  }
}

void nonzero_kernel(const Tensor& self, Tensor& out) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::ComplexHalf,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "nonzero_xpu",
      [&] { nonzero_template<scalar_t>(self, out); });
}
} // namespace at::native::xpu
