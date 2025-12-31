/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions of this file are derived from PyTorch
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning(disable : 4715)
#endif

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <comm/SYCLContext.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/aminmax.h>
#include <ATen/ops/linspace.h>
#endif

#include <ATen/native/xpu/sycl/HistogramKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct HistogramddKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item_id) const {
    auto input_data = input_;
    int64_t wi_id = item_id.get_local_linear_id();
    int64_t wg_id = item_id.get_group_linear_id();
    int64_t batch_idx = (batch_num_ == 1) ? 0 : wi_id / batch_wg_size_;
    int64_t batch_local_id = (batch_num_ == 1) ? wi_id : wi_id % batch_wg_size_;

    int64_t ele_idx = wg_id * batch_num_ + batch_idx;
    bool active = (ele_idx < input_size_);

    for (int64_t dim = 0; active && dim < input_dim_; ++dim) {
      auto elem = input_data[ele_idx][dim];
      const scalar_t* bin_edges = bin_edges_list_[dim];
      int64_t bin_edges_size = num_bin_edges_[dim];
      if (!(elem >= bin_edges[0] && elem <= bin_edges[bin_edges_size - 1])) {
        active = false;
        break;
      }
    }

    // initialize slm
    if (active && batch_local_id == 0) {
      slm_[batch_idx] = 0;
    }
    sycl::group_barrier(item_id.get_group());

    // loop if wg_size_ is smaller than total_bin_size_
    for (int s = 0; active && s < scan_size_; ++s) {
      // map each work item to its corresponding bin
      // (batch_local_id, s) |-> (dim, bin_idx)
      int64_t dim = 0;
      int64_t bin_idx = -1;

      int64_t target_bin_linear_idx = batch_local_id + s * batch_wg_size_;
      if (target_bin_linear_idx >= total_bin_size_) {
        active = false;
        break;
      }

      for (int64_t cnt = 0; dim < input_dim_; ++dim) {
        int64_t bin_size = num_bin_edges_[dim] - 1;
        if (target_bin_linear_idx - cnt < bin_size) {
          bin_idx = target_bin_linear_idx - cnt;
          break;
        }
        cnt += bin_size;
      }

      if (bin_idx == -1) {
        active = false;
        break;
      }

      auto elem = input_data[ele_idx][dim];
      const scalar_t* bin_edges = bin_edges_list_[dim];
      int64_t bin_size = num_bin_edges_[dim] - 1;

      bool match = false;
      if (bin_idx == bin_size - 1) {
        if (bin_edges[bin_idx] <= elem && elem <= bin_edges[bin_idx + 1]) {
          match = true;
        }
      } else {
        if (bin_edges[bin_idx] <= elem && elem < bin_edges[bin_idx + 1]) {
          match = true;
        }
      }
      if (match) {
        auto ptr =
            slm_.template get_multi_ptr<sycl::access::decorated::no>().get();
        atomicAdd(
            sycl_local_ptr<int64_t>(ptr + batch_idx),
            bin_idx * hist_strides_[dim]);
      }
    }
    sycl::group_barrier(item_id.get_group());

    if (active && batch_local_id == 0) {
      auto hist_idx = slm_[batch_idx];
      scalar_t value = (scalar_t)1;
      if (use_weight_) {
        auto weight_data = weight_;
        value = weight_data[ele_idx];
      }
      atomicAdd((sycl_global_ptr<scalar_t>)(hist_ + hist_idx), value);
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    // SLM is used for accumulating hist_idx
    slm_ = sycl_local_acc_t<int64_t, 1>(batch_num_, cgh);
  }

  HistogramddKernelFunctor(
      const PackedTensorAccessor64<const scalar_t, 2> input,
      const scalar_t* const* bin_edges_list,
      scalar_t* hist,
      const int64_t* hist_strides,
      const PackedTensorAccessor64<const scalar_t, 1> weight,
      bool use_weight,
      int64_t input_size,
      int64_t input_dim,
      const int64_t* num_bin_edges,
      int64_t total_bin_size,
      int64_t wg_size,
      int64_t batch_num,
      int64_t batch_wg_size,
      int64_t scan_size)
      : input_(input),
        bin_edges_list_(bin_edges_list),
        hist_(hist),
        hist_strides_(hist_strides),
        weight_(weight),
        use_weight_(use_weight),
        input_size_(input_size),
        input_dim_(input_dim),
        num_bin_edges_(num_bin_edges),
        total_bin_size_(total_bin_size),
        wg_size_(wg_size),
        batch_num_(batch_num),
        batch_wg_size_(batch_wg_size),
        scan_size_(scan_size) {}

 private:
  const PackedTensorAccessor64<const scalar_t, 2> input_;
  const scalar_t* const* bin_edges_list_;
  scalar_t* hist_;
  const int64_t* hist_strides_;
  const PackedTensorAccessor64<const scalar_t, 1> weight_;
  bool use_weight_;
  int64_t input_size_;
  int64_t input_dim_;
  const int64_t* num_bin_edges_;
  const int64_t total_bin_size_;
  const int64_t wg_size_;
  const int64_t batch_num_;
  const int64_t batch_wg_size_;
  const int64_t scan_size_;

  sycl_local_acc_t<int64_t, 1> slm_;
};

template <typename scalar_t>
void histogramdd_template(
    const PackedTensorAccessor64<const scalar_t, 2> input,
    const scalar_t* const* bin_edges_list,
    scalar_t* hist,
    const int64_t* hist_strides,
    const PackedTensorAccessor64<const scalar_t, 1> weight,
    bool use_weight,
    int64_t input_size,
    int64_t input_dim,
    const int64_t* num_bin_edges,
    const int64_t total_bin_size) {
  using Kernel = HistogramddKernelFunctor<scalar_t>;
  const int64_t max_wg_size = syclMaxWorkGroupSize<Kernel>();
  int64_t num_wg = input_size;
  int64_t batch_num = 1;
  int64_t batch_wg_size = max_wg_size;
  int64_t scan_size = (total_bin_size + batch_wg_size - 1) / batch_wg_size;
  int64_t work_group_size = max_wg_size;
  if (max_wg_size >= 2 * total_bin_size) {
    scan_size = 1;
    batch_wg_size = total_bin_size;
    batch_num = max_wg_size / batch_wg_size;
    work_group_size = batch_num * batch_wg_size;
    num_wg = (input_size + batch_num - 1) / batch_num;
  }
  Kernel kfn(
      input,
      bin_edges_list,
      hist,
      hist_strides,
      weight,
      use_weight,
      input_size,
      input_dim,
      num_bin_edges,
      total_bin_size,
      work_group_size,
      batch_num,
      batch_wg_size,
      scan_size);
  sycl_kernel_submit(
      num_wg * work_group_size, work_group_size, getCurrentSYCLQueue(), kfn);
}

template <typename scalar_t>
struct HistogramddLinearKernelFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    auto input_data = input_;
    int64_t wi_id = item_id.get_global_id();
    if (wi_id >= input_size_) {
      return;
    }
    int64_t ele_idx = wi_id;
    int64_t hist_idx = 0;
    for (int dim = 0; dim < input_dim_; ++dim) {
      auto i_value = input_data[ele_idx][dim];
      const scalar_t* bin_edges = bin_edges_list_[dim];
      auto bin_size = num_bin_edges_[dim] - 1;
      auto leftmost_edge = bin_edges[0];
      auto rightmost_edge = bin_edges[bin_size];
      if (!(i_value >= leftmost_edge && i_value <= rightmost_edge)) {
        return;
      }
      int64_t bin_idx =
          (int64_t)(((i_value - leftmost_edge)) * bin_size / (rightmost_edge - leftmost_edge));
      if (bin_idx == bin_size) {
        bin_idx -= 1;
      }
      hist_idx += bin_idx * hist_strides_[dim];
    }

    scalar_t value = (scalar_t)1;
    if (use_weight_) {
      auto weight_data = weight_;
      value = weight_data[wi_id];
    }
    atomicAdd((sycl_global_ptr<scalar_t>)(hist_ + hist_idx), value);
  }

  HistogramddLinearKernelFunctor(
      const PackedTensorAccessor64<const scalar_t, 2> input,
      const scalar_t* const* bin_edges_list,
      scalar_t* hist,
      const int64_t* hist_strides,
      const PackedTensorAccessor64<const scalar_t, 1> weight,
      bool use_weight,
      int64_t input_size,
      int64_t input_dim,
      const int64_t* num_bin_edges)
      : input_(input),
        bin_edges_list_(bin_edges_list),
        hist_(hist),
        hist_strides_(hist_strides),
        weight_(weight),
        use_weight_(use_weight),
        input_size_(input_size),
        input_dim_(input_dim),
        num_bin_edges_(num_bin_edges) {}

 private:
  const PackedTensorAccessor64<const scalar_t, 2> input_;
  const scalar_t* const* bin_edges_list_;
  scalar_t* hist_;
  const int64_t* hist_strides_;
  const PackedTensorAccessor64<const scalar_t, 1> weight_;
  bool use_weight_;
  int64_t input_size_;
  int64_t input_dim_;
  const int64_t* num_bin_edges_;
};

template <typename scalar_t>
void histogramdd_linear_template(
    const PackedTensorAccessor64<const scalar_t, 2> input,
    const scalar_t* const* bin_edges_list,
    scalar_t* hist,
    const int64_t* hist_strides,
    const PackedTensorAccessor64<const scalar_t, 1> weight,
    bool use_weight,
    int64_t input_size,
    int64_t input_dim,
    const int64_t* num_bin_edges) {
  HistogramddLinearKernelFunctor<scalar_t> kfn(
      input,
      bin_edges_list,
      hist,
      hist_strides,
      weight,
      use_weight,
      input_size,
      input_dim,
      num_bin_edges);
  const int64_t work_group_size = syclMaxWorkGroupSize(kfn);
  const int64_t num_wg = (input_size + work_group_size - 1) / work_group_size;
  sycl_kernel_submit(
      num_wg * work_group_size, work_group_size, getCurrentSYCLQueue(), kfn);
}

/* The main algorithm. Expects that the input tensor has shape (N, D).
 * Expects that bin_edges contains D one-dimensional tensors, each specifying
 * an increasing sequences of bin edges.
 *
 * Interprets the input as N different D-dimensional coordinates and maps them
 * into the D-dimensional bins defined by bin_edges, accumulating a
 * D-dimensional histogram in the hist tensor.
 *
 * Accepts a template argument of type BIN_SELECTION_ALGORITHM specifying how
 * the scalars in each dimension should be mapped into the dimension's bins:
 *
 *     - LINEAR_INTERPOLATION: each bin edge sequence must form a linear
 * progression. Scalars are mapped to bins by computing (element -
 * leftmost_edge)/(rightmost_edge - leftmost_edge) * bin_ct and truncating the
 * result to an integer.
 *
 *       This is the fastest option, but its results may not be perfectly
 * consistent with the boundaries specified in bin_edges due to precision
 * issues.
 *
 *       Used by torch.histc, which doesn't need consistency with bin_edges as
 * it does not return bin_edges. Additionally, this implementation is identical
 * to the legacy histc implementation, which was replaced when histogram was
 * implemented.
 *
 *     - LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH: Also expects that each bin edge
 * sequence forms a linear progression. For each scalar, if 'pos' is the bin
 * selected by the LINEAR_INTERPOLATION approach, this approach inspects the
 * boundaries in bin_edges to place the scalar into pos - 1, pos, or pos + 1.
 * The "local search" over neighboring bins allows for correction of
 * misclassifications due to precision issues (a scalar very close to a bin_edge
 * may be misclassified by LINEAR_INTERPOLATION).
 *
 *       Should produce the same output as the general case BINARY_SEARCH, but
 * run about 3x faster asymptotically.
 *
 *       Used by torch.histogram for cases in which bin_edges is constructed
 * using torch.linspace. The behavior of LINEAR_INTERPOLATION may not perfectly
 * align with linspace bin_edges due to precision issues. torch.histogram
 * returns both the hist and bin_edges tensors as output, so the "local search"
 * is needed to keep its output internally consistent.
 *
 *     - PARALLEL_SEARCH: Handles torch.histogram's general case by searching
 * over the elements of bin_edges.
 */
enum BIN_SELECTION_ALGORITHM {
  LINEAR_INTERPOLATION,
  LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH,
  PARALLEL_SEARCH,
};
template <typename input_t, BIN_SELECTION_ALGORITHM algorithm>
void histogramdd_xpu_contiguous(
    Tensor& hist,
    const TensorList& bin_edges,
    const Tensor& self,
    const std::optional<Tensor>& weight) {
  TORCH_INTERNAL_ASSERT(self.dim() == 2);

  const int64_t N = self.size(0);
  if (weight.has_value()) {
    TORCH_INTERNAL_ASSERT(
        weight.value().dim() == 1 && weight.value().numel() == N);
  }

  const int64_t D = self.size(1);
  TORCH_INTERNAL_ASSERT(int64_t(bin_edges.size()) == D);
  for (const auto dim : c10::irange(D)) {
    TORCH_INTERNAL_ASSERT(bin_edges[dim].is_contiguous());
    TORCH_INTERNAL_ASSERT(hist.size(dim) + 1 == bin_edges[dim].numel());
  }

  if (D == 0) {
    // hist is an empty tensor in this case; nothing to do here
    return;
  }

  const auto accessor_in = self.packed_accessor64<const input_t, 2>();
  const auto accessor_wt = weight.has_value()
      ? weight.value().packed_accessor64<const input_t, 1>()
      : PackedTensorAccessor64<const input_t, 1>(
            nullptr, self.sizes().data(), self.strides().data());

  std::vector<int64_t> bin_seq(D);
  std::vector<int64_t> num_bin_edges(D);

  int64_t total_bin_size = 1;
  for (const auto dim : c10::irange(D)) {
    const input_t* data_ptr = bin_edges[dim].const_data_ptr<input_t>();
    bin_seq[dim] = reinterpret_cast<int64_t>(data_ptr);
    num_bin_edges[dim] = bin_edges[dim].numel();

    total_bin_size += num_bin_edges[dim] - 1;
  }

  Tensor hist_strides_xpu = at::tensor(
      hist.strides(),
      self.options()
          .dtype(c10::kLong)
          .memory_format(at::MemoryFormat::Contiguous));
  Tensor bin_edges_contig_ptr_xpu =
      at::tensor(bin_seq, hist_strides_xpu.options());
  Tensor num_bin_edges_xpu =
      at::tensor(num_bin_edges, hist_strides_xpu.options());

  if (algorithm == PARALLEL_SEARCH) {
    histogramdd_template<input_t>(
        accessor_in,
        reinterpret_cast<const input_t* const*>(
            bin_edges_contig_ptr_xpu.const_data_ptr()),
        hist.data_ptr<input_t>(),
        hist_strides_xpu.const_data_ptr<int64_t>(),
        accessor_wt,
        weight.has_value(),
        N,
        D,
        num_bin_edges_xpu.const_data_ptr<int64_t>(),
        total_bin_size);
  } else if (
      algorithm == LINEAR_INTERPOLATION ||
      algorithm == LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH) {
    histogramdd_linear_template<input_t>(
        accessor_in,
        reinterpret_cast<const input_t* const*>(
            bin_edges_contig_ptr_xpu.const_data_ptr()),
        hist.data_ptr<input_t>(),
        hist_strides_xpu.const_data_ptr<int64_t>(),
        accessor_wt,
        weight.has_value(),
        N,
        D,
        num_bin_edges_xpu.const_data_ptr<int64_t>());
  }
}

template <BIN_SELECTION_ALGORITHM bin_algorithm>
void histogramdd_xpu_kernel_template(
    const Tensor& self,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    const TensorList& bin_edges) {
  globalContext().alertNotDeterministic("histogramdd_xpu_kernel_template");
  hist.fill_(0);

  const int64_t N = self.size(-1);
  const int64_t M = std::accumulate(
      self.sizes().begin(),
      self.sizes().end() - 1,
      (int64_t)1,
      std::multiplies<int64_t>());

  const Tensor reshaped_self = self.reshape({M, N});

  const auto reshaped_weight = weight.has_value()
      ? std::optional<Tensor>(weight.value().reshape({M}))
      : std::optional<Tensor>();

  std::vector<Tensor> bin_edges_contig(bin_edges.size());
  for (const auto dim : c10::irange(bin_edges_contig.size())) {
    bin_edges_contig[dim] = bin_edges[dim].contiguous();
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, reshaped_self.scalar_type(), "histogram_xpu", [&]() {
        histogramdd_xpu_contiguous<scalar_t, bin_algorithm>(
            hist, bin_edges_contig, reshaped_self, reshaped_weight);
      });

  /* Divides each bin's value by the total count/weight in all bins,
   * and by the bin's volume.
   */
  if (density) {
    const auto hist_sum = hist.sum().item();
    hist.div_(hist_sum);

    /* For each dimension, divides each bin's value
     * by the bin's length in that dimension.
     */
    for (const auto dim : c10::irange(N)) {
      const auto bin_lengths = bin_edges[dim].diff();

      // Used to reshape bin_lengths to align with the corresponding dimension
      // of hist.
      std::vector<int64_t> shape(N, 1);
      shape[dim] = bin_lengths.numel();

      hist.div_(bin_lengths.reshape(shape));
    }
  }
}

void histogramdd_kernel(
    const Tensor& self,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    const TensorList& bin_edges) {
  histogramdd_xpu_kernel_template<PARALLEL_SEARCH>(
      self, weight, density, hist, bin_edges);
}

void histogramdd_linear_kernel(
    const Tensor& self,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    const TensorList& bin_edges,
    bool local_search) {
  if (local_search) {
    histogramdd_xpu_kernel_template<LINEAR_INTERPOLATION>(
        self, weight, density, hist, bin_edges);
  } else {
    histogramdd_xpu_kernel_template<LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH>(
        self, weight, density, hist, bin_edges);
  }
}

void histogram_select_outer_bin_edges_kernel(
    const Tensor& input,
    const int64_t N,
    std::vector<double>& leftmost_edges,
    std::vector<double>& rightmost_edges) {
  auto [min, max] = at::aminmax(input, 0);

  for (const auto i : c10::irange(N)) {
    leftmost_edges[i] = min[i].item().to<double>();
    rightmost_edges[i] = max[i].item().to<double>();
  }
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
#ifdef _MSC_VER
  #pragma warning(pop)
#endif

