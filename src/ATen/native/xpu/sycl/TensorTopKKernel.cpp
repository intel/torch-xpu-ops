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
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <ATen/native/xpu/sycl/Sorting.h>
#include <ATen/native/xpu/sycl/SortingKernels.h>
#include <ATen/native/xpu/sycl/SortingRadixSelect.h>

#include <ATen/native/xpu/sycl/TensorTopKKernel.h>

namespace at {
namespace native {
namespace xpu {

void topk_out_with_sort(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    const Tensor& values,
    const Tensor& indices) {
  Tensor sorted_values, sorted_indices;
  std::tie(sorted_values, sorted_indices) =
      at::sort(self, /* stable= */ false, dim, largest);
  values.copy_(sorted_values.narrow(dim, 0, k));
  indices.copy_(sorted_indices.narrow(dim, 0, k));
}

// ============================================================
// Single-block topk (sbtopk) with vectorized radix selection
// and deterministic prefix-scan gather
//
// Architecture aligned with CUDA's sbtopk (TensorTopK.cu):
//   Phase 1: Radix selection with RADIX_BITS=8, VEC=4 vectorized loads
//   Phase 2: Two-pass gather using exclusive prefix scan
//            (mirrors CUDA's exclusiveBinaryPrefixScan from ScanUtils.cuh)
//
// The prefix scan uses:
//   - Intra-sub-group: sycl::exclusive_scan_over_group (maps to CUDA warp scan)
//   - Cross-sub-group: SLM carries + single-thread serial prefix sum
//     (maps to CUDA's cross-warp carry pattern)
//
// Deterministic: write positions determined by prefix scan, no atomics in gather
// ============================================================

constexpr int SBTOPK_BLOCK = 256;
constexpr int SBTOPK_R_BITS = 8;
constexpr int SBTOPK_R_SIZE = 1 << SBTOPK_R_BITS;  // 256
constexpr int SBTOPK_R_MASK = SBTOPK_R_SIZE - 1;
constexpr int SBTOPK_MAX_K = 16;

// SLM layout:
// Phase 1 (radix select): [0..255] histogram, [256] selectedDigit, [257] kToFind
// Phase 2 (gather): [0..num_subgroups-1] sub-group carries, [num_subgroups] total
constexpr int SBTOPK_SLM_SIZE = SBTOPK_R_SIZE + 2;  // 258

template <typename scalar_t, bool Largest>
struct SbtopkFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  using RadixT = typename TopKTypeConfig<scalar_t>::RadixType;
  static constexpr int VEC = 4;
  using vec_t = at::native::memory::aligned_vector<scalar_t, VEC>;
  static constexpr int NUM_BITS = sizeof(scalar_t) * 8;
  // Mask for 16-bit types whose convert() returns uint32_t with upper bits set
  static constexpr RadixT CONVERT_MASK = (NUM_BITS == sizeof(RadixT) * 8)
      ? ~static_cast<RadixT>(0)
      : (static_cast<RadixT>(1) << NUM_BITS) - 1;

  inline RadixT convert_masked(scalar_t v) const {
    return TopKTypeConfig<scalar_t>::convert(v) & CONVERT_MASK;
  }

  void operator()(sycl::nd_item<1> item) const {
    int slice_idx = item.get_group(0);
    int lid = item.get_local_id(0);
    int block_size = item.get_local_range(0);
    auto sg = item.get_sub_group();
    int sg_size = sg.get_local_range()[0];
    int sg_lid = sg.get_local_linear_id();
    int sg_id = sg.get_group_linear_id();
    int num_sgs = block_size / sg_size;

    const scalar_t* slice_data = input_ + (int64_t)slice_idx * slice_size_;
    scalar_t* out_vals = values_ + (int64_t)slice_idx * k_;
    int64_t* out_idxs = indices_ + (int64_t)slice_idx * k_;

    auto smem_ptr =
        smem_.template get_multi_ptr<sycl::access::decorated::no>().get();

    RadixT desired = 0;
    RadixT desiredMask = 0;
    int kToFind = k_;

    int64_t numVecs = slice_size_ / VEC;
    int64_t tailStart = numVecs * VEC;
    const vec_t* data_vec = reinterpret_cast<const vec_t*>(slice_data);

    // ======== Phase 1: Radix Selection (find kth value) ========
    for (int digitPos = NUM_BITS - SBTOPK_R_BITS; digitPos >= 0;
         digitPos -= SBTOPK_R_BITS) {
      // Clear histogram
      for (int i = lid; i < SBTOPK_R_SIZE; i += block_size) {
        smem_ptr[i] = 0;
      }
      sycl::group_barrier(item.get_group());

      // Count digits using vectorized loads
      for (int64_t vi = lid; vi < numVecs; vi += block_size) {
        vec_t v = data_vec[vi];
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
          RadixT val = convert_masked(v.val[j]);
          if ((val & desiredMask) == desired) {
            int digit = (val >> digitPos) & SBTOPK_R_MASK;
            sycl::atomic_ref<
                int,
                sycl::memory_order::relaxed,
                sycl::memory_scope::work_group,
                sycl::access::address_space::local_space>
                ref(smem_ptr[digit]);
            ref.fetch_add(1);
          }
        }
      }
      // Tail elements
      for (int64_t i = tailStart + lid; i < slice_size_; i += block_size) {
        RadixT val = convert_masked(slice_data[i]);
        if ((val & desiredMask) == desired) {
          int digit = (val >> digitPos) & SBTOPK_R_MASK;
          sycl::atomic_ref<
              int,
              sycl::memory_order::relaxed,
              sycl::memory_scope::work_group,
              sycl::access::address_space::local_space>
              ref(smem_ptr[digit]);
          ref.fetch_add(1);
        }
      }
      sycl::group_barrier(item.get_group());

      // Find the digit bucket containing the kth element
      if (lid == 0) {
        int cumCount = 0;
        if (Largest) {
          for (int i = SBTOPK_R_SIZE - 1; i >= 0; --i) {
            int count = smem_ptr[i];
            cumCount += count;
            if (cumCount >= kToFind) {
              smem_ptr[SBTOPK_R_SIZE] = i;
              smem_ptr[SBTOPK_R_SIZE + 1] = kToFind - (cumCount - count);
              break;
            }
          }
        } else {
          for (int i = 0; i < SBTOPK_R_SIZE; ++i) {
            int count = smem_ptr[i];
            cumCount += count;
            if (cumCount >= kToFind) {
              smem_ptr[SBTOPK_R_SIZE] = i;
              smem_ptr[SBTOPK_R_SIZE + 1] = kToFind - (cumCount - count);
              break;
            }
          }
        }
      }
      sycl::group_barrier(item.get_group());

      int selectedDigit = smem_ptr[SBTOPK_R_SIZE];
      kToFind = smem_ptr[SBTOPK_R_SIZE + 1];

      desired =
          (desired & ~(static_cast<RadixT>(SBTOPK_R_MASK) << digitPos)) |
          (static_cast<RadixT>(selectedDigit) << digitPos);
      desiredMask |= (static_cast<RadixT>(SBTOPK_R_MASK) << digitPos);
      sycl::group_barrier(item.get_group());
    }

    RadixT kth_radix = desired;

    // ======== Phase 2: Gather using prefix scan ========
    // Two-pass gather (mirrors CUDA's gatherTopK):
    //   Pass 1: collect elements strictly better than kth
    //   Pass 2: fill remaining with elements equal to kth
    //
    // Each iteration loads VEC=4 elements per thread, counts matches, runs
    // exclusive prefix scan on the counts, and writes results at computed
    // positions. The prefix scan structure mirrors CUDA's ScanUtils.cuh:
    //   1. Intra-sub-group: exclusive_scan_over_group
    //   2. Cross-sub-group: SLM carries + thread-0 serial prefix sum

    int64_t numVecIters = (numVecs + block_size - 1) / block_size;
    int writeIndexStart = 0;

    // --- Pass 1: elements strictly better than kth ---
    for (int64_t vi_base = 0; vi_base < numVecIters; vi_base++) {
      int64_t vi = vi_base * block_size + lid;
      bool vecInRange = (vi < numVecs);

      int count = 0;
      scalar_t vals[VEC];
      int64_t idxs[VEC];
      if (vecInRange) {
        vec_t v = data_vec[vi];
        int64_t base_idx = vi * VEC;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
          vals[j] = v.val[j];
          idxs[j] = base_idx + j;
          RadixT rv = convert_masked(v.val[j]);
          bool take = Largest ? (rv > kth_radix) : (rv < kth_radix);
          if (take)
            count++;
        }
      }

      // Exclusive prefix scan (2 work-group barriers per iteration)
      int sg_exclusive =
          sycl::exclusive_scan_over_group(sg, count, 0, sycl::plus<int>());
      int sg_total = sycl::reduce_over_group(sg, count, sycl::plus<int>());

      if (sg_lid == sg_size - 1) {
        smem_ptr[sg_id] = sg_total;
      }
      sycl::group_barrier(item.get_group());

      int carry;
      if (lid == 0) {
        int current = 0;
        for (int s = 0; s < num_sgs; ++s) {
          int v = smem_ptr[s];
          smem_ptr[s] = current;
          current += v;
        }
        smem_ptr[num_sgs] = current;
      }
      sycl::group_barrier(item.get_group());

      int sg_prefix = smem_ptr[sg_id];
      carry = smem_ptr[num_sgs];
      int exclusive_idx = sg_exclusive + sg_prefix;

      if (vecInRange) {
        int writePos = writeIndexStart + exclusive_idx;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
          RadixT rv = convert_masked(vals[j]);
          bool take = Largest ? (rv > kth_radix) : (rv < kth_radix);
          if (take) {
            if (writePos < k_) {
              out_vals[writePos] = vals[j];
              out_idxs[writePos] = idxs[j];
            }
            writePos++;
          }
        }
      }

      writeIndexStart += carry;
    }

    // Handle tail elements for pass 1
    {
      bool hasTail = (lid == 0) && (tailStart < slice_size_);
      int tailCount = 0;
      scalar_t tailVals[VEC];
      int64_t tailIdxs[VEC];
      if (hasTail) {
        for (int64_t i = tailStart; i < slice_size_; i++) {
          RadixT rv = convert_masked(slice_data[i]);
          bool take = Largest ? (rv > kth_radix) : (rv < kth_radix);
          if (take) {
            tailVals[tailCount] = slice_data[i];
            tailIdxs[tailCount] = i;
            tailCount++;
          }
        }
        for (int t = 0; t < tailCount; t++) {
          int writePos = writeIndexStart + t;
          if (writePos < k_) {
            out_vals[writePos] = tailVals[t];
            out_idxs[writePos] = tailIdxs[t];
          }
        }
        writeIndexStart += tailCount;
      }
      // Broadcast writeIndexStart from thread 0
      sycl::group_barrier(item.get_group());
      if (lid == 0)
        smem_ptr[0] = writeIndexStart;
      sycl::group_barrier(item.get_group());
      writeIndexStart = smem_ptr[0];
    }

    // --- Pass 2: elements equal to kth (fill remaining slots) ---
    int topKRemaining = k_ - writeIndexStart;
    sycl::group_barrier(item.get_group());

    for (int64_t vi_base = 0; vi_base < numVecIters; vi_base++) {
      int64_t vi = vi_base * block_size + lid;
      bool vecInRange = (vi < numVecs);

      int count = 0;
      scalar_t vals[VEC];
      int64_t idxs[VEC];
      if (vecInRange) {
        vec_t v = data_vec[vi];
        int64_t base_idx = vi * VEC;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
          vals[j] = v.val[j];
          idxs[j] = base_idx + j;
          RadixT rv = convert_masked(v.val[j]);
          if (rv == kth_radix)
            count++;
        }
      }

      // Exclusive prefix scan (same as pass 1)
      int sg_exclusive =
          sycl::exclusive_scan_over_group(sg, count, 0, sycl::plus<int>());
      int sg_total = sycl::reduce_over_group(sg, count, sycl::plus<int>());

      if (sg_lid == sg_size - 1) {
        smem_ptr[sg_id] = sg_total;
      }
      sycl::group_barrier(item.get_group());

      int carry;
      if (lid == 0) {
        int current = 0;
        for (int s = 0; s < num_sgs; ++s) {
          int v = smem_ptr[s];
          smem_ptr[s] = current;
          current += v;
        }
        smem_ptr[num_sgs] = current;
      }
      sycl::group_barrier(item.get_group());

      int sg_prefix = smem_ptr[sg_id];
      carry = smem_ptr[num_sgs];
      int exclusive_idx = sg_exclusive + sg_prefix;

      if (vecInRange) {
        int writePos = writeIndexStart + exclusive_idx;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
          RadixT rv = convert_masked(vals[j]);
          if (rv == kth_radix) {
            if (exclusive_idx < topKRemaining && writePos < k_) {
              out_vals[writePos] = vals[j];
              out_idxs[writePos] = idxs[j];
            }
            writePos++;
            exclusive_idx++;
          }
        }
      }

      // Early exit when we've found enough
      if (carry >= topKRemaining) {
        break;
      }

      topKRemaining -= carry;
      writeIndexStart += carry;
    }

    // Handle tail for pass 2
    if (tailStart < slice_size_ && topKRemaining > 0) {
      if (lid == 0) {
        for (int64_t i = tailStart; i < slice_size_ && topKRemaining > 0;
             i++) {
          RadixT rv = convert_masked(slice_data[i]);
          if (rv == kth_radix) {
            if (writeIndexStart < k_) {
              out_vals[writeIndexStart] = slice_data[i];
              out_idxs[writeIndexStart] = i;
            }
            writeIndexStart++;
            topKRemaining--;
          }
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    smem_ = sycl_local_acc_t<int>(SBTOPK_SLM_SIZE, cgh);
  }

  SbtopkFunctor(
      const scalar_t* input,
      scalar_t* values,
      int64_t* indices,
      int64_t slice_size,
      int64_t k)
      : input_(input),
        values_(values),
        indices_(indices),
        slice_size_(slice_size),
        k_(k) {}

 private:
  const scalar_t* input_;
  scalar_t* values_;
  int64_t* indices_;
  int64_t slice_size_;
  int64_t k_;
  sycl_local_acc_t<int> smem_;
};

template <typename scalar_t>
void sbtopk_launch(
    const scalar_t* input,
    scalar_t* values,
    int64_t* indices,
    int64_t nsegments,
    int64_t nelements,
    int64_t k,
    bool largest) {
  auto& queue = at::xpu::getCurrentSYCLQueue();

  if (largest) {
    auto f = SbtopkFunctor<scalar_t, true>(
        input, values, indices, nelements, k);
    sycl_kernel_submit(nsegments * SBTOPK_BLOCK, SBTOPK_BLOCK, queue, f);
  } else {
    auto f = SbtopkFunctor<scalar_t, false>(
        input, values, indices, nelements, k);
    sycl_kernel_submit(nsegments * SBTOPK_BLOCK, SBTOPK_BLOCK, queue, f);
  }
}

void topk_kernel(
    const at::Tensor& input,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    const at::Tensor& values,
    const at::Tensor& indices) {
  if (k == 0) {
    return;
  }

  TORCH_CHECK(
      input.defined() && values.defined() && indices.defined(),
      "invalid inputs");

  auto self = (input.dim() == 0) ? input.view(1) : input;

  int64_t numel = self.numel();
  int64_t ndim = self.dim();
  dim = maybe_wrap_dim(dim, ndim);
  int64_t nelements = self.sizes()[dim];
  int64_t nsegments = numel / nelements;

  TORCH_CHECK(
      nelements <= std::numeric_limits<int>::max(),
      "The dimension being select can not have more than INT_MAX elements.");

  const auto self_dtype = self.dtype();
  TORCH_CHECK(
      self_dtype != ScalarType::ComplexFloat &&
          self_dtype != ScalarType::ComplexDouble,
      "Topk currently does not support complex dtypes on XPU.");

  auto out_sizes = self.sizes().vec();
  out_sizes[dim] = k;
  values.resize_(out_sizes);
  indices.resize_(out_sizes);

  if (k > 256) { // The segmented_group_select_pairs supports k<=256
    topk_out_with_sort(self.contiguous(), k, dim, largest, values, indices);
    return;
  }

  // Determine whether to use sbtopk (single-block radix select).
  // sbtopk launches one work-group (SBTOPK_BLOCK threads) per slice,
  // so it requires enough slices to fill the GPU's hardware thread slots.
  //
  // total_hw_threads = syclGpuEuCount() * syclGpuHWThreadsPerEU()
  //   (= number of sub-groups the GPU can run concurrently)
  // subgroups_per_wg = SBTOPK_BLOCK / subgroup_size
  // max_concurrent_wgs = total_hw_threads / subgroups_per_wg
  //
  // When nsegments < max_concurrent_wgs, the GPU would be underutilized,
  // so we fall back to the original group radix select (which uses fewer,
  // larger work-groups with multiple elements per thread).
  bool use_sbtopk = false;
  if (nelements > 4096 && k <= SBTOPK_MAX_K) {
    int64_t total_hw_threads = syclGpuEuCount() * syclGpuHWThreadsPerEU();
    constexpr int SUBGROUP_SIZE = 32;
    constexpr int64_t SUBGROUPS_PER_WG = SBTOPK_BLOCK / SUBGROUP_SIZE;
    int64_t max_concurrent_wgs = total_hw_threads / SUBGROUPS_PER_WG;
    use_sbtopk = (nsegments >= max_concurrent_wgs);
  }

  Tensor self_;
  bool need_infer_dim = dim != ndim - 1;
  if (!need_infer_dim) {
    self_ = self.contiguous();
  } else {
    self_ = self.transpose(ndim - 1, dim).contiguous();
    std::swap(out_sizes[ndim - 1], out_sizes[dim]);
  }

  Tensor values_, indices_;
  bool newvalues = false;
  bool newindices = false;
  if (!need_infer_dim && values.is_contiguous()) {
    values_ = values;
  } else {
    values_ = at::empty(out_sizes, values.options());
    newvalues = true;
  }
  if (!need_infer_dim && indices.is_contiguous()) {
    indices_ = indices;
  } else {
    indices_ = at::empty(out_sizes, indices.options());
    newindices = true;
  }

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self_.scalar_type(),
      "topk_xpu",
      [&]() {
        scalar_t* self_ptr = self_.data_ptr<scalar_t>();
        scalar_t* values_ptr = values_.data_ptr<scalar_t>();
        int64_t* indices_ptr = indices_.data_ptr<int64_t>();

        if (use_sbtopk) {
          sbtopk_launch<scalar_t>(
              self_ptr,
              values_ptr,
              indices_ptr,
              nsegments,
              nelements,
              k,
              largest);
        } else {
          segmented_group_select_pairs<scalar_t, int64_t>(
              self_ptr,
              (scalar_t*)values_ptr,
              nullptr,
              (int64_t*)indices_ptr,
              nsegments,
              nelements,
              k,
              largest);
        }

        if (sorted) {
          segmented_sort_pairs<scalar_t, int64_t>(
              values_ptr,
              values_ptr,
              indices_ptr,
              indices_ptr,
              nsegments,
              k,
              largest);
        }
      });

  if (newvalues) {
    if (need_infer_dim)
      values_.transpose_(ndim - 1, dim);
    values.copy_(values_);
  }
  if (newindices) {
    if (need_infer_dim)
      indices_.transpose_(ndim - 1, dim);
    indices.copy_(indices_);
  }
}

} // namespace xpu
} // namespace native
} // namespace at
