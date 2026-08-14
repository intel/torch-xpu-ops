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

#include <ATen/ceil_div.h>
#include <ATen/native/Resize.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <comm/SYCLContext.h>
#include <comm/XPUMathCompat.h>

namespace at::native::xpu {

std::tuple<int, int, int, int> get_adaptive_config(
    const int reduction,
    const int n_channels,
    const int vec_size,
    int max_wg_size,
    int loops_per_item = 8) {
  loops_per_item /= vec_size;
  int group_size_x = std::min(last_pow2(n_channels / vec_size), 32);
  int group_size_y = std::min(
      last_pow2(at::ceil_div(reduction, loops_per_item)),
      max_wg_size / group_size_x);
  if (group_size_x * group_size_y != max_wg_size) {
    group_size_x =
        std::min(last_pow2(n_channels / vec_size), max_wg_size / group_size_y);
  }

  int nwg_x = at::ceil_div(n_channels, group_size_x * vec_size);
  int nwg_y = std::min(
      at::ceil_div(reduction, group_size_y * loops_per_item),
      int(syclMaxWorkItemsPerTile()) / (nwg_x * group_size_x) / (group_size_y));
  // When per-WG row bytes is not a multiple of 256B (DRAM read granularity),
  // nwg_x > 1 causes adjacent WGs to share DRAM transactions, leading to
  // read amplification. Collapse to nwg_x=1 when occupancy permits.
  int per_wg_channels = group_size_x * vec_size;
  int per_wg_row_bytes = per_wg_channels * 2; // fp16 Vec kernel
  if (nwg_x > 1 && per_wg_row_bytes % 256 != 0) {
    // Check whether nwg_x=1 would leave tile occupancy unchanged.
    // For B580 with WG=1024, tile has at most 2 such WGs, so nwg_x=1
    // is preferred whenever it does not reduce total WG count.
    int nwg_y_x1 = std::min(
        at::ceil_div(reduction, group_size_y * loops_per_item),
        int(syclMaxWorkItemsPerTile()) / (1 * group_size_x) / (group_size_y));
    nwg_y_x1 = std::max(nwg_y_x1, 1);
    if (nwg_y_x1 >= nwg_y) {
      nwg_x = 1;
    }
  }
  // Cap per-WG streaming volume at ~1MB so each WG's working set stays
  // L2-resident. Large-reduction shapes would otherwise be pinned to a low
  // nwg_y by the tile cap and stream multi-MB per WG, which inflates L3
  // partial-write traffic and caps bandwidth. Splitting the reduction into
  // more cooperative groups keeps each WG's slice L2-friendly. Shapes whose
  // per-WG read is already under budget (good residency) are left untouched,
  // and nwg_y stays bounded by the work actually required.
  const int64_t per_wg_byte_budget = 1 << 20;  // ~1MB
  const int bytes_per_elem = 2;                // fp16 Vec kernel
  const int64_t bytes_per_row =
      int64_t(n_channels / nwg_x) * bytes_per_elem;
  const int64_t rows_budget =
      std::max<int64_t>(per_wg_byte_budget / bytes_per_row, 1);
  nwg_y = std::max(nwg_y, 1);  // must be >= 1 before dividing below
  const int64_t rows_now = at::ceil_div((int64_t)reduction, (int64_t)nwg_y);
  if (rows_now > rows_budget) {
    nwg_y = (int)at::ceil_div((int64_t)reduction, rows_budget);
  }
  nwg_y = std::min(
      nwg_y, at::ceil_div(reduction, group_size_y * loops_per_item));
  nwg_y = std::max(nwg_y, 1);
  // dimension is not big enough
  // nwg_y = nwg_y < 4 ? 1 : nwg_y; // removed: allow small nwg_y for better occupancy

  return std::make_tuple(group_size_y, group_size_x, nwg_y, nwg_x);
}



// Hot loop uses sum-based accumulators (sum, sum_sq, count) to avoid the
// loop-carried divide chain that caused large DistStall on B580. Once per
// work-item the K sum-based accumulators are converted to Welford tuples
// (mean, m2n, count) and all downstream reductions (K-merge, in-WG vertical
// merge, cross-WG merge) use numerically stable Welford combine. This
// mirrors the pattern used in torch-xpu-ops PR #4132 (fused GroupNorm) and
// upstream CUDA batch_norm_collect_statistics_channels_last_kernel.
template <typename T, typename C>
inline void welford_merge(
    C& count,
    T& mean,
    T& m2n,
    const C& count_new,
    const T& mean_new,
    const T& m2n_new) {
  T factor = T(1.0) / std::max(1, (count + count_new));
  T delta0 = mean - mean_new;
  mean = (mean_new * count_new + mean * count) * factor;
  m2n += m2n_new + delta0 * delta0 * count_new * count * factor;
  count += count_new;
}

template <int VEC_SIZE, typename T, typename C, typename TACC, typename CACC>
inline void welford_vertical_merge(
    sycl::nd_item<2>& item,
    C& count,
    T& mean,
    T& m2n,
    CACC& shmem_count,
    TACC& shmem_mean,
    TACC& shmem_m2n) {
  // write to shared memory
  auto address_base = item.get_local_linear_id();
#pragma unroll
  for (int offset = item.get_local_range(0) / 2; offset > 0; offset >>= 1) {
    if (item.get_local_id(0) < offset * 2) {
      shmem_mean[address_base] = mean;
      shmem_m2n[address_base] = m2n;
      shmem_count[address_base] = count;
    }
    sycl::group_barrier(item.get_group());
    if (item.get_local_id(0) < offset &&
        item.get_local_id(0) + offset < item.get_local_range(0)) {
      auto address = address_base + offset * item.get_local_range(1);
      // read shared memory back to register for reduction
      auto count_new = shmem_count[address];
      auto mean_new = shmem_mean[address];
      auto m2n_new = shmem_m2n[address];
#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v) {
        welford_merge(
            count[v], mean[v], m2n[v], count_new[v], mean_new[v], m2n_new[v]);
      }
    }
  }
}

template <
    typename VarTransform,
    typename scalar_t,
    typename acc_t,
    int VEC_SIZE = 2>
struct WelfordBatchNormStatChannelsLastVecKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  using vec_t = memory::aligned_vector<scalar_t, VEC_SIZE>;
  using acc_vec_t = memory::aligned_vector<acc_t, VEC_SIZE>;
  using int_vec_t = memory::aligned_vector<int, VEC_SIZE>;

  void operator()(sycl::nd_item<2> item) const {
    //  init private counter
    acc_vec_t mean;
    acc_vec_t m2n;
    int_vec_t count;
#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
      mean[v] = acc_t(0);
      m2n[v] = acc_t(0);
      count[v] = int(0);
    }

    int gy = item.get_group(0);
    int gx = item.get_group(1);
    int c_vec_offset = item.get_global_id(1) * VEC_SIZE;
    int num_cooperative_groups = item.get_group_range(0);
    int inner_loop_stride = item.get_local_range(0) * num_cooperative_groups;

    for (int m_offset = item.get_global_id(0); m_offset < reduction_size_;
         m_offset += inner_loop_stride) {
      if (c_vec_offset < n_channels_) {
        int address_vec_base = m_offset * n_channels_ + c_vec_offset;
        auto input_vec = *reinterpret_cast<vec_t*>(
            const_cast<scalar_t*>(&input_[address_vec_base]));
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
          auto x = input_vec[v];
          count[v]++;
          acc_t delta0 = x - mean[v];
          mean[v] += delta0 / count[v];
          acc_t delta1 = x - mean[v];
          m2n[v] += delta0 * delta1;
        }
      }
    }

    welford_vertical_merge<VEC_SIZE>(
        item, count, mean, m2n, shmem_count_, shmem_mean_, shmem_m2n_);

    // welford vertical merge
    if (num_cooperative_groups > 1) {
      acc_t* staging_mean = staging_data_;
      acc_t* staging_m2n = &staging_data_[n_channels_ * num_cooperative_groups];
      int* staging_count = reinterpret_cast<int*>(
          &staging_m2n[n_channels_ * num_cooperative_groups]);
      int address_vec_base = c_vec_offset + gy * n_channels_;

      // write data to staging_data;
      if (item.get_local_id(0) == 0 && c_vec_offset < n_channels_) {
        *reinterpret_cast<acc_vec_t*>(&staging_mean[address_vec_base]) = mean;
        *reinterpret_cast<acc_vec_t*>(&staging_m2n[address_vec_base]) = m2n;
        *reinterpret_cast<int_vec_t*>(&staging_count[address_vec_base]) = count;
      }
      sycl::group_barrier(item.get_group());

      // mark group done
      if (item.get_local_linear_id() == 0) {
        sycl_atomic_ref_rlx_dev_global_t<int> atomic_count(semaphores_[gx]);
        int old = atomic_count.fetch_add(
            1, sycl_mem_odr_acq_rel
            /* , default memory scope is device */);
        is_last_group_done_[0] = (old == (num_cooperative_groups - 1));
      }
      sycl::group_barrier(item.get_group());

      // check that all data is now available in global memory
      if (is_last_group_done_[0]) {
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
          mean[v] = acc_t(0);
          m2n[v] = acc_t(0);
          count[v] = int(0);
        }

        for (int y = item.get_local_id(0); y < num_cooperative_groups;
             y += item.get_local_range(0)) {
          if (c_vec_offset < n_channels_) {
            address_vec_base = y * n_channels_ + c_vec_offset;
            auto mean_new =
                *reinterpret_cast<acc_vec_t*>(&staging_mean[address_vec_base]);
            auto m2n_new =
                *reinterpret_cast<acc_vec_t*>(&staging_m2n[address_vec_base]);
            auto count_new =
                *reinterpret_cast<int_vec_t*>(&staging_count[address_vec_base]);
#pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v) {
              welford_merge(
                  count[v],
                  mean[v],
                  m2n[v],
                  count_new[v],
                  mean_new[v],
                  m2n_new[v]);
            }
          }
        }
        welford_vertical_merge<VEC_SIZE>(
            item, count, mean, m2n, shmem_count_, shmem_mean_, shmem_m2n_);
      }
    }

    if (item.get_local_id(0) == 0 &&
        (num_cooperative_groups == 1 || is_last_group_done_[0]) &&
        c_vec_offset < n_channels_) {
      acc_vec_t invstd_vec;
#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v) {
        invstd_vec[v] = VarTransform{}(m2n[v] / count[v], epsilon_);
      }

      *reinterpret_cast<acc_vec_t*>(&save_mean_[c_vec_offset]) = mean;
      *reinterpret_cast<acc_vec_t*>(&save_invstd_[c_vec_offset]) = invstd_vec;
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    auto local_size = group_size_x_ * group_size_y_;
    shmem_mean_ = sycl_local_acc_t<acc_vec_t>(sycl::range<1>(local_size), cgh);
    shmem_m2n_ = sycl_local_acc_t<acc_vec_t>(sycl::range<1>(local_size), cgh);
    shmem_count_ = sycl_local_acc_t<int_vec_t>(sycl::range<1>(local_size), cgh);
    is_last_group_done_ = sycl_local_acc_t<bool>(sycl::range<1>(1), cgh);
  }

  WelfordBatchNormStatChannelsLastVecKernelFunctor(
      const scalar_t* input,
      acc_t* save_mean,
      acc_t* save_invstd,
      int reduction_size,
      int n_channels,
      acc_t* staging_data,
      int* semaphores,
      double epsilon)
      : input_(input),
        save_mean_(save_mean),
        save_invstd_(save_invstd),
        reduction_size_(reduction_size),
        n_channels_(n_channels),
        staging_data_(staging_data),
        semaphores_(semaphores),
        epsilon_(epsilon) {}

  void init() {
    using KernelT = WelfordBatchNormStatChannelsLastVecKernelFunctor<
        VarTransform,
        scalar_t,
        acc_t,
        VEC_SIZE>;
    auto max_group_size = syclMaxWorkGroupSize<KernelT>();
    std::tie(group_size_y_, group_size_x_, ngroups_y_, ngroups_x_) =
        get_adaptive_config(
            reduction_size_, n_channels_, VEC_SIZE, max_group_size);
  }

  static bool valid(
      int reduction_size,
      int n_channels,
      const scalar_t* input,
      acc_t* save_mean,
      acc_t* save_invstd) {
    bool valid = sizeof(scalar_t) <= 2;
    valid = valid && (n_channels % VEC_SIZE == 0);
    valid = valid &&
        (memory::can_vectorize_up_to<scalar_t>((char*)input) >= VEC_SIZE);
    valid = valid &&
        (memory::can_vectorize_up_to<acc_t>((char*)save_mean) >= VEC_SIZE);
    valid = valid &&
        (memory::can_vectorize_up_to<acc_t>((char*)save_invstd) >= VEC_SIZE);
    return valid;
  }

  sycl::range<2> local_range() const {
    return sycl::range<2>(group_size_y_, group_size_x_);
  }

  sycl::range<2> global_range() const {
    return sycl::range<2>(
        group_size_y_ * ngroups_y_, group_size_x_ * ngroups_x_);
  }

  int staging_size() const {
    return ngroups_y_ * n_channels_ * 4;
  }

  int semaphores_size() const {
    return ngroups_x_;
  }

  bool set_staging_data_check(acc_t* staging_data) {
    staging_data_ = staging_data;
    return (
        (staging_data == nullptr) ||
        (memory::can_vectorize_up_to<acc_t>((char*)staging_data) >= VEC_SIZE));
  }

  void set_semaphores(int* semaphores) {
    semaphores_ = semaphores;
  }

  int num_cooperative_groups() const {
    return ngroups_y_;
  }

 private:
  const scalar_t* input_;
  acc_t* save_mean_;
  acc_t* save_invstd_;
  int reduction_size_;
  int n_channels_;
  acc_t* staging_data_;
  int* semaphores_;
  double epsilon_;

  size_t group_size_y_;
  size_t group_size_x_;
  size_t ngroups_y_;
  size_t ngroups_x_;

  sycl_local_acc_t<acc_vec_t> shmem_mean_;
  sycl_local_acc_t<acc_vec_t> shmem_m2n_;
  sycl_local_acc_t<int_vec_t> shmem_count_;
  sycl_local_acc_t<bool> is_last_group_done_;
};

} // namespace at::native::xpu
