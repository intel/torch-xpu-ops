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
  nwg_y = std::max(nwg_y, 1);

  return std::make_tuple(group_size_y, group_size_x, nwg_y, nwg_x);
}

template <typename T, typename C>
inline void welford_merge(
    C& count,
    T& mean,
    T& m2n,
    const C& count_new,
    const T& mean_new,
    const T& m2n_new) {
  if (count_new == 0)
    return;
  C new_count = count + count_new;
  T nb_over_n = T(count_new) / T(new_count);
  T delta = mean_new - mean;
  mean += delta * nb_over_n;
  m2n += m2n_new + delta * delta * T(count) * nb_over_n;
  count = new_count;
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
      auto count_new = shmem_count[address];
      auto mean_new = shmem_mean[address];
      auto m2n_new = shmem_m2n[address];
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
}

// ============================================================
// ORIGINAL kernel: handles nwg_x >= 1, one chunk per WI
// ============================================================
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

  static constexpr int K = 4;

  void operator()(sycl::nd_item<2> item) const {
    acc_vec_t sum_k[K];
    acc_vec_t sum_sq_k[K];
    int_vec_t count_k[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v) {
        sum_k[k][v] = acc_t(0);
        sum_sq_k[k][v] = acc_t(0);
        count_k[k][v] = int(0);
      }
    }

    int gy = item.get_group(0);
    int gx = item.get_group(1);
    int c_vec_offset = item.get_global_id(1) * VEC_SIZE;
    int num_cooperative_groups = item.get_group_range(0);
    int inner_loop_stride = item.get_local_range(0) * num_cooperative_groups;

    if (c_vec_offset < n_channels_) {
      int m_offset = item.get_global_id(0);
      int unroll_stride = inner_loop_stride * K;

      const vec_t* base_ptr = reinterpret_cast<const vec_t*>(
          const_cast<scalar_t*>(
              &input_[m_offset * n_channels_ + c_vec_offset]));
      const int k_stride_vec = inner_loop_stride * (n_channels_ / VEC_SIZE);
      const int iter_stride_vec = unroll_stride * (n_channels_ / VEC_SIZE);

      int m_end = reduction_size_ - (K - 1) * inner_loop_stride;
      for (; m_offset < m_end;
           m_offset += unroll_stride, base_ptr += iter_stride_vec) {
        vec_t xv[K];
#pragma unroll
        for (int k = 0; k < K; ++k) {
          xv[k] = *(base_ptr + k * k_stride_vec);
        }
#pragma unroll
        for (int k = 0; k < K; ++k) {
#pragma unroll
          for (int v = 0; v < VEC_SIZE; ++v) {
            acc_t x = acc_t(xv[k][v]);
            count_k[k][v]++;
            sum_k[k][v] += x;
            sum_sq_k[k][v] += x * x;
          }
        }
      }
      for (; m_offset < reduction_size_;
           m_offset += inner_loop_stride, base_ptr += k_stride_vec) {
        auto input_vec = *base_ptr;
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
          acc_t x = acc_t(input_vec[v]);
          count_k[0][v]++;
          sum_k[0][v] += x;
          sum_sq_k[0][v] += x * x;
        }
      }
    }

    acc_vec_t mean;
    acc_vec_t m2n;
    int_vec_t count;
#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
      int c0 = count_k[0][v];
      acc_t m0 = c0 > 0 ? sum_k[0][v] / acc_t(c0) : acc_t(0);
      mean[v] = m0;
      m2n[v] = sum_sq_k[0][v] - sum_k[0][v] * m0;
      count[v] = c0;
#pragma unroll
      for (int k = 1; k < K; ++k) {
        int ck = count_k[k][v];
        if (ck == 0)
          continue;
        acc_t mk = sum_k[k][v] / acc_t(ck);
        acc_t m2k = sum_sq_k[k][v] - sum_k[k][v] * mk;
        welford_merge(count[v], mean[v], m2n[v], ck, mk, m2k);
      }
    }

    welford_vertical_merge<VEC_SIZE>(
        item, count, mean, m2n, shmem_count_, shmem_mean_, shmem_m2n_);

    if (num_cooperative_groups > 1) {
      acc_t* staging_mean = staging_data_;
      acc_t* staging_m2n =
          &staging_data_[n_channels_ * num_cooperative_groups];
      int* staging_count = reinterpret_cast<int*>(
          &staging_m2n[n_channels_ * num_cooperative_groups]);
      int address_vec_base = c_vec_offset + gy * n_channels_;

      if (item.get_local_id(0) == 0 && c_vec_offset < n_channels_) {
        *reinterpret_cast<acc_vec_t*>(&staging_mean[address_vec_base]) = mean;
        *reinterpret_cast<acc_vec_t*>(&staging_m2n[address_vec_base]) = m2n;
        *reinterpret_cast<int_vec_t*>(&staging_count[address_vec_base]) =
            count;
      }
      sycl::group_barrier(item.get_group());

      if (item.get_local_linear_id() == 0) {
        sycl_atomic_ref_rlx_dev_global_t<int> atomic_count(semaphores_[gx]);
        int old = atomic_count.fetch_add(1, sycl_mem_odr_acq_rel);
        is_last_group_done_[0] = (old == (num_cooperative_groups - 1));
      }
      sycl::group_barrier(item.get_group());

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
            auto mean_new = *reinterpret_cast<acc_vec_t*>(
                &staging_mean[address_vec_base]);
            auto m2n_new = *reinterpret_cast<acc_vec_t*>(
                &staging_m2n[address_vec_base]);
            auto count_new = *reinterpret_cast<int_vec_t*>(
                &staging_count[address_vec_base]);
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
    shmem_m2n_ =
        sycl_local_acc_t<acc_vec_t>(sycl::range<1>(local_size), cgh);
    shmem_count_ =
        sycl_local_acc_t<int_vec_t>(sycl::range<1>(local_size), cgh);
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
        VarTransform, scalar_t, acc_t, VEC_SIZE>;
    auto max_group_size = syclMaxWorkGroupSize<KernelT>();
    std::tie(group_size_y_, group_size_x_, ngroups_y_, ngroups_x_) =
        get_adaptive_config(
            reduction_size_, n_channels_, VEC_SIZE, max_group_size);
  }

  static bool valid(
      int reduction_size, int n_channels,
      const scalar_t* input, acc_t* save_mean, acc_t* save_invstd) {
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
  int staging_size() const { return ngroups_y_ * n_channels_ * 4; }
  int semaphores_size() const { return ngroups_x_; }

  bool set_staging_data_check(acc_t* staging_data) {
    staging_data_ = staging_data;
    return ((staging_data == nullptr) ||
        (memory::can_vectorize_up_to<acc_t>((char*)staging_data) >= VEC_SIZE));
  }
  void set_semaphores(int* semaphores) { semaphores_ = semaphores; }
  int num_cooperative_groups() const { return ngroups_y_; }

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

// ============================================================
// ROW-OUTER kernel: nwg_x=1, handles n_chunks 2-4
// Separate struct to avoid register pressure spillover
// ============================================================
template <
    typename VarTransform,
    typename scalar_t,
    typename acc_t,
    int VEC_SIZE = 2>
struct WelfordBatchNormStatChannelsLastVecRowOuterKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  using vec_t = memory::aligned_vector<scalar_t, VEC_SIZE>;
  using acc_vec_t = memory::aligned_vector<acc_t, VEC_SIZE>;
  using int_vec_t = memory::aligned_vector<int, VEC_SIZE>;

  static constexpr int K = 4;

  // Helper: merge one chunk's accumulators to mean/m2n, vertical merge, output
  template <typename OutputFunc>
  void finalize_chunk(
      sycl::nd_item<2> item,
      acc_vec_t& sum, acc_vec_t& sum_sq, int_vec_t& cnt,
      OutputFunc output_fn) const {
    acc_vec_t mean, m2n;
    int_vec_t count;
#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
      count[v] = cnt[v];
      mean[v] = count[v] > 0 ? sum[v] / acc_t(count[v]) : acc_t(0);
      m2n[v] = sum_sq[v] - sum[v] * mean[v];
    }
    welford_vertical_merge<VEC_SIZE>(
        item, count, mean, m2n, shmem_count_, shmem_mean_, shmem_m2n_);
    output_fn(count, mean, m2n);
  }

  void operator()(sycl::nd_item<2> item) const {
    // Two chunks, scalar accumulators (no arrays)
    acc_vec_t sum0, sum_sq0, sum1, sum_sq1;
    int_vec_t cnt0, cnt1;
#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
      sum0[v] = acc_t(0); sum_sq0[v] = acc_t(0); cnt0[v] = 0;
      sum1[v] = acc_t(0); sum_sq1[v] = acc_t(0); cnt1[v] = 0;
    }

    int gy = item.get_group(0);
    int num_cooperative_groups = item.get_group_range(0);
    int inner_loop_stride = item.get_local_range(0) * num_cooperative_groups;
    int lid_x = item.get_local_id(1);
    int group_x = item.get_local_range(1);

    int off_vec0 = lid_x;
    int off_vec1 = lid_x + group_x;
    int c_off0 = lid_x * VEC_SIZE;
    int c_off1 = (lid_x + group_x) * VEC_SIZE;
    bool valid0 = c_off0 < n_channels_;
    bool valid1 = c_off1 < n_channels_;

    int m_offset = item.get_global_id(0);
    int unroll_stride = inner_loop_stride * K;
    const int n_channels_vec = n_channels_ / VEC_SIZE;
    const vec_t* base_ptr = reinterpret_cast<const vec_t*>(
        const_cast<scalar_t*>(&input_[m_offset * n_channels_]));
    const int k_stride_vec = inner_loop_stride * n_channels_vec;
    const int iter_stride_vec = unroll_stride * n_channels_vec;

    int m_end = reduction_size_ - (K - 1) * inner_loop_stride;
    vec_t xv[K];
    for (; m_offset < m_end;
         m_offset += unroll_stride, base_ptr += iter_stride_vec) {
      // Chunk 0
#pragma unroll
      for (int k = 0; k < K; ++k)
        xv[k] = *(base_ptr + off_vec0 + k * k_stride_vec);
#pragma unroll
      for (int k = 0; k < K; ++k) {
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
          acc_t x = acc_t(xv[k][v]);
          cnt0[v]++; sum0[v] += x; sum_sq0[v] += x * x;
        }
      }
      // Chunk 1
      if (valid1) {
#pragma unroll
        for (int k = 0; k < K; ++k)
          xv[k] = *(base_ptr + off_vec1 + k * k_stride_vec);
#pragma unroll
        for (int k = 0; k < K; ++k) {
#pragma unroll
          for (int v = 0; v < VEC_SIZE; ++v) {
            acc_t x = acc_t(xv[k][v]);
            cnt1[v]++; sum1[v] += x; sum_sq1[v] += x * x;
          }
        }
      }
    }
    // Tail
    for (; m_offset < reduction_size_;
         m_offset += inner_loop_stride, base_ptr += k_stride_vec) {
      {
        auto xv0 = *(base_ptr + off_vec0);
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
          acc_t x = acc_t(xv0[v]);
          cnt0[v]++; sum0[v] += x; sum_sq0[v] += x * x;
        }
      }
      if (valid1) {
        auto xv0 = *(base_ptr + off_vec1);
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
          acc_t x = acc_t(xv0[v]);
          cnt1[v]++; sum1[v] += x; sum_sq1[v] += x * x;
        }
      }
    }

    // Post-loop: per-chunk vertical merge + output
    if (num_cooperative_groups == 1) {
      // Chunk 0
      finalize_chunk(item, sum0, sum_sq0, cnt0,
          [&](int_vec_t& count, acc_vec_t& mean, acc_vec_t& m2n) {
        if (item.get_local_id(0) == 0 && valid0) {
          acc_vec_t invstd_vec;
#pragma unroll
          for (int v = 0; v < VEC_SIZE; ++v)
            invstd_vec[v] = VarTransform{}(m2n[v] / count[v], epsilon_);
          *reinterpret_cast<acc_vec_t*>(&save_mean_[c_off0]) = mean;
          *reinterpret_cast<acc_vec_t*>(&save_invstd_[c_off0]) = invstd_vec;
        }
      });
      // Chunk 1
      finalize_chunk(item, sum1, sum_sq1, cnt1,
          [&](int_vec_t& count, acc_vec_t& mean, acc_vec_t& m2n) {
        if (item.get_local_id(0) == 0 && valid1) {
          acc_vec_t invstd_vec;
#pragma unroll
          for (int v = 0; v < VEC_SIZE; ++v)
            invstd_vec[v] = VarTransform{}(m2n[v] / count[v], epsilon_);
          *reinterpret_cast<acc_vec_t*>(&save_mean_[c_off1]) = mean;
          *reinterpret_cast<acc_vec_t*>(&save_invstd_[c_off1]) = invstd_vec;
        }
      });
    } else {
      acc_t* staging_mean = staging_data_;
      acc_t* staging_m2n = &staging_data_[n_channels_ * num_cooperative_groups];
      int* staging_count = reinterpret_cast<int*>(
          &staging_m2n[n_channels_ * num_cooperative_groups]);

      // Write staging for chunk 0
      finalize_chunk(item, sum0, sum_sq0, cnt0,
          [&](int_vec_t& count, acc_vec_t& mean, acc_vec_t& m2n) {
        if (item.get_local_id(0) == 0 && valid0) {
          int addr = c_off0 + gy * n_channels_;
          *reinterpret_cast<acc_vec_t*>(&staging_mean[addr]) = mean;
          *reinterpret_cast<acc_vec_t*>(&staging_m2n[addr]) = m2n;
          *reinterpret_cast<int_vec_t*>(&staging_count[addr]) = count;
        }
      });
      // Write staging for chunk 1
      finalize_chunk(item, sum1, sum_sq1, cnt1,
          [&](int_vec_t& count, acc_vec_t& mean, acc_vec_t& m2n) {
        if (item.get_local_id(0) == 0 && valid1) {
          int addr = c_off1 + gy * n_channels_;
          *reinterpret_cast<acc_vec_t*>(&staging_mean[addr]) = mean;
          *reinterpret_cast<acc_vec_t*>(&staging_m2n[addr]) = m2n;
          *reinterpret_cast<int_vec_t*>(&staging_count[addr]) = count;
        }
      });

      sycl::group_barrier(item.get_group());
      if (item.get_local_linear_id() == 0) {
        sycl_atomic_ref_rlx_dev_global_t<int> atomic_count(semaphores_[0]);
        int old = atomic_count.fetch_add(1, sycl_mem_odr_acq_rel);
        is_last_group_done_[0] = (old == (num_cooperative_groups - 1));
      }
      sycl::group_barrier(item.get_group());

      if (is_last_group_done_[0]) {
        // Cross-WG merge chunk 0
        {
          acc_vec_t mean, m2n;
          int_vec_t count;
#pragma unroll
          for (int v = 0; v < VEC_SIZE; ++v) {
            mean[v] = acc_t(0); m2n[v] = acc_t(0); count[v] = 0;
          }
          for (int y = item.get_local_id(0); y < num_cooperative_groups;
               y += item.get_local_range(0)) {
            if (valid0) {
              int addr = y * n_channels_ + c_off0;
              auto mean_new = *reinterpret_cast<acc_vec_t*>(&staging_mean[addr]);
              auto m2n_new = *reinterpret_cast<acc_vec_t*>(&staging_m2n[addr]);
              auto count_new = *reinterpret_cast<int_vec_t*>(&staging_count[addr]);
#pragma unroll
              for (int v = 0; v < VEC_SIZE; ++v)
                welford_merge(count[v], mean[v], m2n[v],
                    count_new[v], mean_new[v], m2n_new[v]);
            }
          }
          welford_vertical_merge<VEC_SIZE>(
              item, count, mean, m2n, shmem_count_, shmem_mean_, shmem_m2n_);
          if (item.get_local_id(0) == 0 && valid0) {
            acc_vec_t invstd_vec;
#pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v)
              invstd_vec[v] = VarTransform{}(m2n[v] / count[v], epsilon_);
            *reinterpret_cast<acc_vec_t*>(&save_mean_[c_off0]) = mean;
            *reinterpret_cast<acc_vec_t*>(&save_invstd_[c_off0]) = invstd_vec;
          }
        }
        // Cross-WG merge chunk 1
        {
          acc_vec_t mean, m2n;
          int_vec_t count;
#pragma unroll
          for (int v = 0; v < VEC_SIZE; ++v) {
            mean[v] = acc_t(0); m2n[v] = acc_t(0); count[v] = 0;
          }
          for (int y = item.get_local_id(0); y < num_cooperative_groups;
               y += item.get_local_range(0)) {
            if (valid1) {
              int addr = y * n_channels_ + c_off1;
              auto mean_new = *reinterpret_cast<acc_vec_t*>(&staging_mean[addr]);
              auto m2n_new = *reinterpret_cast<acc_vec_t*>(&staging_m2n[addr]);
              auto count_new = *reinterpret_cast<int_vec_t*>(&staging_count[addr]);
#pragma unroll
              for (int v = 0; v < VEC_SIZE; ++v)
                welford_merge(count[v], mean[v], m2n[v],
                    count_new[v], mean_new[v], m2n_new[v]);
            }
          }
          welford_vertical_merge<VEC_SIZE>(
              item, count, mean, m2n, shmem_count_, shmem_mean_, shmem_m2n_);
          if (item.get_local_id(0) == 0 && valid1) {
            acc_vec_t invstd_vec;
#pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v)
              invstd_vec[v] = VarTransform{}(m2n[v] / count[v], epsilon_);
            *reinterpret_cast<acc_vec_t*>(&save_mean_[c_off1]) = mean;
            *reinterpret_cast<acc_vec_t*>(&save_invstd_[c_off1]) = invstd_vec;
          }
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    auto local_size = group_size_x_ * group_size_y_;
    shmem_mean_ = sycl_local_acc_t<acc_vec_t>(sycl::range<1>(local_size), cgh);
    shmem_m2n_ = sycl_local_acc_t<acc_vec_t>(sycl::range<1>(local_size), cgh);
    shmem_count_ = sycl_local_acc_t<int_vec_t>(sycl::range<1>(local_size), cgh);
    is_last_group_done_ = sycl_local_acc_t<bool>(sycl::range<1>(1), cgh);
  }

  WelfordBatchNormStatChannelsLastVecRowOuterKernelFunctor(
      const scalar_t* input, acc_t* save_mean, acc_t* save_invstd,
      int reduction_size, int n_channels,
      acc_t* staging_data, int* semaphores, double epsilon)
      : input_(input), save_mean_(save_mean), save_invstd_(save_invstd),
        reduction_size_(reduction_size), n_channels_(n_channels),
        staging_data_(staging_data), semaphores_(semaphores), epsilon_(epsilon) {}

  void init() {
    using KernelT = WelfordBatchNormStatChannelsLastVecRowOuterKernelFunctor<
        VarTransform, scalar_t, acc_t, VEC_SIZE>;
    auto max_group_size = syclMaxWorkGroupSize<KernelT>();
    std::tie(group_size_y_, group_size_x_, ngroups_y_, ngroups_x_) =
        get_adaptive_config(
            reduction_size_, n_channels_, VEC_SIZE, max_group_size);

    ngroups_x_ = 1;

    // Cap nwg_y so total WGs <= tile_cap
    int tile_cap = int(syclMaxWorkItemsPerTile()) /
        ((int)group_size_x_ * (int)group_size_y_);
    tile_cap = std::max(tile_cap, 1);
    int loops_per_item = 8 / VEC_SIZE;
    int64_t nwg_y_max = at::ceil_div(
        reduction_size_, (int)group_size_y_ * loops_per_item);
    ngroups_y_ = std::min(nwg_y_max, (int64_t)(tile_cap / (int)ngroups_x_));
    ngroups_y_ = std::max(ngroups_y_, (size_t)1);
  }

  static bool valid(
      int reduction_size, int n_channels,
      const scalar_t* input, acc_t* save_mean, acc_t* save_invstd) {
    bool v = sizeof(scalar_t) <= 2;
    v = v && (n_channels % VEC_SIZE == 0);
    v = v && (memory::can_vectorize_up_to<scalar_t>((char*)input) >= VEC_SIZE);
    v = v && (memory::can_vectorize_up_to<acc_t>((char*)save_mean) >= VEC_SIZE);
    v = v && (memory::can_vectorize_up_to<acc_t>((char*)save_invstd) >= VEC_SIZE);
    // Only for n_chunks == 2
    int channels_per_tile = 32 * VEC_SIZE;
    int n_chunks = at::ceil_div(n_channels, channels_per_tile);
    v = v && (n_chunks == 2);
    // Only enable when per-WG data >= 1MB (tensor > tile_cap * 1MB)
    int64_t tensor_bytes = (int64_t)reduction_size * n_channels * sizeof(scalar_t);
    int tile_cap = int(syclMaxWorkItemsPerTile()) / (32 * 32);
    tile_cap = std::max(tile_cap, 1);
    v = v && (tensor_bytes > (int64_t)tile_cap * (1 << 20));
    return v;
  }

  sycl::range<2> local_range() const {
    return sycl::range<2>(group_size_y_, group_size_x_);
  }
  sycl::range<2> global_range() const {
    return sycl::range<2>(
        group_size_y_ * ngroups_y_, group_size_x_ * ngroups_x_);
  }
  int staging_size() const { return ngroups_y_ * n_channels_ * 4; }
  int semaphores_size() const { return ngroups_x_; }

  bool set_staging_data_check(acc_t* staging_data) {
    staging_data_ = staging_data;
    return ((staging_data == nullptr) ||
        (memory::can_vectorize_up_to<acc_t>((char*)staging_data) >= VEC_SIZE));
  }
  void set_semaphores(int* semaphores) { semaphores_ = semaphores; }
  int num_cooperative_groups() const { return ngroups_y_; }

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

