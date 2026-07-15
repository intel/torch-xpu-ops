/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/GroupReduceUtils.h>
#include <ATen/native/xpu/sycl/IntegerDivider.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/SharedReduceOps.h>
#include <comm/MemoryFormat.h>
#include <comm/XPUMathCompat.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/GroupNormKernels.h>

namespace at::native::xpu {

#define PREFERRED_VEC_SIZE \
  2 // To reduce register spill, we should not allocate too many

int64_t get_adaptive_workgroup_size(
    int64_t prob_size,
    int simd,
    int max_wg_size) {
  auto max_size = std::min(max_wg_size, simd * simd);
  int64_t sizes[5] = {simd, 64, 128, 256, max_size};
  for (int i = 0; i < 5; ++i) {
    if (prob_size <= sizes[i]) {
      return sizes[i];
    }
  }
  return max_size;
}

template <
    typename scalar_t,
    typename acc_scalar_t,
    typename index_t,
    typename res_t>
struct WelfordOpsXPU
    : public WelfordOps<scalar_t, acc_scalar_t, index_t, res_t> {
  sycl::nd_item<1>& item;

 public:
  using acc_t =
      typename WelfordOps<scalar_t, acc_scalar_t, index_t, res_t>::acc_t;
  inline acc_t shfl_down(acc_t acc, int offset) const {
    auto sg = item.get_sub_group();
    return {
        sycl::shift_group_left(sg, acc.mean, offset),
        sycl::shift_group_left(sg, acc.m2, offset),
        sycl::shift_group_left(sg, acc.n, offset),
        sycl::shift_group_left(sg, acc.nf, offset)};
  }

  WelfordOpsXPU(acc_scalar_t correction, bool take_sqrt, sycl::nd_item<1>& item)
      : WelfordOps<scalar_t, acc_scalar_t, index_t, res_t>(
            correction,
            take_sqrt),
        item(item) {}
};

template <typename T, typename T_ACC, int SIMD>
struct GNRowwiseMomentsFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp =
      WelfordOpsXPU<T_ACC, T_ACC, int64_t, std::pair<T_ACC, T_ACC>>;

  SYCL_REQD_SUB_GROUP_SIZE(SIMD) void operator()(sycl::nd_item<1> item) const {
    const int64_t i = item.get_group(0);
    WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false, item};
    WelfordType val(0, 0, 0, 0);
    for (int64_t j = item.get_local_id(0); j < N_;
         j += item.get_local_range(0)) {
      const int64_t index = i * N_ + j;
      val = welford_op.reduce(val, static_cast<T_ACC>(X_[index]), index);
    }

    val = GroupReduceWithoutBroadcast<WelfordType, WelfordOp, SIMD>(
        item, val, welford_op, shared_);

    if (item.get_local_id(0) == 0) {
      T_ACC m1;
      T_ACC m2;
      std::tie(m2, m1) = welford_op.project(val);
      T_ACC rstd_val = c10::xpu::compat::rsqrt(m2 + static_cast<T_ACC>(eps_));
      mean_[i] = m1;
      rstd_[i] = rstd_val;
      // save off the accelerated-precision output, if different
      if constexpr (!std::is_same_v<T, T_ACC>) {
        mean_acc_[i] = m1;
        rstd_acc_[i] = rstd_val;
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<WelfordType>(SIMD, cgh);
  }

  GNRowwiseMomentsFunctor(
      int64_t N,
      T eps,
      const T* X,
      T* mean,
      T* rstd,
      T_ACC* mean_acc,
      T_ACC* rstd_acc)
      : N_(N),
        eps_(eps),
        X_(X),
        mean_(mean),
        rstd_(rstd),
        mean_acc_(mean_acc),
        rstd_acc_(rstd_acc) {}

 private:
  int64_t N_;
  T eps_;
  const T* X_;
  T* mean_;
  T* rstd_;
  T_ACC* mean_acc_;
  T_ACC* rstd_acc_;
  sycl_local_acc_t<WelfordType> shared_;
};

template <typename T, typename T_ACC, int SIMD, int VEC_SIZE>
struct GNRowwiseMomentsVectorizedFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp =
      WelfordOpsXPU<T_ACC, T_ACC, int64_t, std::pair<T_ACC, T_ACC>>;
  using vec_t = memory::aligned_vector<T, VEC_SIZE>;

  SYCL_REQD_SUB_GROUP_SIZE(SIMD) void operator()(sycl::nd_item<1> item) const {
    WelfordType val[VEC_SIZE];
    WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false, item};
    auto group_start = item.get_group(0) * VEC_SIZE;

#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
      const int64_t i = group_start + v;
      for (int64_t j = item.get_local_id(0) * VEC_SIZE; j < N_;
           j += item.get_local_range(0) * VEC_SIZE) {
        const int64_t vec_index = i * N_ + j;
        vec_t vec_in =
            *reinterpret_cast<vec_t*>(const_cast<T*>(X_) + vec_index);
#pragma unroll
        for (int iv = 0; iv < VEC_SIZE; ++iv) {
          val[v] = welford_op.reduce(
              val[v], static_cast<T_ACC>(vec_in[iv]), vec_index + iv);
        }
      }
    }

#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
      val[v] = GroupReduceWithoutBroadcast<WelfordType, WelfordOp, SIMD>(
          item, val[v], welford_op, shared_);
    }

    if (item.get_local_id(0) == 0) {
      vec_t mean_vec;
      vec_t rstd_vec;
#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v) {
        T_ACC m1;
        T_ACC m2;
        std::tie(m2, m1) = welford_op.project(val[v]);
        T_ACC rstd_val = c10::xpu::compat::rsqrt(m2 + static_cast<T_ACC>(eps_));
        mean_vec[v] = m1;
        rstd_vec[v] = rstd_val;
        if constexpr (!std::is_same_v<T, T_ACC>) {
          mean_acc_[group_start + v] = m1;
          rstd_acc_[group_start + v] = rstd_val;
        }
      }
      *(reinterpret_cast<vec_t*>(mean_ + group_start)) = mean_vec;
      *(reinterpret_cast<vec_t*>(rstd_ + group_start)) = rstd_vec;
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<WelfordType>(SIMD, cgh);
  }

  GNRowwiseMomentsVectorizedFunctor(
      int64_t N,
      T eps,
      const T* X,
      T* mean,
      T* rstd,
      T_ACC* mean_acc,
      T_ACC* rstd_acc)
      : N_(N),
        eps_(eps),
        X_(X),
        mean_(mean),
        rstd_(rstd),
        mean_acc_(mean_acc),
        rstd_acc_(rstd_acc) {}

 private:
  int64_t N_;
  T eps_;
  const T* X_;
  T* mean_;
  T* rstd_;
  T_ACC* mean_acc_;
  T_ACC* rstd_acc_;
  sycl_local_acc_t<WelfordType> shared_;
};

template <typename T, typename T_ACC>
struct ComputeFusedParamsFunctor {
  void operator()(sycl::item<1> item) const {
    auto index = item.get_id(0);
    const int64_t ng = index / (C_ / group_);
    const int64_t c = index % C_;
    const T_ACC scale = (gamma_ == nullptr)
        ? rstd_acc_[ng]
        : rstd_acc_[ng] * static_cast<T_ACC>(gamma_[c]);
    a_[index] = scale;
    b_[index] = -scale * mean_acc_[ng] +
        ((beta_ == nullptr) ? T_ACC(0) : static_cast<T_ACC>(beta_[c]));
  }
  ComputeFusedParamsFunctor(
      int64_t C,
      int64_t group,
      const T_ACC* mean_acc,
      const T_ACC* rstd_acc,
      const T* gamma,
      const T* beta,
      T_ACC* a,
      T_ACC* b)
      : C_(C),
        group_(group),
        mean_acc_(mean_acc),
        rstd_acc_(rstd_acc),
        gamma_(gamma),
        beta_(beta),
        a_(a),
        b_(b) {}

 private:
  int64_t C_;
  int64_t group_;
  const T_ACC* mean_acc_;
  const T_ACC* rstd_acc_;
  const T* gamma_;
  const T* beta_;
  T_ACC* a_;
  T_ACC* b_;
};

template <typename T, typename T_ACC>
struct GroupNorm1dGammaBetaFunctor {
  T operator()(T x, T_ACC mean, T_ACC rstd, T gamma, T beta) const {
    return (static_cast<T_ACC>(x) - mean) * rstd * static_cast<T_ACC>(gamma) +
        static_cast<T_ACC>(beta);
  }
};

template <typename T, typename T_ACC>
struct GroupNorm1dGammaFunctor {
  T operator()(T x, T_ACC mean, T_ACC rstd, T gamma) const {
    return (static_cast<T_ACC>(x) - mean) * rstd * static_cast<T_ACC>(gamma);
  }
};

template <typename T, typename T_ACC>
struct GroupNorm1dBetaFunctor {
  T operator()(T x, T_ACC mean, T_ACC rstd, T beta) const {
    return (static_cast<T_ACC>(x) - mean) * rstd + static_cast<T_ACC>(beta);
  }
};

template <typename T, typename T_ACC>
struct GroupNorm1dFunctor {
  T operator()(T x, T_ACC mean, T_ACC rstd) const {
    return (static_cast<T_ACC>(x) - mean) * rstd;
  }
};

template <typename T, typename T_ACC>
void group_norm_1d_forward(
    const Tensor& X,
    const Tensor& mean_acc,
    const Tensor& rstd_acc,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t group,
    Tensor& Y) {
  const int64_t G = group;
  const int64_t D = C / G;
  if (gamma.defined() && beta.defined()) {
    auto iter = TensorIteratorConfig()
                    .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N, G, D}))
                    .add_owned_const_input(X.view({N, G, D}))
                    .add_owned_input(mean_acc.view({N, G, 1}))
                    .add_owned_input(rstd_acc.view({N, G, 1}))
                    .add_owned_const_input(gamma.view({1, G, D}))
                    .add_owned_const_input(beta.view({1, G, D}))
                    .build();
    gpu_kernel(iter, GroupNorm1dGammaBetaFunctor<T, T_ACC>());
  } else if (gamma.defined()) {
    auto iter = TensorIteratorConfig()
                    .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N, G, D}))
                    .add_owned_const_input(X.view({N, G, D}))
                    .add_owned_input(mean_acc.view({N, G, 1}))
                    .add_owned_input(rstd_acc.view({N, G, 1}))
                    .add_owned_const_input(gamma.view({1, G, D}))
                    .build();
    gpu_kernel(iter, GroupNorm1dGammaFunctor<T, T_ACC>());
  } else if (beta.defined()) {
    auto iter = TensorIteratorConfig()
                    .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N, G, D}))
                    .add_owned_const_input(X.view({N, G, D}))
                    .add_owned_input(mean_acc.view({N, G, 1}))
                    .add_owned_input(rstd_acc.view({N, G, 1}))
                    .add_owned_const_input(beta.view({1, G, D}))
                    .build();
    gpu_kernel(iter, GroupNorm1dBetaFunctor<T, T_ACC>());
  } else {
    auto iter = TensorIteratorConfig()
                    .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N * G, D}))
                    .add_owned_const_input(X.view({N * G, D}))
                    .add_owned_input(mean_acc.view({N * G, 1}))
                    .add_owned_input(rstd_acc.view({N * G, 1}))
                    .build();
    gpu_kernel(iter, GroupNorm1dFunctor<T, T_ACC>());
  }
}

template <typename T, typename T_ACC>
struct GroupNormFunctor {
  T operator()(T x, T_ACC a, T_ACC b) const {
    volatile T_ACC res = a * static_cast<T_ACC>(x) + b;
    return res;
  }
};

template <typename T>
bool can_use_vectorization(T* p, int vec_size) {
  return memory::can_vectorize_up_to<T>((char*)p) >= vec_size;
}

template <typename T_ACC>
struct WelfordState {
  T_ACC mean;
  T_ACC m2;
  T_ACC nf;
};

template <typename T_ACC>
inline WelfordState<T_ACC> welford_combine(
    WelfordState<T_ACC> a,
    WelfordState<T_ACC> b) {
  T_ACC delta = b.mean - a.mean;
  T_ACC total = a.nf + b.nf;
  T_ACC r = (total > T_ACC(0)) ? (b.nf / total) : T_ACC(0);
  return {a.mean + delta * r, a.m2 + b.m2 + delta * delta * a.nf * r, total};
}

// Symmetric Welford merge: assumes a.nf == b.nf (r = 0.5 always).
// Eliminates division and branch from the general formula.
template <typename T_ACC>
inline WelfordState<T_ACC> welford_combine_symmetric(
    WelfordState<T_ACC> a,
    WelfordState<T_ACC> b) {
  T_ACC delta = b.mean - a.mean;
  return {
      a.mean + delta * T_ACC(0.5),
      a.m2 + b.m2 + delta * delta * a.nf * T_ACC(0.5),
      a.nf + b.nf};
}

template <typename T_ACC>
inline WelfordState<T_ACC> welford_shfl(
    sycl::sub_group sg,
    WelfordState<T_ACC> s,
    int offset) {
  return {
      sycl::shift_group_left(sg, s.mean, offset),
      sycl::shift_group_left(sg, s.m2, offset),
      sycl::shift_group_left(sg, s.nf, offset)};
}

template <typename T_ACC>
inline WelfordState<T_ACC> welford_shfl_xor(
    sycl::sub_group sg,
    WelfordState<T_ACC> s,
    int mask) {
  return {
      sycl::permute_group_by_xor(sg, s.mean, mask),
      sycl::permute_group_by_xor(sg, s.m2, mask),
      sycl::permute_group_by_xor(sg, s.nf, mask)};
}

// Small-DS fused GroupNorm forward: DS fits in 1 vec4 per lane.
// WG = 1 SG, flat (N,G) mapping with grid-stride loop.
// When G < SIMD/lanes, packs groups from multiple batch items per SG.
// Gamma/beta preloaded into registers.
template <
    typename T,
    typename T_ACC,
    int SIMD,
    int VEC_SIZE,
    int LANES_PER_GROUP,
    typename index_t>
struct GNFusedForwardSmallFunctor {
  using vec_t = memory::aligned_vector<T, VEC_SIZE>;
  static_assert(VEC_SIZE == 4, "Tree reduction assumes VEC_SIZE == 4");
  static constexpr int DS = LANES_PER_GROUP * VEC_SIZE;
  static constexpr int GROUPS_PER_SG = SIMD / LANES_PER_GROUP;

  SYCL_REQD_SUB_GROUP_SIZE(SIMD) void operator()(sycl::nd_item<1> item) const {
    auto sg = item.get_sub_group();
    const int sg_lid = sg.get_local_linear_id();
    const index_t wg_id = item.get_group(0);

    const int group_in_sg = sg_lid / LANES_PER_GROUP;
    const int local_lid = sg_lid % LANES_PER_GROUP;

    // Flat mapping across (N, G): packs groups from multiple batch items
    // into one SG when G < GROUPS_PER_SG.
    const index_t flat_base =
        static_cast<index_t>(wg_id) * GROUPS_PER_SG + group_in_sg;
    // g = flat_base % G_. Since G_ = 2^k, this is equivalent to flat_base & (G_
    // - 1).
    const index_t g = flat_base & (G_ - 1);
    const index_t total_groups = N_ * G_;
    const index_t stride =
        static_cast<index_t>(item.get_group_range(0)) * GROUPS_PER_SG;

    // Pre-load gamma/beta (g is invariant across iterations since
    // stride % G == 0 for power-of-2 G and GROUPS_PER_SG).
    T_ACC my_gamma[VEC_SIZE];
    T_ACC my_beta[VEC_SIZE];
    const index_t g_offset = g * D_;
#pragma unroll
    for (int v = 0; v < VEC_SIZE; v++) {
      const index_t cv = (local_lid * VEC_SIZE + v) >> log2_S_;
      my_gamma[v] =
          gamma_ ? static_cast<T_ACC>(gamma_[g_offset + cv]) : T_ACC(1);
      my_beta[v] = beta_ ? static_cast<T_ACC>(beta_[g_offset + cv]) : T_ACC(0);
    }

    for (index_t ng = flat_base; ng < total_groups; ng += stride) {
      const T* x_base = X_ + ng * DS;
      T* y_base = Y_ + ng * DS;

      vec_t xv = *reinterpret_cast<const vec_t*>(x_base + local_lid * VEC_SIZE);

      T_ACC x0 = static_cast<T_ACC>(xv[0]);
      T_ACC x1 = static_cast<T_ACC>(xv[1]);
      T_ACC x2 = static_cast<T_ACC>(xv[2]);
      T_ACC x3 = static_cast<T_ACC>(xv[3]);
      constexpr T_ACC inv_vec =
          static_cast<T_ACC>(1.0) / static_cast<T_ACC>(VEC_SIZE);
      T_ACC batch_sum = (x0 + x1) + (x2 + x3);
      T_ACC batch_sum_sq = (x0 * x0 + x1 * x1) + (x2 * x2 + x3 * x3);
      T_ACC batch_mean = batch_sum * inv_vec;
      WelfordState<T_ACC> st = {
          batch_mean,
          batch_sum_sq - batch_sum * batch_mean,
          static_cast<T_ACC>(VEC_SIZE)};

#pragma unroll
      for (int off = LANES_PER_GROUP / 2; off > 0; off >>= 1)
        st = welford_combine_symmetric(st, welford_shfl_xor(sg, st, off));

      // XOR all-reduce: all lanes have the final result, no broadcast needed.
      constexpr T_ACC inv_DS = static_cast<T_ACC>(1.0) / static_cast<T_ACC>(DS);
      const T_ACC mean_val = st.mean;
      const T_ACC rstd_val =
          sycl::rsqrt(st.m2 * inv_DS + static_cast<T_ACC>(eps_));

      if (local_lid == 0) {
        mean_[ng] = static_cast<T>(mean_val);
        rstd_[ng] = static_cast<T>(rstd_val);
        if constexpr (!std::is_same_v<T, T_ACC>) {
          mean_acc_[ng] = mean_val;
          rstd_acc_[ng] = rstd_val;
        }
      }

      vec_t yv;
#pragma unroll
      for (int v = 0; v < VEC_SIZE; v++) {
        yv[v] = static_cast<T>(
            rstd_val * my_gamma[v] * (static_cast<T_ACC>(xv[v]) - mean_val) +
            my_beta[v]);
      }
      *reinterpret_cast<vec_t*>(y_base + local_lid * VEC_SIZE) = yv;
    }
  }

  GNFusedForwardSmallFunctor(
      index_t D,
      int log2_S,
      index_t G,
      index_t N,
      T eps,
      const T* X,
      T* Y,
      const T* gamma,
      const T* beta,
      T* mean,
      T* rstd,
      T_ACC* mean_acc,
      T_ACC* rstd_acc)
      : D_(D),
        log2_S_(log2_S),
        G_(G),
        N_(N),
        eps_(eps),
        X_(X),
        Y_(Y),
        gamma_(gamma),
        beta_(beta),
        mean_(mean),
        rstd_(rstd),
        mean_acc_(mean_acc),
        rstd_acc_(rstd_acc) {}

 private:
  index_t D_;
  int log2_S_;
  index_t G_;
  index_t N_;
  T eps_;
  const T* X_;
  T* Y_;
  const T* gamma_;
  const T* beta_;
  T* mean_;
  T* rstd_;
  T_ACC* mean_acc_;
  T_ACC* rstd_acc_;
};

// Medium-DS fused GroupNorm forward: multiple vec4 loads per lane.
// WG = 1 SG, all SIMD lanes on 1 group, persistent loop over N.
// Gamma/beta fetched on-the-fly (small D, L1-cached).
template <typename T, typename T_ACC, int SIMD, int VEC_SIZE, typename index_t>
struct GNFusedForwardMediumFunctor {
  using vec_t = memory::aligned_vector<T, VEC_SIZE>;
  static_assert(VEC_SIZE == 4, "Tree reduction assumes VEC_SIZE == 4");

  SYCL_REQD_SUB_GROUP_SIZE(SIMD) void operator()(sycl::nd_item<1> item) const {
    auto sg = item.get_sub_group();
    const int sg_lid = sg.get_local_linear_id();
    const index_t wg_id = item.get_group(0);

    const int loads_per_lane = DS_ / (VEC_SIZE * SIMD);
    const T_ACC inv_DS = static_cast<T_ACC>(1.0) / static_cast<T_ACC>(DS_);

    // Each workgroup deals G=g, N=n_begin..n_begin + n_per_wg.
    const index_t g = wg_id % G_;
    const index_t wgs_per_g = item.get_group_range(0) / G_;
    const index_t wg_rank = wg_id / G_;
    const index_t n_per_wg = (N_ + wgs_per_g - 1) / wgs_per_g;
    const index_t n_begin = wg_rank * n_per_wg;
    const index_t g_offset = g * D_;

    for (index_t n = n_begin; n < n_begin + n_per_wg && n < N_; ++n) {
      const index_t ng = n * G_ + g;
      const T* x_base = X_ + ng * DS_;
      T* y_base = Y_ + ng * DS_;

      constexpr int MAX_LOADS = 4;
      vec_t xv_cache[MAX_LOADS];
      WelfordState<T_ACC> st = {0, 0, 0};
      constexpr T_ACC inv_vec =
          static_cast<T_ACC>(1.0) / static_cast<T_ACC>(VEC_SIZE);
      for (int li = 0; li < loads_per_lane; li++) {
        xv_cache[li] = *reinterpret_cast<const vec_t*>(
            x_base + (sg_lid + li * SIMD) * VEC_SIZE);
        T_ACC x0 = static_cast<T_ACC>(xv_cache[li][0]);
        T_ACC x1 = static_cast<T_ACC>(xv_cache[li][1]);
        T_ACC x2 = static_cast<T_ACC>(xv_cache[li][2]);
        T_ACC x3 = static_cast<T_ACC>(xv_cache[li][3]);
        T_ACC batch_sum = (x0 + x1) + (x2 + x3);
        T_ACC batch_sum_sq = (x0 * x0 + x1 * x1) + (x2 * x2 + x3 * x3);
        T_ACC batch_mean = batch_sum * inv_vec;
        WelfordState<T_ACC> batch = {
            batch_mean,
            batch_sum_sq - batch_sum * batch_mean,
            static_cast<T_ACC>(VEC_SIZE)};
        st = welford_combine(st, batch);
      }

      for (int off = SIMD / 2; off > 0; off >>= 1)
        st = welford_combine_symmetric(st, welford_shfl_xor(sg, st, off));

      // XOR all-reduce: all lanes have the final result.
      const T_ACC mean_val = st.mean;
      const T_ACC rstd_val =
          sycl::rsqrt(st.m2 * inv_DS + static_cast<T_ACC>(eps_));

      if (sg_lid == 0) {
        mean_[ng] = static_cast<T>(mean_val);
        rstd_[ng] = static_cast<T>(rstd_val);
        if constexpr (!std::is_same_v<T, T_ACC>) {
          mean_acc_[ng] = mean_val;
          rstd_acc_[ng] = rstd_val;
        }
      }

      for (int li = 0; li < loads_per_lane; li++) {
        vec_t yv;
#pragma unroll
        for (int v = 0; v < VEC_SIZE; v++) {
          const int pos = (sg_lid + li * SIMD) * VEC_SIZE + v;
          const index_t c = static_cast<index_t>(
              s_divider_.div(static_cast<unsigned_index_t>(pos)));
          T_ACC gv =
              gamma_ ? static_cast<T_ACC>(gamma_[g_offset + c]) : T_ACC(1);
          T_ACC bv = beta_ ? static_cast<T_ACC>(beta_[g_offset + c]) : T_ACC(0);
          yv[v] = static_cast<T>(
              rstd_val * gv * (static_cast<T_ACC>(xv_cache[li][v]) - mean_val) +
              bv);
        }
        *reinterpret_cast<vec_t*>(y_base + (sg_lid + li * SIMD) * VEC_SIZE) =
            yv;
      }
    }
  }

  GNFusedForwardMediumFunctor(
      index_t D,
      index_t S,
      index_t DS,
      index_t G,
      index_t N,
      T eps,
      const T* X,
      T* Y,
      const T* gamma,
      const T* beta,
      T* mean,
      T* rstd,
      T_ACC* mean_acc,
      T_ACC* rstd_acc)
      : D_(D),
        S_(S),
        DS_(DS),
        G_(G),
        N_(N),
        eps_(eps),
        X_(X),
        Y_(Y),
        gamma_(gamma),
        beta_(beta),
        mean_(mean),
        rstd_(rstd),
        mean_acc_(mean_acc),
        rstd_acc_(rstd_acc),
        s_divider_(static_cast<unsigned_index_t>(S)) {}

 private:
  // See GNFusedForwardFunctor's s_divider_ for rationale (magic-number
  // multiply+shift for 32-bit index_t, plain division fallback for 64-bit).
  using unsigned_index_t = std::make_unsigned_t<index_t>;
  index_t D_;
  index_t S_;
  index_t DS_;
  index_t G_;
  index_t N_;
  T eps_;
  const T* X_;
  T* Y_;
  const T* gamma_;
  const T* beta_;
  T* mean_;
  T* rstd_;
  T_ACC* mean_acc_;
  T_ACC* rstd_acc_;
  at::detail::IntDivider<unsigned_index_t> s_divider_;
};

// Fused GroupNorm forward kernel: combines reduction + normalization
// in a single kernel launch to avoid reading X twice from global memory.
// Each workgroup handles one (n, g) group. Pass 1 computes mean/rstd via
// batched Welford (sum within vec4, Welford merge across iterations);
// Pass 2 applies Y = a[c]*X + b[c] with coefficients either precomputed
// in SLM or computed on-the-fly depending on SLM budget.
template <typename T, typename T_ACC, int SIMD, int VEC_SIZE, typename index_t>
struct GNFusedForwardFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  using vec_t = memory::aligned_vector<T, VEC_SIZE>;

  SYCL_REQD_SUB_GROUP_SIZE(SIMD) void operator()(sycl::nd_item<1> item) const {
    static_assert(VEC_SIZE == 4, "Tree reduction assumes VEC_SIZE == 4");
    const index_t ng = item.get_group(0);
    const int lid = item.get_local_id(0);
    const int wg_size = item.get_local_range(0);

    const T* x_base = X_ + static_cast<int64_t>(ng) * DS_;
    T* y_base = Y_ + static_cast<int64_t>(ng) * DS_;

    // Compute aligned region for vectorized access
    const int ptr_elem =
        static_cast<int>(reinterpret_cast<uintptr_t>(x_base) / sizeof(T));
    const index_t head = ((VEC_SIZE - ptr_elem % VEC_SIZE) % VEC_SIZE);
    const index_t n_vecs = (DS_ - head) / VEC_SIZE;
    const index_t tail_start = head + n_vecs * VEC_SIZE;

    // === Pass 1: Batched Welford reduction ===
    WelfordState<T_ACC> st = {0, 0, 0};

    // Head + tail (work-item 0 only): accumulate as sum, merge once
    T_ACC ht_sum = 0, ht_sum_sq = 0;
    int ht_count = 0;
    if (lid == 0) {
      for (index_t j = 0; j < head; j++) {
        T_ACC xf = static_cast<T_ACC>(x_base[j]);
        ht_sum += xf;
        ht_sum_sq += xf * xf;
      }
      for (index_t j = tail_start; j < DS_; j++) {
        T_ACC xf = static_cast<T_ACC>(x_base[j]);
        ht_sum += xf;
        ht_sum_sq += xf * xf;
      }
      ht_count = static_cast<int>(head + DS_ - tail_start);
    }

    // Vectorized middle: tree-reduce within vec4, Welford merge across iters
    constexpr T_ACC inv_vec =
        static_cast<T_ACC>(1.0) / static_cast<T_ACC>(VEC_SIZE);
    for (index_t vi = lid; vi < n_vecs; vi += wg_size) {
      vec_t xv = *reinterpret_cast<const vec_t*>(x_base + head + vi * VEC_SIZE);
      T_ACC x0 = static_cast<T_ACC>(xv[0]);
      T_ACC x1 = static_cast<T_ACC>(xv[1]);
      T_ACC x2 = static_cast<T_ACC>(xv[2]);
      T_ACC x3 = static_cast<T_ACC>(xv[3]);
      T_ACC batch_sum = (x0 + x1) + (x2 + x3);
      T_ACC batch_sum_sq = (x0 * x0 + x1 * x1) + (x2 * x2 + x3 * x3);
      T_ACC batch_mean = batch_sum * inv_vec;
      T_ACC batch_M2 = batch_sum_sq - batch_sum * batch_mean;
      st = welford_combine(
          st,
          WelfordState<T_ACC>{
              batch_mean, batch_M2, static_cast<T_ACC>(VEC_SIZE)});
    }

    // Merge head+tail into work-item 0's Welford state
    if (ht_count > 0) {
      T_ACC ht_nf = static_cast<T_ACC>(ht_count);
      T_ACC ht_mean = ht_sum / ht_nf;
      T_ACC ht_M2 = ht_sum_sq - ht_sum * ht_mean;
      st = welford_combine(st, WelfordState<T_ACC>{ht_mean, ht_M2, ht_nf});
    }

    // Reduce across workgroup via Welford parallel merge
    auto sg = item.get_sub_group();
    int sg_tid = sg.get_local_linear_id();
    int sg_id = sg.get_group_linear_id();
    int n_sg = wg_size / SIMD;

    // Intra-subgroup Welford merge
    for (int off = SIMD / 2; off > 0; off >>= 1) {
      WelfordState<T_ACC> r = welford_shfl(sg, st, off);
      st = welford_combine(st, r);
    }

    // Each subgroup leader writes to SLM
    if (sg_tid == 0) {
      reduce_shared_[sg_id * 3] = st.mean;
      reduce_shared_[sg_id * 3 + 1] = st.m2;
      reduce_shared_[sg_id * 3 + 2] = st.nf;
    }
    sycl::group_barrier(item.get_group());

    // First subgroup reduces all partial results
    if (sg_id == 0) {
      st.mean = (sg_tid < n_sg) ? reduce_shared_[sg_tid * 3] : T_ACC(0);
      st.m2 = (sg_tid < n_sg) ? reduce_shared_[sg_tid * 3 + 1] : T_ACC(0);
      st.nf = (sg_tid < n_sg) ? reduce_shared_[sg_tid * 3 + 2] : T_ACC(0);
      for (int off = SIMD / 2; off > 0; off >>= 1) {
        WelfordState<T_ACC> r = welford_shfl(sg, st, off);
        st = welford_combine(st, r);
      }
    }

    // Work-item 0 computes final rstd and broadcasts via SLM
    if (lid == 0) {
      T_ACC var = st.m2 / st.nf;
      T_ACC rstd_val = sycl::rsqrt(var + static_cast<T_ACC>(eps_));
      mean_[ng] = static_cast<T>(st.mean);
      rstd_[ng] = static_cast<T>(rstd_val);
      if constexpr (!std::is_same_v<T, T_ACC>) {
        mean_acc_[ng] = st.mean;
        rstd_acc_[ng] = rstd_val;
      }
      broadcast_[0] = st.mean;
      broadcast_[1] = rstd_val;
    }
    sycl::group_barrier(item.get_group());
    const T_ACC g_mean = broadcast_[0];
    const T_ACC g_rstd = broadcast_[1];

    // === Precompute a[c], b[c] in SLM (if budget allows) ===
    const index_t g = ng % G_;
    const index_t g_offset = g * D_;
    if (use_slm_coeff_) {
      if (gamma_ != nullptr && beta_ != nullptr) {
        const T* gp = gamma_ + g_offset;
        const T* bp = beta_ + g_offset;
        const int gp_elem =
            static_cast<int>(reinterpret_cast<uintptr_t>(gp) / sizeof(T));
        // Clamp to D_: for D_ < VEC_SIZE (e.g. num_groups == num_channels),
        // the raw alignment-derived head can exceed D_, which would make
        // c_nvecs truncate to 0 and c_tail stay above D_ -- the head loop
        // below would then run past D_, reading gp/bp out of bounds and
        // writing beyond coeff_'s intended a[]/b[] sub-ranges.
        const index_t c_head = std::min(
            static_cast<index_t>((VEC_SIZE - gp_elem % VEC_SIZE) % VEC_SIZE),
            D_);
        const index_t c_nvecs = (D_ - c_head) / VEC_SIZE;
        const index_t c_tail = c_head + c_nvecs * VEC_SIZE;
        if (lid == 0) {
          for (index_t c = 0; c < c_head; c++) {
            T_ACC gv = static_cast<T_ACC>(gp[c]);
            T_ACC bv = static_cast<T_ACC>(bp[c]);
            T_ACC a_c = g_rstd * gv;
            coeff_[c] = a_c;
            coeff_[D_ + c] = bv - g_mean * a_c;
          }
        }
        for (index_t ci = lid; ci < c_nvecs; ci += wg_size) {
          const index_t c = c_head + ci * VEC_SIZE;
          vec_t gv_vec = *reinterpret_cast<const vec_t*>(gp + c);
          vec_t bv_vec = *reinterpret_cast<const vec_t*>(bp + c);
#pragma unroll
          for (int v = 0; v < VEC_SIZE; v++) {
            T_ACC gv = static_cast<T_ACC>(gv_vec[v]);
            T_ACC bv = static_cast<T_ACC>(bv_vec[v]);
            T_ACC a_c = g_rstd * gv;
            coeff_[c + v] = a_c;
            coeff_[D_ + c + v] = bv - g_mean * a_c;
          }
        }
        if (lid == 0) {
          for (index_t c = c_tail; c < D_; c++) {
            T_ACC gv = static_cast<T_ACC>(gp[c]);
            T_ACC bv = static_cast<T_ACC>(bp[c]);
            T_ACC a_c = g_rstd * gv;
            coeff_[c] = a_c;
            coeff_[D_ + c] = bv - g_mean * a_c;
          }
        }
      } else {
        for (index_t c = lid; c < D_; c += wg_size) {
          const index_t gc = g_offset + c;
          const T_ACC gv =
              (gamma_ != nullptr) ? static_cast<T_ACC>(gamma_[gc]) : T_ACC(1);
          const T_ACC bv =
              (beta_ != nullptr) ? static_cast<T_ACC>(beta_[gc]) : T_ACC(0);
          const T_ACC a_c = g_rstd * gv;
          coeff_[c] = a_c;
          coeff_[D_ + c] = bv - g_mean * a_c;
        }
      }
      sycl::group_barrier(item.get_group());
    }

    // === Pass 2: Y = a[c]*X + b[c] ===
    if (use_slm_coeff_) {
      // Use precomputed coefficients from SLM
      if (lid == 0) {
        for (index_t j = 0; j < head; j++) {
          const index_t c = static_cast<index_t>(
              s_divider_.div(static_cast<unsigned_index_t>(j)));
          y_base[j] = static_cast<T>(
              coeff_[c] * static_cast<T_ACC>(x_base[j]) + coeff_[D_ + c]);
        }
      }
      for (index_t vi = lid; vi < n_vecs; vi += wg_size) {
        const index_t j = head + vi * VEC_SIZE;
        vec_t xv =
            *reinterpret_cast<const vec_t*>(x_base + head + vi * VEC_SIZE);
        vec_t yv;
#pragma unroll
        for (int v = 0; v < VEC_SIZE; v++) {
          const index_t c = static_cast<index_t>(
              s_divider_.div(static_cast<unsigned_index_t>(j + v)));
          yv[v] = static_cast<T>(
              coeff_[c] * static_cast<T_ACC>(xv[v]) + coeff_[D_ + c]);
        }
        *reinterpret_cast<vec_t*>(y_base + j) = yv;
      }
      if (lid == 0) {
        for (index_t j = tail_start; j < DS_; j++) {
          const index_t c = static_cast<index_t>(
              s_divider_.div(static_cast<unsigned_index_t>(j)));
          y_base[j] = static_cast<T>(
              coeff_[c] * static_cast<T_ACC>(x_base[j]) + coeff_[D_ + c]);
        }
      }
    } else {
      // Compute coefficients on-the-fly (gamma/beta L1/L2 cached)
      if (lid == 0) {
        for (index_t j = 0; j < head; j++) {
          const index_t c = static_cast<index_t>(
              s_divider_.div(static_cast<unsigned_index_t>(j)));
          const index_t gc = g_offset + c;
          T_ACC gv = (gamma_) ? static_cast<T_ACC>(gamma_[gc]) : T_ACC(1);
          T_ACC bv = (beta_) ? static_cast<T_ACC>(beta_[gc]) : T_ACC(0);
          T_ACC a_c = g_rstd * gv;
          y_base[j] = static_cast<T>(
              a_c * static_cast<T_ACC>(x_base[j]) + bv - g_mean * a_c);
        }
      }
      for (index_t vi = lid; vi < n_vecs; vi += wg_size) {
        const index_t j = head + vi * VEC_SIZE;
        vec_t xv =
            *reinterpret_cast<const vec_t*>(x_base + head + vi * VEC_SIZE);
        vec_t yv;
#pragma unroll
        for (int v = 0; v < VEC_SIZE; v++) {
          const index_t c = static_cast<index_t>(
              s_divider_.div(static_cast<unsigned_index_t>(j + v)));
          const index_t gc = g_offset + c;
          T_ACC gv = (gamma_) ? static_cast<T_ACC>(gamma_[gc]) : T_ACC(1);
          T_ACC bv = (beta_) ? static_cast<T_ACC>(beta_[gc]) : T_ACC(0);
          T_ACC a_c = g_rstd * gv;
          yv[v] = static_cast<T>(
              a_c * static_cast<T_ACC>(xv[v]) + bv - g_mean * a_c);
        }
        *reinterpret_cast<vec_t*>(y_base + j) = yv;
      }
      if (lid == 0) {
        for (index_t j = tail_start; j < DS_; j++) {
          const index_t c = static_cast<index_t>(
              s_divider_.div(static_cast<unsigned_index_t>(j)));
          const index_t gc = g_offset + c;
          T_ACC gv = (gamma_) ? static_cast<T_ACC>(gamma_[gc]) : T_ACC(1);
          T_ACC bv = (beta_) ? static_cast<T_ACC>(beta_[gc]) : T_ACC(0);
          T_ACC a_c = g_rstd * gv;
          y_base[j] = static_cast<T>(
              a_c * static_cast<T_ACC>(x_base[j]) + bv - g_mean * a_c);
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    reduce_shared_ = sycl_local_acc_t<T_ACC>(wg_size_ / SIMD * 3, cgh);
    broadcast_ = sycl_local_acc_t<T_ACC>(2, cgh);
    coeff_ = sycl_local_acc_t<T_ACC>(use_slm_coeff_ ? 2 * D_ : 1, cgh);
  }

  GNFusedForwardFunctor(
      index_t D,
      index_t S,
      index_t DS,
      index_t G,
      T eps,
      const T* X,
      T* Y,
      const T* gamma,
      const T* beta,
      T* mean,
      T* rstd,
      T_ACC* mean_acc,
      T_ACC* rstd_acc,
      int wg_size,
      bool use_slm_coeff)
      : D_(D),
        S_(S),
        DS_(DS),
        G_(G),
        eps_(eps),
        X_(X),
        Y_(Y),
        gamma_(gamma),
        beta_(beta),
        mean_(mean),
        rstd_(rstd),
        mean_acc_(mean_acc),
        rstd_acc_(rstd_acc),
        wg_size_(wg_size),
        use_slm_coeff_(use_slm_coeff),
        s_divider_(static_cast<unsigned_index_t>(S)) {}

 private:
  // at::detail::IntDivider<unsigned int> replaces the Pass-2 channel-index
  // division (c = j / S_) with a precomputed magic-number multiply + shift
  // (see IntegerDivider.h). Falls back to plain division when index_t is
  // 64-bit (IntDivider has no fast specialization for that width, matching
  // the CUDA/XPU IntDivider's own precedent elsewhere in this codebase).
  using unsigned_index_t = std::make_unsigned_t<index_t>;
  index_t D_;
  index_t S_;
  index_t DS_;
  index_t G_;
  T eps_;
  const T* X_;
  T* Y_;
  const T* gamma_;
  const T* beta_;
  T* mean_;
  T* rstd_;
  T_ACC* mean_acc_;
  T_ACC* rstd_acc_;
  int wg_size_;
  bool use_slm_coeff_;
  sycl_local_acc_t<T_ACC> reduce_shared_;
  sycl_local_acc_t<T_ACC> broadcast_;
  sycl_local_acc_t<T_ACC> coeff_;
  at::detail::IntDivider<unsigned_index_t> s_divider_;
};

template <typename T, typename T_ACC = acc_type_device<T, kXPU>>
void group_norm_kernel_impl(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    T eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  TORCH_CHECK(!beta.defined() || beta.numel() == C);
  if (N == 0) {
    return;
  }

  const int64_t G = group;
  const int64_t D = C / G;
  const T* X_data = X.const_data_ptr<T>();
  T* mean_data = mean.mutable_data_ptr<T>();
  T* rstd_data = rstd.mutable_data_ptr<T>();
  auto& queue = getCurrentSYCLQueue();
  int64_t simd = syclMaxSubGroupSize();

  // --- Fused forward path: single kernel for Welford + normalization ---
  constexpr int FUSED_VEC_SIZE = 4;
  int64_t DS = D * HxW;
  bool can_use_int32 = canUse32BitIndexMath(X);
  constexpr int64_t kElemsPerWorkItem = 16;

  T* Y_data = Y.mutable_data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.const_data_ptr<T>() : nullptr;
  const bool needMeanAcc{!std::is_same_v<T, T_ACC>};
  T_ACC* mean_acc_data = nullptr;
  T_ACC* rstd_acc_data = nullptr;
  Tensor mean_acc, rstd_acc;
  if (needMeanAcc) {
    const auto kAccOpts{X.options().dtype(kFloat)};
    mean_acc = at::empty(mean.sizes(), kAccOpts);
    rstd_acc = at::empty(rstd.sizes(), kAccOpts);
    mean_acc_data = mean_acc.mutable_data_ptr<T_ACC>();
    rstd_acc_data = rstd_acc.mutable_data_ptr<T_ACC>();
  }

  int64_t thread_slots = syclGpuEuCount() * syclGpuHWThreadsPerEU();

  // Small-DS path: single vec4 per lane (covers DS where lanes <= SIMD).
  // WG = 1 SG, flat mapping over (N, G) with grid-stride loop.
  auto try_single_vec = [&](auto index_tag) -> bool {
    if (simd < 16)
      return false;
    using index_t = decltype(index_tag);
    if (DS % FUSED_VEC_SIZE != 0)
      return false;
    int64_t lanes = DS / FUSED_VEC_SIZE;
    if (lanes <= 0 || lanes > SIMD32 || (lanes & (lanes - 1)) != 0)
      return false;
    // Require G and HxW to be powers of 2 for bitwise modulo/division.
    if ((G & (G - 1)) != 0 || (HxW & (HxW - 1)) != 0)
      return false;
    int64_t groups_per_sg = SIMD32 / lanes;
    if ((N * G) % groups_per_sg != 0)
      return false;
    int log2_S = 0;
    for (int64_t tmp = HxW; tmp > 1; tmp >>= 1)
      log2_S++;
    int64_t total_sgs = (N * G) / groups_per_sg;
    // Ensure stride preserves g across iterations (stride % G == 0).
    int64_t n_wgs;
    if (G >= groups_per_sg) {
      int64_t g_chunks = G / groups_per_sg;
      n_wgs = std::min(total_sgs, (thread_slots / g_chunks) * g_chunks);
    } else {
      n_wgs = std::min(total_sgs, thread_slots);
    }
    n_wgs = std::max(n_wgs, (int64_t)1);
    constexpr int64_t wg_sz = SIMD32;
    auto launch = [&](auto lanes_tag) {
      constexpr int LANES = decltype(lanes_tag)::value;
      using K = GNFusedForwardSmallFunctor<
          T,
          T_ACC,
          SIMD32,
          FUSED_VEC_SIZE,
          LANES,
          index_t>;
      auto kfn =
          K(static_cast<index_t>(D),
            log2_S,
            static_cast<index_t>(G),
            static_cast<index_t>(N),
            eps,
            X_data,
            Y_data,
            gamma_data,
            beta_data,
            mean_data,
            rstd_data,
            mean_acc_data,
            rstd_acc_data);
      sycl_kernel_submit(
          sycl::range<1>(n_wgs * wg_sz), sycl::range<1>(wg_sz), queue, kfn);
    };
    switch (lanes) {
      case 1:
        launch(std::integral_constant<int, 1>{});
        break;
      case 2:
        launch(std::integral_constant<int, 2>{});
        break;
      case 4:
        launch(std::integral_constant<int, 4>{});
        break;
      case 8:
        launch(std::integral_constant<int, 8>{});
        break;
      case 16:
        launch(std::integral_constant<int, 16>{});
        break;
      case 32:
        launch(std::integral_constant<int, 32>{});
        break;
      default:
        return false;
    }
    return true;
  };
  auto try_multi_vec = [&](auto index_tag) -> bool {
    if (simd < 16)
      return false;
    using index_t = decltype(index_tag);
    constexpr int64_t MAX_LOADS = 4;
    int64_t elems_per_sg = FUSED_VEC_SIZE * SIMD32;
    if (DS % elems_per_sg != 0)
      return false;
    int64_t loads = DS / elems_per_sg;
    if (loads < 1 || loads > MAX_LOADS)
      return false;
    using K =
        GNFusedForwardMediumFunctor<T, T_ACC, SIMD32, FUSED_VEC_SIZE, index_t>;
    int64_t n_wgs = std::max(G, std::min(N * G, thread_slots));
    constexpr int64_t wg_sz = SIMD32;
    auto kfn =
        K(static_cast<index_t>(D),
          static_cast<index_t>(HxW),
          static_cast<index_t>(DS),
          static_cast<index_t>(G),
          static_cast<index_t>(N),
          eps,
          X_data,
          Y_data,
          gamma_data,
          beta_data,
          mean_data,
          rstd_data,
          mean_acc_data,
          rstd_acc_data);
    sycl_kernel_submit(
        sycl::range<1>(n_wgs * wg_sz), sycl::range<1>(wg_sz), queue, kfn);
    return true;
  };
  auto dispatch_packed = [&](auto index_tag) -> bool {
    if (try_single_vec(index_tag))
      return true;
    if (try_multi_vec(index_tag))
      return true;
    return false;
  };
  if (can_use_int32) {
    if (dispatch_packed(int{}))
      return;
  } else {
    if (dispatch_packed(int64_t{}))
      return;
  }

  // Large-DS fused path: check occupancy with VEC_SIZE=4.
  // If n_groups * min(DS/4, max_wg) >= 50% thread slots, use fused kernel.
  int64_t n_groups = N * G;
  int64_t max_wg_est = std::min((int64_t)1024, DS / FUSED_VEC_SIZE);
  bool fused_has_occupancy = (n_groups * max_wg_est >= thread_slots / 2);
  if (fused_has_occupancy && simd == 32) {
    constexpr int64_t wg_choices[] = {32, 64, 128, 256, 512, 1024};
    int64_t wg_size = 32;
    auto launch = [&](auto index_tag) {
      using index_t = decltype(index_tag);
      using K =
          GNFusedForwardFunctor<T, T_ACC, SIMD32, FUSED_VEC_SIZE, index_t>;
      int64_t max_wg = syclMaxWorkGroupSize<K>();
      int64_t ideal = (DS + kElemsPerWorkItem - 1) / kElemsPerWorkItem;
      int64_t min_wg_for_occ =
          ((thread_slots / 2 + n_groups - 1) / n_groups) * simd;
      int64_t max_wg_from_ds = std::min(max_wg, DS / FUSED_VEC_SIZE);
      int64_t target_wg = std::max(ideal, min_wg_for_occ);
      for (int64_t w : wg_choices) {
        if (w <= max_wg_from_ds) {
          wg_size = w;
          if (w >= target_wg)
            break;
        }
      }
      // SLM budget: local_mem_size shared among concurrent WGs per Xe-core
      int64_t eu_per_xc = syclGpuEUCountPerSubslice();
      int64_t hw_thr = syclGpuHWThreadsPerEU();
      int64_t slots_per_xc = eu_per_xc * hw_thr;
      int64_t sgs_per_wg = wg_size / simd;
      int64_t concurrent_wgs = slots_per_xc / sgs_per_wg;
      int64_t slm_per_wg = syclLocalMemSize() / concurrent_wgs;
      bool use_slm_coeff = (2 * D * (int64_t)sizeof(T_ACC) <= slm_per_wg);
      auto kfn =
          K(static_cast<index_t>(D),
            static_cast<index_t>(HxW),
            static_cast<index_t>(DS),
            static_cast<index_t>(G),
            eps,
            X_data,
            Y_data,
            gamma_data,
            beta_data,
            mean_data,
            rstd_data,
            mean_acc_data,
            rstd_acc_data,
            (int)wg_size,
            use_slm_coeff);
      sycl_kernel_submit(
          sycl::range<1>(n_groups * wg_size),
          sycl::range<1>(wg_size),
          queue,
          kfn);
    };
    if (can_use_int32) {
      launch(int{});
    } else {
      launch(int64_t{});
    }
    return;
  }

  // --- Fallback: original 3-kernel path (low occupancy for fused) ---
  const auto kAccTypeOpts{
      X.options().dtype(needMeanAcc ? kFloat : X.scalar_type())};
  if (!mean_acc.defined()) {
    mean_acc = needMeanAcc ? at::empty(mean.sizes(), kAccTypeOpts) : mean;
    rstd_acc = needMeanAcc ? at::empty(rstd.sizes(), kAccTypeOpts) : rstd;
    mean_acc_data = mean_acc.mutable_data_ptr<T_ACC>();
    rstd_acc_data = rstd_acc.mutable_data_ptr<T_ACC>();
  }

  int64_t prob_size = D * HxW;
  int64_t stride = N * G;
  constexpr int VEC_SIZE = PREFERRED_VEC_SIZE;

  if (can_use_vectorization(X_data, VEC_SIZE) &&
      can_use_vectorization(mean_data, VEC_SIZE) &&
      can_use_vectorization(rstd_data, VEC_SIZE) && prob_size % VEC_SIZE == 0 &&
      stride % VEC_SIZE == 0) {
    using KernelS16T =
        GNRowwiseMomentsVectorizedFunctor<T, T_ACC, SIMD16, VEC_SIZE>;
    using KernelS32T =
        GNRowwiseMomentsVectorizedFunctor<T, T_ACC, SIMD32, VEC_SIZE>;
    auto max_size = std::min(
        syclMaxWorkGroupSize<KernelS16T>(), syclMaxWorkGroupSize<KernelS32T>());
    auto wg_size =
        get_adaptive_workgroup_size(prob_size / VEC_SIZE, simd, max_size);
    auto global_range = sycl::range<1>((stride / VEC_SIZE) * wg_size);
    auto local_range = sycl::range<1>(wg_size);
    group_norm_kernel_simd_choice_and_launch<KernelS16T, KernelS32T>(
        simd,
        global_range,
        local_range,
        queue,
        prob_size,
        eps,
        X_data,
        mean_data,
        rstd_data,
        mean_acc_data,
        rstd_acc_data);
  } else {
    using KernelS16T = GNRowwiseMomentsFunctor<T, T_ACC, SIMD16>;
    using KernelS32T = GNRowwiseMomentsFunctor<T, T_ACC, SIMD32>;
    auto max_size = std::min(
        syclMaxWorkGroupSize<KernelS16T>(), syclMaxWorkGroupSize<KernelS32T>());
    auto wg_size = get_adaptive_workgroup_size(prob_size, simd, max_size);
    auto global_range = sycl::range<1>(stride * wg_size);
    auto local_range = sycl::range<1>(wg_size);
    group_norm_kernel_simd_choice_and_launch<KernelS16T, KernelS32T>(
        simd,
        global_range,
        local_range,
        queue,
        prob_size,
        eps,
        X_data,
        mean_data,
        rstd_data,
        mean_acc_data,
        rstd_acc_data);
  }

  if (HxW == 1) {
    group_norm_1d_forward<T, T_ACC>(
        X, mean_acc, rstd_acc, gamma, beta, N, C, G, Y);
  } else if (!gamma.defined() && !beta.defined()) {
    auto iter = TensorIteratorConfig()
                    .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N * G, prob_size}))
                    .add_owned_const_input(X.view({N * G, prob_size}))
                    .add_owned_input(mean_acc.view({N * G, 1}))
                    .add_owned_input(rstd_acc.view({N * G, 1}))
                    .build();
    gpu_kernel(iter, GroupNorm1dFunctor<T, T_ACC>());
  } else {
    Tensor a = at::empty({N, C}, kAccTypeOpts);
    Tensor b = at::empty({N, C}, kAccTypeOpts);
    T_ACC* a_data = a.mutable_data_ptr<T_ACC>();
    T_ACC* b_data = b.mutable_data_ptr<T_ACC>();

    auto caller = ComputeFusedParamsFunctor<T, T_ACC>(
        C,
        G,
        mean_acc_data,
        rstd_acc_data,
        gamma_data,
        beta_data,
        a_data,
        b_data);
    sycl_kernel_submit(sycl::range<1>(N * C), queue, caller);

    TensorIterator iter = TensorIteratorConfig()
                              .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                              .resize_outputs(false)
                              .add_owned_output(Y.view({N * C, HxW}))
                              .add_owned_const_input(X.view({N * C, HxW}))
                              .add_owned_input(a.view({N * C, 1}))
                              .add_owned_input(b.view({N * C, 1}))
                              .build();
    gpu_kernel(iter, GroupNormFunctor<T, T_ACC>());
  }
}

void group_norm_kernel(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "group_norm_xpu",
      [&]() {
        group_norm_kernel_impl<scalar_t>(
            X,
            gamma,
            beta,
            N,
            C,
            HxW,
            group,
            static_cast<scalar_t>(eps),
            Y,
            mean,
            rstd);
      });
}

template <typename T, typename T_ACC, int SIMD>
struct Compute1dBackwardFusedParamsFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  SYCL_REQD_SUB_GROUP_SIZE(SIMD) void operator()(sycl::nd_item<2> item) const {
    const int64_t G = group_;
    const int64_t D = C_ / G;
    const int64_t n = item.get_group(1);
    const int64_t g = item.get_group(0);
    const int64_t ng = n * G + g;
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t i = item.get_local_id(1); i < D;
         i += item.get_local_range(1)) {
      const int64_t index = ng * D + i;
      const int64_t c = g * D + i;
      const T_ACC gamma_v =
          gamma_ == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma_[c]);
      sum1 += dY_[index] * X_[index] * gamma_v;
      sum2 += dY_[index] * gamma_v;
    }
    sum1 = GroupReduceSumWithoutBroadcast<T_ACC, SIMD>(item, sum1, ds_shared_);
    sum2 = GroupReduceSumWithoutBroadcast<T_ACC, SIMD>(item, sum2, db_shared_);
    if (item.get_local_id(1) == 0) {
      const T_ACC s = T_ACC(1) / static_cast<T_ACC>(D);
      const T_ACC x = (sum2 * static_cast<T_ACC>(mean_[ng]) - sum1) *
          static_cast<T_ACC>(rstd_[ng]) * static_cast<T_ACC>(rstd_[ng]) *
          static_cast<T_ACC>(rstd_[ng]) * s;
      c2_[ng] = x;
      c3_[ng] = -x * static_cast<T_ACC>(mean_[ng]) -
          sum2 * static_cast<T_ACC>(rstd_[ng]) * s;
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    ds_shared_ =
        sycl_local_acc_t<T_ACC>(get_group_reduce_group_size(SIMD), cgh);
    db_shared_ =
        sycl_local_acc_t<T_ACC>(get_group_reduce_group_size(SIMD), cgh);
  }

  Compute1dBackwardFusedParamsFunctor(
      int64_t C,
      int64_t group,
      const T* dY,
      const T* X,
      const T* mean,
      const T* rstd,
      const T* gamma,
      T_ACC* c2,
      T_ACC* c3)
      : C_(C),
        group_(group),
        dY_(dY),
        X_(X),
        mean_(mean),
        rstd_(rstd),
        gamma_(gamma),
        c2_(c2),
        c3_(c3) {}

 private:
  int64_t C_;
  int64_t group_;
  const T* dY_;
  const T* X_;
  const T* mean_;
  const T* rstd_;
  const T* gamma_;
  T_ACC* c2_;
  T_ACC* c3_;
  sycl_local_acc_t<T_ACC> ds_shared_;
  sycl_local_acc_t<T_ACC> db_shared_;
};

template <typename T, typename T_ACC>
struct GroupNorm1dBackwardGammaFunctor {
  T operator()(T dy, T x, T rstd, T gamma, T_ACC c2, T_ACC c3) const {
    const T_ACC c1 = static_cast<T_ACC>(rstd) * static_cast<T_ACC>(gamma);
    return c1 * static_cast<T_ACC>(dy) + c2 * static_cast<T_ACC>(x) + c3;
  }
};

template <typename T, typename T_ACC>
struct GroupNorm1dBackwardFunctor {
  T operator()(T dy, T x, T rstd, T_ACC c2, T_ACC c3) const {
    const T_ACC c1 = static_cast<T_ACC>(rstd);
    return c1 * static_cast<T_ACC>(dy) + c2 * static_cast<T_ACC>(x) + c3;
  }
};

template <typename T>
struct GammaBeta1dBackwardSmallKernel {
  void operator()(sycl::nd_item<1> item) const {
    using T_ACC = acc_type_device<T, kXPU>;
    const int64_t c = item.get_local_linear_id();
    if (c < C_) {
      const int64_t G = group_;
      const int64_t D = C_ / G;
      T_ACC sum1 = 0;
      T_ACC sum2 = 0;
      for (int64_t n = 0; n < N_; ++n) {
        const int64_t nc = n * C_ + c;
        const int64_t ng = n * G + c / D;
        const T_ACC dy_acc = static_cast<T_ACC>(dY_[nc]);
        const T_ACC x_acc = static_cast<T_ACC>(X_[nc]);
        sum1 += (dgamma_ == nullptr)
            ? T_ACC(0)
            : ((dy_acc * x_acc - dy_acc * static_cast<T_ACC>(mean_[ng])) *
               static_cast<T_ACC>(rstd_[ng]));
        sum2 += (dbeta_ == nullptr) ? T_ACC(0) : dy_acc;
      }
      if (dgamma_ != nullptr) {
        dgamma_[c] = sum1;
      }
      if (dbeta_ != nullptr) {
        dbeta_[c] = sum2;
      }
    }
  }

  GammaBeta1dBackwardSmallKernel(
      int64_t N,
      int64_t C,
      int64_t group,
      const T* dY,
      const T* X,
      const T* mean,
      const T* rstd,
      T* dgamma,
      T* dbeta)
      : N_(N),
        C_(C),
        group_(group),
        dY_(dY),
        X_(X),
        mean_(mean),
        rstd_(rstd),
        dgamma_(dgamma),
        dbeta_(dbeta) {}

 private:
  int64_t N_;
  int64_t C_;
  int64_t group_;
  const T* dY_;
  const T* X_;
  const T* mean_;
  const T* rstd_;
  T* dgamma_;
  T* dbeta_;
};

template <typename T, int SIMD, int kReduceTileSize>
struct GammaBeta1dBackwardLargeKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  using T_ACC = acc_type_device<T, kXPU>;

  SYCL_REQD_SUB_GROUP_SIZE(SIMD) void operator()(sycl::nd_item<2> item) const {
    const int64_t c =
        item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
    T_ACC dg_sum1 = 0;
    T_ACC dg_sum2 = 0;
    T_ACC db_sum1 = 0;
    T_ACC db_sum2 = 0;
    if (c < C_) {
      const int64_t G = group_;
      const int64_t D = C_ / G;
      // Accumulate each (subgroup_size) cols into a (subgroup_size^2) tile.
      for (int64_t n = item.get_local_id(0); n < N_;
           n += item.get_local_range(0) * 2) {
        const int64_t n1 = n;
        const int64_t n2 = n + item.get_local_range(0);
        const int64_t nc1 = n1 * C_ + c;
        const int64_t nc2 = n2 * C_ + c;
        const int64_t ng1 = n1 * G + c / D;
        const int64_t ng2 = n2 * G + c / D;
        const T_ACC dy1_acc = static_cast<T_ACC>(dY_[nc1]);
        const T_ACC x1_acc = static_cast<T_ACC>(X_[nc1]);
        dg_sum1 += dgamma_ == nullptr
            ? T_ACC(0)
            : ((dy1_acc * x1_acc - dy1_acc * static_cast<T_ACC>(mean_[ng1])) *
               static_cast<T_ACC>(rstd_[ng1]));
        db_sum1 += dbeta_ == nullptr ? T_ACC(0) : dy1_acc;
        if (n2 < N_) {
          const T_ACC dy2_acc = static_cast<T_ACC>(dY_[nc2]);
          const T_ACC x2_acc = static_cast<T_ACC>(X_[nc2]);
          dg_sum2 += dgamma_ == nullptr
              ? T_ACC(0)
              : ((dy2_acc * x2_acc - dy2_acc * static_cast<T_ACC>(mean_[ng2])) *
                 static_cast<T_ACC>(rstd_[ng2]));
          db_sum2 += dbeta_ == nullptr ? T_ACC(0) : dy2_acc;
        }
      }
    }

    // Write accumulated tile to shared memory.
    int tid_y = item.get_local_id(0);
    int tid_x = item.get_local_id(1);
    g_shared_[tid_y][tid_x] = dg_sum1;
    g_shared_[tid_y + item.get_local_range(0)][tid_x] = dg_sum2;
    b_shared_[tid_y][tid_x] = db_sum1;
    b_shared_[tid_y + item.get_local_range(0)][tid_x] = db_sum2;
    sycl::group_barrier(item.get_group());

    // Do subgroup reduce for the 1st 16 cols in the tile.
    T_ACC sum1 = g_shared_[tid_x][tid_y];
    T_ACC sum2 = b_shared_[tid_x][tid_y];
    sum1 = SubgroupReduceSumWithoutBroadcast<T_ACC, SIMD>(item, sum1);
    sum2 = SubgroupReduceSumWithoutBroadcast<T_ACC, SIMD>(item, sum2);
    if (tid_x == 0) {
      const int64_t c = item.get_group(1) * item.get_local_range(1) + tid_y;
      if (c < C_) {
        if (dgamma_ != nullptr) {
          dgamma_[c] = sum1;
        }
        if (dbeta_ != nullptr) {
          dbeta_[c] = sum2;
        }
      }
    }

    // Do subgroup reduce for the 2nd 16 cols in the tile.
    sum1 = g_shared_[tid_x][tid_y + item.get_local_range(0)];
    sum2 = b_shared_[tid_x][tid_y + item.get_local_range(0)];
    sum1 = SubgroupReduceSumWithoutBroadcast<T_ACC, SIMD>(item, sum1);
    sum2 = SubgroupReduceSumWithoutBroadcast<T_ACC, SIMD>(item, sum2);
    if (tid_x == 0) {
      const int64_t c = item.get_group(1) * item.get_local_range(1) + tid_y +
          item.get_local_range(0);
      if (c < C_) {
        if (dgamma_ != nullptr) {
          dgamma_[c] = sum1;
        }
        if (dbeta_ != nullptr) {
          dbeta_[c] = sum2;
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    g_shared_ = sycl_local_acc_t<T_ACC, 2>(
        sycl::range<2>(kReduceTileSize, kReduceTileSize + 1), cgh);
    b_shared_ = sycl_local_acc_t<T_ACC, 2>(
        sycl::range<2>(kReduceTileSize, kReduceTileSize + 1), cgh);
  }

  GammaBeta1dBackwardLargeKernel(
      int64_t N,
      int64_t C,
      int64_t group,
      const T* dY,
      const T* X,
      const T* mean,
      const T* rstd,
      T* dgamma,
      T* dbeta)
      : N_(N),
        C_(C),
        group_(group),
        dY_(dY),
        X_(X),
        mean_(mean),
        rstd_(rstd),
        dgamma_(dgamma),
        dbeta_(dbeta) {}

 private:
  int64_t N_;
  int64_t C_;
  int64_t group_;
  const T* dY_;
  const T* X_;
  const T* mean_;
  const T* rstd_;
  T* dgamma_;
  T* dbeta_;
  sycl_local_acc_t<T_ACC, 2> g_shared_;
  sycl_local_acc_t<T_ACC, 2> b_shared_;
};

template <typename T>
void group_norm_1d_backward(
    const Tensor dY,
    const Tensor X,
    const Tensor mean,
    const Tensor rstd,
    const Tensor gamma,
    int64_t N,
    int64_t C,
    int64_t group,
    Tensor& dX,
    Tensor& dgamma,
    Tensor& dbeta) {
  using T_ACC = acc_type_device<T, kXPU>;
  const int64_t G = group;
  const int64_t D = C / G;
  const T* dY_data = dY.const_data_ptr<T>();
  const T* X_data = X.const_data_ptr<T>();
  const T* mean_data = mean.const_data_ptr<T>();
  const T* rstd_data = rstd.const_data_ptr<T>();

  auto& queue = getCurrentSYCLQueue();
  int64_t simd = syclMaxSubGroupSize();

  if (dX.defined()) {
    const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
    const auto kAccType =
        (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
        ? kFloat
        : X.scalar_type();
    Tensor c2 = at::empty({N, G}, X.options().dtype(kAccType));
    Tensor c3 = at::empty({N, G}, X.options().dtype(kAccType));
    T_ACC* c2_data = c2.mutable_data_ptr<T_ACC>();
    T_ACC* c3_data = c3.mutable_data_ptr<T_ACC>();

    const int64_t wg_size = (C / G) < get_group_reduce_group_size(simd)
        ? simd
        : get_group_reduce_group_size(simd);
    auto global_range = sycl::range<2>(G, N * wg_size);
    auto local_range = sycl::range<2>(1, wg_size);
    group_norm_kernel_simd_choice_and_launch<
        Compute1dBackwardFusedParamsFunctor<T, T_ACC, SIMD16>,
        Compute1dBackwardFusedParamsFunctor<T, T_ACC, SIMD32>>(
        simd,
        global_range,
        local_range,
        queue,
        C,
        G,
        dY_data,
        X_data,
        mean_data,
        rstd_data,
        gamma_data,
        c2_data,
        c3_data);

    if (gamma.defined()) {
      auto iter = TensorIteratorConfig()
                      .check_all_same_dtype(std::is_same<T, T_ACC>::value)
                      .resize_outputs(false)
                      .add_owned_output(dX.view({N, G, D}))
                      .add_owned_const_input(dY.view({N, G, D}))
                      .add_owned_const_input(X.view({N, G, D}))
                      .add_owned_const_input(rstd.view({N, G, 1}))
                      .add_owned_const_input(gamma.view({1, G, D}))
                      .add_owned_const_input(c2.view({N, G, 1}))
                      .add_owned_const_input(c3.view({N, G, 1}))
                      .build();
      gpu_kernel(iter, GroupNorm1dBackwardGammaFunctor<T, T_ACC>());
    } else {
      auto iter = TensorIteratorConfig()
                      .check_all_same_dtype(std::is_same<T, T_ACC>::value)
                      .resize_outputs(false)
                      .add_owned_output(dX.view({N * G, D}))
                      .add_owned_const_input(dY.view({N * G, D}))
                      .add_owned_const_input(X.view({N * G, D}))
                      .add_owned_const_input(rstd.view({N * G, 1}))
                      .add_owned_const_input(c2.view({N * G, 1}))
                      .add_owned_const_input(c3.view({N * G, 1}))
                      .build();
      gpu_kernel(iter, GroupNorm1dBackwardFunctor<T, T_ACC>());
    }
  }
  if (dgamma.defined() || dbeta.defined()) {
    T* dgamma_data = dgamma.defined() ? dgamma.mutable_data_ptr<T>() : nullptr;
    T* dbeta_data = dbeta.defined() ? dbeta.mutable_data_ptr<T>() : nullptr;
    if (N <= 128) {
      const int64_t wg_size = get_group_reduce_group_size(simd);
      const int64_t B = (C + wg_size - 1) / wg_size;
      auto caller = GammaBeta1dBackwardSmallKernel<T>(
          N,
          C,
          G,
          dY_data,
          X_data,
          mean_data,
          rstd_data,
          dgamma_data,
          dbeta_data);
      sycl_kernel_submit(
          sycl::range<1>(B * wg_size), sycl::range<1>(wg_size), queue, caller);
    } else {
      // The algorithm for colwise reduction here is to accumulate each
      // (sub_group_size) cols to a (sub_group_size^2) tile and write the tile
      // to shared memory. Then do subgroup reduce for each col in the tile.
      const int64_t kReduceTileSize = simd;
      const int64_t B = (C + kReduceTileSize - 1) / kReduceTileSize;
      auto global_range =
          sycl::range<2>(kReduceTileSize / 2, B * kReduceTileSize);
      auto local_range = sycl::range<2>(kReduceTileSize / 2, kReduceTileSize);
      group_norm_kernel_simd_choice_and_launch<
          GammaBeta1dBackwardLargeKernel<T, SIMD16, SIMD16>,
          GammaBeta1dBackwardLargeKernel<T, SIMD32, SIMD32>>(
          simd,
          global_range,
          local_range,
          queue,
          N,
          C,
          G,
          dY_data,
          X_data,
          mean_data,
          rstd_data,
          dgamma_data,
          dbeta_data);
    }
  }
}

template <typename T, int SIMD>
struct ComputeInternalGradientsFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  using T_ACC = acc_type_device<T, kXPU>;

  SYCL_REQD_SUB_GROUP_SIZE(SIMD) void operator()(sycl::nd_item<1> item) const {
    const int64_t nc = item.get_group(0);
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t hw = item.get_local_id(0); hw < HxW_;
         hw += item.get_local_range(0)) {
      const int64_t index = nc * HxW_ + hw;
      sum1 += static_cast<T_ACC>(dY_[index]) * static_cast<T_ACC>(X_[index]);
      sum2 += static_cast<T_ACC>(dY_[index]);
    }
    sum1 = GroupReduceSumWithoutBroadcast<T_ACC, SIMD>(item, sum1, ds_shared_);
    sum2 = GroupReduceSumWithoutBroadcast<T_ACC, SIMD>(item, sum2, db_shared_);
    if (item.get_local_id(0) == 0) {
      ds_[nc] = sum1;
      db_[nc] = sum2;
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    ds_shared_ =
        sycl_local_acc_t<T_ACC>(get_group_reduce_group_size(SIMD), cgh);
    db_shared_ =
        sycl_local_acc_t<T_ACC>(get_group_reduce_group_size(SIMD), cgh);
  }

  ComputeInternalGradientsFunctor(
      int64_t HxW,
      const T* dY,
      const T* X,
      T_ACC* ds,
      T_ACC* db)
      : HxW_(HxW), dY_(dY), X_(X), ds_(ds), db_(db) {}

 private:
  int64_t HxW_;
  const T* dY_;
  const T* X_;
  T_ACC* ds_;
  T_ACC* db_;
  sycl_local_acc_t<T_ACC> ds_shared_;
  sycl_local_acc_t<T_ACC> db_shared_;
};

template <typename T, int SIMD, int VEC_SIZE>
struct ComputeInternalGradientsVectorizedFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  using T_ACC = acc_type_device<T, kXPU>;
  using vec_t = memory::aligned_vector<T, VEC_SIZE>;
  using acc_vec_t = memory::aligned_vector<T_ACC, VEC_SIZE>;

  SYCL_REQD_SUB_GROUP_SIZE(SIMD) void operator()(sycl::nd_item<1> item) const {
    acc_vec_t sum1_vec;
    acc_vec_t sum2_vec;

#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
      sum1_vec[v] = 0;
      sum2_vec[v] = 0;
    }

    auto group_start = item.get_group(0) * VEC_SIZE;

#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
      const int64_t nc = group_start + v;
      for (int64_t hw = item.get_local_id(0) * VEC_SIZE; hw < HxW_;
           hw += item.get_local_range(0) * VEC_SIZE) {
        const int64_t vec_index = nc * HxW_ + hw;
        vec_t vec_dY_ =
            *reinterpret_cast<vec_t*>(const_cast<T*>(dY_) + vec_index);
        vec_t vec_X_ =
            *reinterpret_cast<vec_t*>(const_cast<T*>(X_) + vec_index);

#pragma unroll
        for (int iv = 0; iv < VEC_SIZE; ++iv) {
          sum1_vec[v] += static_cast<T_ACC>(vec_dY_[iv] * vec_X_[iv]);
          sum2_vec[v] += static_cast<T_ACC>(vec_dY_[iv]);
        }
      }
    }

#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
      sum1_vec[v] = GroupReduceSumWithoutBroadcast<T_ACC, SIMD>(
          item, sum1_vec[v], ds_shared_);
      sum2_vec[v] = GroupReduceSumWithoutBroadcast<T_ACC, SIMD>(
          item, sum2_vec[v], db_shared_);
    }

    if (item.get_local_id(0) == 0) {
      acc_vec_t ds_vec;
      acc_vec_t db_vec;
#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v) {
        ds_vec[v] = sum1_vec[v];
        db_vec[v] = sum2_vec[v];
      }
      *(reinterpret_cast<acc_vec_t*>(ds_ + group_start)) = ds_vec;
      *(reinterpret_cast<acc_vec_t*>(db_ + group_start)) = db_vec;
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    ds_shared_ =
        sycl_local_acc_t<T_ACC>(get_group_reduce_group_size(SIMD), cgh);
    db_shared_ =
        sycl_local_acc_t<T_ACC>(get_group_reduce_group_size(SIMD), cgh);
  }

  ComputeInternalGradientsVectorizedFunctor(
      int64_t HxW,
      const T* dY,
      const T* X,
      T_ACC* ds,
      T_ACC* db)
      : HxW_(HxW), dY_(dY), X_(X), ds_(ds), db_(db) {}

 private:
  int64_t HxW_;
  const T* dY_;
  const T* X_;
  T_ACC* ds_;
  T_ACC* db_;
  sycl_local_acc_t<T_ACC> ds_shared_;
  sycl_local_acc_t<T_ACC> db_shared_;
};

template <typename T, typename T_ACC>
struct GroupNormBackwardC1Functor {
  T_ACC operator()(T rstd, T gamma) const {
    return static_cast<T_ACC>(rstd) * static_cast<T_ACC>(gamma);
  }
};

template <typename T, typename T_ACC>
struct GroupNormBackwardDXFunctor {
  T operator()(T dy, T x, T_ACC c1, T_ACC c2, T_ACC c3) const {
    return c1 * static_cast<T_ACC>(dy) + c2 * static_cast<T_ACC>(x) + c3;
  }
};

template <typename T, int SIMD>
struct ComputeBackwardFusedParamsFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  using T_ACC = acc_type_device<T, kXPU>;

  SYCL_REQD_SUB_GROUP_SIZE(SIMD) void operator()(sycl::nd_item<2> item) const {
    const int64_t G = group_;
    const int64_t D = C_ / G;
    const int64_t n = item.get_group(1);
    const int64_t g = item.get_group(0);
    const int64_t ng = n * G + g;
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t i = item.get_local_id(1); i < D;
         i += item.get_local_range(1)) {
      const int64_t index = ng * D + i;
      const int64_t c = g * D + i;
      const T_ACC gamma_v =
          gamma_ == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma_[c]);
      sum1 += ds_[index] * gamma_v;
      sum2 += db_[index] * gamma_v;
    }
    sum1 = GroupReduceSumWithoutBroadcast<T_ACC, SIMD>(item, sum1, ds_shared_);
    sum2 = GroupReduceSumWithoutBroadcast<T_ACC, SIMD>(item, sum2, db_shared_);
    if (item.get_local_id(1) == 0) {
      const T_ACC s = T_ACC(1) / static_cast<T_ACC>(D * HxW_);
      const T_ACC x = (sum2 * static_cast<T_ACC>(mean_[ng]) - sum1) *
          static_cast<T_ACC>(rstd_[ng]) * static_cast<T_ACC>(rstd_[ng]) *
          static_cast<T_ACC>(rstd_[ng]) * s;
      c2_[ng] = x;
      c3_[ng] = -x * static_cast<T_ACC>(mean_[ng]) -
          sum2 * static_cast<T_ACC>(rstd_[ng]) * s;
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    ds_shared_ =
        sycl_local_acc_t<T_ACC>(get_group_reduce_group_size(SIMD), cgh);
    db_shared_ =
        sycl_local_acc_t<T_ACC>(get_group_reduce_group_size(SIMD), cgh);
  }

  ComputeBackwardFusedParamsFunctor(
      int64_t C,
      int64_t HxW,
      int64_t group,
      const T* mean,
      const T* rstd,
      const T* gamma,
      const T_ACC* ds,
      const T_ACC* db,
      T_ACC* c2,
      T_ACC* c3)
      : C_(C),
        HxW_(HxW),
        group_(group),
        mean_(mean),
        rstd_(rstd),
        gamma_(gamma),
        ds_(ds),
        db_(db),
        c2_(c2),
        c3_(c3) {}

 private:
  int64_t C_;
  int64_t HxW_;
  int64_t group_;
  const T* mean_;
  const T* rstd_;
  const T* gamma_;
  const T_ACC* ds_;
  const T_ACC* db_;
  T_ACC* c2_;
  T_ACC* c3_;
  sycl_local_acc_t<T_ACC> ds_shared_;
  sycl_local_acc_t<T_ACC> db_shared_;
};

template <typename T>
struct GammaBetaBackwardPlainFunctor {
  using T_ACC = acc_type_device<T, kXPU>;

  void operator()(sycl::item<1> item) const {
    auto c = item.get_id(0);
    auto G = group_;
    auto D = C_ / G;
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t n = 0; n < N_; ++n) {
      auto nc = n * C_ + c;
      auto ng = n * G + c / D;
      sum1 += (dgamma_ == nullptr)
          ? T_ACC(0)
          : ((ds_[nc] - db_[nc] * static_cast<T_ACC>(mean_[ng])) *
             static_cast<T_ACC>(rstd_[ng]));
      sum2 += (dbeta_ == nullptr) ? T_ACC(0) : db_[nc];
    }
    if (dgamma_ != nullptr) {
      dgamma_[c] = sum1;
    }
    if (dbeta_ != nullptr) {
      dbeta_[c] = sum2;
    }
  }

  GammaBetaBackwardPlainFunctor(
      int64_t N,
      int64_t C,
      int64_t group,
      const T* mean,
      const T* rstd,
      const T_ACC* ds,
      const T_ACC* db,
      T* dgamma,
      T* dbeta)
      : N_(N),
        C_(C),
        group_(group),
        mean_(mean),
        rstd_(rstd),
        ds_(ds),
        db_(db),
        dgamma_(dgamma),
        dbeta_(dbeta) {}

 private:
  int64_t N_;
  int64_t C_;
  int64_t group_;
  const T* mean_;
  const T* rstd_;
  const T_ACC* ds_;
  const T_ACC* db_;
  T* dgamma_;
  T* dbeta_;
};

template <typename T, int SIMD, int kReduceTileSize>
struct GammaBetaBackwardFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  using T_ACC = acc_type_device<T, kXPU>;

  SYCL_REQD_SUB_GROUP_SIZE(SIMD) void operator()(sycl::nd_item<2> item) const {
    auto group_x = item.get_group(1);
    auto group_size_x = item.get_local_range(1);
    auto group_size_y = item.get_local_range(0);
    auto tid_x = item.get_local_id(1);
    auto tid_y = item.get_local_id(0);

    const int64_t c = group_x * group_size_x + tid_x;
    T_ACC dg_sum1 = 0;
    T_ACC dg_sum2 = 0;
    T_ACC db_sum1 = 0;
    T_ACC db_sum2 = 0;
    if (c < C_) {
      const int64_t G = group_;
      const int64_t D = C_ / G;
      // Accumulate each 32 cols into a 32 * 32 tile.
      // Since the group size is (32, 16), accumulate twice for 1st and 2nd 16
      // rows of a 32 contiguous elements.
      for (int64_t n = tid_y; n < N_; n += group_size_y * 2) {
        const int64_t n1 = n;
        const int64_t n2 = n + group_size_y;
        const int64_t nc1 = n1 * C_ + c;
        const int64_t nc2 = n2 * C_ + c;
        const int64_t ng1 = n1 * G + c / D;
        const int64_t ng2 = n2 * G + c / D;
        dg_sum1 += dgamma_ == nullptr
            ? T_ACC(0)
            : ((ds_[nc1] - db_[nc1] * static_cast<T_ACC>(mean_[ng1])) *
               static_cast<T_ACC>(rstd_[ng1]));
        db_sum1 += dbeta_ == nullptr ? T_ACC(0) : db_[nc1];
        if (n2 < N_) {
          dg_sum2 += dgamma_ == nullptr
              ? T_ACC(0)
              : ((ds_[nc2] - db_[nc2] * static_cast<T_ACC>(mean_[ng2])) *
                 static_cast<T_ACC>(rstd_[ng2]));
          db_sum2 += dbeta_ == nullptr ? T_ACC(0) : db_[nc2];
        }
      }
    }

    // Write accumulated tile to shared memory.
    g_shared_[tid_y][tid_x] = dg_sum1;
    g_shared_[tid_y + group_size_y][tid_x] = dg_sum2;
    b_shared_[tid_y][tid_x] = db_sum1;
    b_shared_[tid_y + group_size_y][tid_x] = db_sum2;
    sycl::group_barrier(item.get_group());

    // Do subgroup reduce for the 1st 16 cols in the tile.
    T_ACC sum1 = g_shared_[tid_x][tid_y];
    T_ACC sum2 = b_shared_[tid_x][tid_y];
    sum1 = SubgroupReduceSumWithoutBroadcast<T_ACC, SIMD>(item, sum1);
    sum2 = SubgroupReduceSumWithoutBroadcast<T_ACC, SIMD>(item, sum2);
    if (tid_x == 0) {
      const int64_t c = group_x * group_size_x + tid_y;
      if (c < C_) {
        if (dgamma_ != nullptr) {
          dgamma_[c] = sum1;
        }
        if (dbeta_ != nullptr) {
          dbeta_[c] = sum2;
        }
      }
    }

    // Do subgroup reduce for the 2st 16 cols in the tile.
    sum1 = g_shared_[tid_x][tid_y + group_size_y];
    sum2 = b_shared_[tid_x][tid_y + group_size_y];
    sum1 = SubgroupReduceSumWithoutBroadcast<T_ACC, SIMD>(item, sum1);
    sum2 = SubgroupReduceSumWithoutBroadcast<T_ACC, SIMD>(item, sum2);
    if (tid_x == 0) {
      const int64_t c = group_x * group_size_x + tid_y + group_size_y;
      if (c < C_) {
        if (dgamma_ != nullptr) {
          dgamma_[c] = sum1;
        }
        if (dbeta_ != nullptr) {
          dbeta_[c] = sum2;
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    g_shared_ = sycl_local_acc_t<T_ACC, 2>(
        sycl::range<2>(kReduceTileSize, kReduceTileSize + 1), cgh);
    b_shared_ = sycl_local_acc_t<T_ACC, 2>(
        sycl::range<2>(kReduceTileSize, kReduceTileSize + 1), cgh);
  }

  GammaBetaBackwardFunctor(
      int64_t N,
      int64_t C,
      int64_t group,
      const T* mean,
      const T* rstd,
      const T_ACC* ds,
      const T_ACC* db,
      T* dgamma,
      T* dbeta)
      : N_(N),
        C_(C),
        group_(group),
        mean_(mean),
        rstd_(rstd),
        ds_(ds),
        db_(db),
        dgamma_(dgamma),
        dbeta_(dbeta) {}

 private:
  int64_t N_;
  int64_t C_;
  int64_t group_;
  const T* mean_;
  const T* rstd_;
  const T_ACC* ds_;
  const T_ACC* db_;
  T* dgamma_;
  T* dbeta_;
  sycl_local_acc_t<T_ACC, 2> g_shared_;
  sycl_local_acc_t<T_ACC, 2> b_shared_;
};

template <typename T>
void group_norm_backward_kernel_impl(
    const Tensor& dY_,
    const Tensor& X_,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    Tensor& dX,
    Tensor& dgamma,
    Tensor& dbeta) {
  auto dY = dY_.contiguous();
  auto X = X_.contiguous();

  using T_ACC = acc_type_device<T, kXPU>;
  const int64_t G = group;
  const int64_t D = C / G;
  TORCH_CHECK(dY.numel() == N * C * HxW);
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(mean.numel() == N * G);
  TORCH_CHECK(rstd.numel() == N * G);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);

  if (N == 0) {
    if (dgamma.defined()) {
      dgamma.fill_(T(0));
    }
    if (dbeta.defined()) {
      dbeta.fill_(T(0));
    }
    return;
  }

  const T* dY_data = dY.const_data_ptr<T>();
  const T* X_data = X.const_data_ptr<T>();
  const T* mean_data = mean.const_data_ptr<T>();
  const T* rstd_data = rstd.const_data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
  const auto kAccType =
      (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
      ? kFloat
      : X.scalar_type();
  Tensor ds = at::empty({N, C}, X.options().dtype(kAccType));
  Tensor db = at::empty({N, C}, X.options().dtype(kAccType));
  T_ACC* ds_data = ds.mutable_data_ptr<T_ACC>();
  T_ACC* db_data = db.mutable_data_ptr<T_ACC>();

  if (HxW == 1) {
    group_norm_1d_backward<T>(
        dY, X, mean, rstd, gamma, N, C, G, dX, dgamma, dbeta);
    return;
  }

  auto& queue = getCurrentSYCLQueue();
  int64_t simd = syclMaxSubGroupSize();

  constexpr int VEC_SIZE = PREFERRED_VEC_SIZE;
  int64_t wg_size = 0;

  if (can_use_vectorization(dY_data, VEC_SIZE) &&
      can_use_vectorization(X_data, VEC_SIZE) &&
      can_use_vectorization(ds_data, VEC_SIZE) &&
      can_use_vectorization(db_data, VEC_SIZE) && HxW % VEC_SIZE == 0 &&
      (N * C) % VEC_SIZE == 0) {
    using KernelS16T =
        ComputeInternalGradientsVectorizedFunctor<T, SIMD16, VEC_SIZE>;
    using KernelS32T =
        ComputeInternalGradientsVectorizedFunctor<T, SIMD32, VEC_SIZE>;
    wg_size = (HxW / VEC_SIZE) < get_group_reduce_group_size(simd)
        ? simd
        : get_group_reduce_group_size(simd);
    group_norm_kernel_simd_choice_and_launch<KernelS16T, KernelS32T>(
        simd,
        sycl::range<1>((N * C / VEC_SIZE) * wg_size),
        sycl::range<1>(wg_size),
        queue,
        HxW,
        dY_data,
        X_data,
        ds_data,
        db_data);
  } else {
    using KernelS16T = ComputeInternalGradientsFunctor<T, SIMD16>;
    using KernelS32T = ComputeInternalGradientsFunctor<T, SIMD32>;
    wg_size = HxW < get_group_reduce_group_size(simd)
        ? simd
        : get_group_reduce_group_size(simd);
    group_norm_kernel_simd_choice_and_launch<KernelS16T, KernelS32T>(
        simd,
        sycl::range<1>(N * C * wg_size),
        sycl::range<1>(wg_size),
        queue,
        HxW,
        dY_data,
        X_data,
        ds_data,
        db_data);
  }

  if (dX.defined()) {
    Tensor c1 = at::empty({0}, X.options().dtype(kAccType));
    Tensor c2 = at::empty({N, G}, X.options().dtype(kAccType));
    Tensor c3 = at::empty({N, G}, X.options().dtype(kAccType));
    T_ACC* c2_data = c2.mutable_data_ptr<T_ACC>();
    T_ACC* c3_data = c3.mutable_data_ptr<T_ACC>();

    if (gamma.defined()) {
      auto iter = TensorIteratorConfig()
                      .check_all_same_dtype(std::is_same<T, T_ACC>::value)
                      .add_output(c1)
                      .add_owned_const_input(rstd.view({N, G, 1}))
                      .add_owned_const_input(gamma.view({1, G, D}))
                      .build();
      gpu_kernel(iter, GroupNormBackwardC1Functor<T, T_ACC>());
    }

    wg_size = (C / G) < get_group_reduce_group_size(simd)
        ? simd
        : get_group_reduce_group_size(simd);
    group_norm_kernel_simd_choice_and_launch<
        ComputeBackwardFusedParamsFunctor<T, SIMD16>,
        ComputeBackwardFusedParamsFunctor<T, SIMD32>>(
        simd,
        sycl::range<2>(G, N * wg_size),
        sycl::range<2>(1, wg_size),
        queue,
        C,
        HxW,
        G,
        mean_data,
        rstd_data,
        gamma_data,
        ds_data,
        db_data,
        c2_data,
        c3_data);

    if (gamma.defined()) {
      auto iter = TensorIteratorConfig()
                      .check_all_same_dtype(std::is_same<T, T_ACC>::value)
                      .resize_outputs(false)
                      .add_owned_output(dX.view({N * G, D, HxW}))
                      .add_owned_const_input(dY.view({N * G, D, HxW}))
                      .add_owned_const_input(X.view({N * G, D, HxW}))
                      .add_owned_const_input(c1.view({N * G, D, 1}))
                      .add_owned_const_input(c2.view({N * G, 1, 1}))
                      .add_owned_const_input(c3.view({N * G, 1, 1}))
                      .build();
      gpu_kernel(iter, GroupNormBackwardDXFunctor<T, T_ACC>());
    } else {
      auto iter = TensorIteratorConfig()
                      .check_all_same_dtype(std::is_same<T, T_ACC>::value)
                      .resize_outputs(false)
                      .add_owned_output(dX.view({N * G, D * HxW}))
                      .add_owned_const_input(dY.view({N * G, D * HxW}))
                      .add_owned_const_input(X.view({N * G, D * HxW}))
                      .add_owned_const_input(rstd.view({N * G, 1}))
                      .add_owned_const_input(c2.view({N * G, 1}))
                      .add_owned_const_input(c3.view({N * G, 1}))
                      .build();
      gpu_kernel(iter, GroupNormBackwardDXFunctor<T, T_ACC>());
    }
  }

  if (dgamma.defined() || dbeta.defined()) {
    T* dgamma_data = dgamma.defined() ? dgamma.mutable_data_ptr<T>() : nullptr;
    T* dbeta_data = dbeta.defined() ? dbeta.mutable_data_ptr<T>() : nullptr;
    if (N <= 128) {
      // For small batch size, do colwise reduce directly.
      auto caller = GammaBetaBackwardPlainFunctor<T>(
          N,
          C,
          G,
          mean_data,
          rstd_data,
          ds_data,
          db_data,
          dgamma_data,
          dbeta_data);
      sycl_kernel_submit(sycl::range<1>(C), queue, caller);
    } else {
      // The algorithm for colwise reduction here is to accumulate each
      // (subgroup_size) cols to a (subgroup_size^2) tile and write the tile
      // to shared memory. Then do subgroup reduce for each col in the tile.
      const int64_t kReduceTileSize = simd;
      const int64_t B = (C + kReduceTileSize - 1) / kReduceTileSize;
      auto global_range =
          sycl::range<2>(kReduceTileSize / 2, B * kReduceTileSize);
      auto local_range = sycl::range<2>(kReduceTileSize / 2, kReduceTileSize);
      group_norm_kernel_simd_choice_and_launch<
          GammaBetaBackwardFunctor<T, SIMD16, SIMD16>,
          GammaBetaBackwardFunctor<T, SIMD32, SIMD32>>(
          simd,
          global_range,
          local_range,
          queue,
          N,
          C,
          G,
          mean_data,
          rstd_data,
          ds_data,
          db_data,
          dgamma_data,
          dbeta_data);
    }
  }
}

void group_norm_backward_kernel(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    Tensor& dX,
    Tensor& dgamma,
    Tensor& dbeta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "group_norm_backward_xpu",
      [&]() {
        group_norm_backward_kernel_impl<scalar_t>(
            dY, X, mean, rstd, gamma, N, C, HxW, group, dX, dgamma, dbeta);
      });
}

} // namespace at::native::xpu
