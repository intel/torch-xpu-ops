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
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/GroupReduceUtils.h>
#include <ATen/xpu/XPUContext.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/Norm.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/LayerNormKernels.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t, typename mean_t, typename weight_t, bool rms_norm>
class LayerNormBackward
    : public NormBackward<scalar_t, mean_t, weight_t, rms_norm> {
 public:
  using accscalar_t = acc_type_device<scalar_t, kXPU>;
  LayerNormBackward() = delete;
  LayerNormBackward(
      const scalar_t* X_data,
      const scalar_t* dY_data,
      scalar_t* dX_data,
      const mean_t* mean_data,
      const mean_t* var_data,
      const weight_t* gamma_data,
      int64_t M,
      int64_t N)
      : NormBackward<scalar_t, mean_t, weight_t, rms_norm>(
            X_data,
            dY_data,
            dX_data,
            mean_data,
            var_data,
            gamma_data,
            nullptr,
            nullptr),
        M(M),
        N(N) {}

  LayerNormBackward(
      const scalar_t* X_data,
      const scalar_t* dY_data,
      scalar_t* dX_data,
      const mean_t* mean_data,
      const mean_t* var_data,
      const weight_t* gamma_data,
      accscalar_t* a_data,
      accscalar_t* b_data,
      int64_t M,
      int64_t N)
      : NormBackward<scalar_t, mean_t, weight_t, rms_norm>(
            X_data,
            dY_data,
            dX_data,
            mean_data,
            var_data,
            gamma_data,
            a_data,
            b_data),
        M(M),
        N(N) {}
  using NB = NormBackward<scalar_t, mean_t, weight_t, rms_norm>;

  template <
      int vec_size,
      typename vec_t,
      typename weight_vec_t,
      typename index_t,
      typename nd_item_id>
  void reduce_combine(
      nd_item_id item_id,
      const NormConfig& cfg,
      accscalar_t& sum1,
      accscalar_t& sum2) const {
    auto group_id = item_id.get_group(0);
    auto group_id_foreach = item_id.get_group(1);
    auto local_id = item_id.get_local_id(2);
    index_t group_offset = group_id * cfg.problem_size;

    mean_t mean_val = NB::mean_data[group_id];
    mean_t rstd_val = NB::var_data[group_id];
    for (index_t j = local_id * vec_size; j < cfg.workgroup_work_size;
         j += cfg.workgroup_size * vec_size) {
      index_t plane_offset = group_id_foreach * cfg.workgroup_work_size + j;
      if (plane_offset < cfg.problem_size) {
        weight_vec_t gamma_val;
        if (NB::gamma_data != nullptr) {
          gamma_val = *(reinterpret_cast<const weight_vec_t*>(
              NB::gamma_data + plane_offset));
        }
        vec_t dY_val = *(reinterpret_cast<const vec_t*>(
            NB::dY_data + group_offset + plane_offset));
        vec_t X_val = *(reinterpret_cast<const vec_t*>(
            NB::X_data + group_offset + plane_offset));
        for (int v = 0; v < vec_size; ++v) {
          accscalar_t value = (NB::gamma_data == nullptr)
              ? static_cast<accscalar_t>(dY_val[v])
              : (static_cast<accscalar_t>(dY_val[v]) *
                 static_cast<accscalar_t>(gamma_val[v]));
          if constexpr (!rms_norm) {
            sum1 += value;
            sum2 += value * static_cast<accscalar_t>(X_val[v] - mean_val) *
                rstd_val;
          } else {
            sum2 += value * static_cast<accscalar_t>(X_val[v]) * rstd_val;
          }
        }
      }
    }
  };

  template <
      int vec_size,
      typename index_t,
      typename vec_t,
      typename weight_vec_t,
      typename nd_item_id>
  void update(
      nd_item_id item_id,
      const NormConfig& cfg,
      accscalar_t sum1 = 0,
      accscalar_t sum2 = 0) const {
    auto local_id = item_id.get_local_id(2);
    auto group_id_foreach = item_id.get_group(1);
    auto group_id = item_id.get_group(0);
    if (cfg.workgroup_num_foreach > 1) {
      if constexpr (!rms_norm) {
        sum1 = NB::a_data[group_id];
      }
      sum2 = NB::b_data[group_id];
    }

    index_t group_offset = group_id * cfg.problem_size;
    mean_t mean_val = NB::mean_data[group_id];
    mean_t var_val = NB::var_data[group_id];

    int fH = cfg.problem_size;
    accscalar_t term1 = (accscalar_t(1) / fH) * var_val;
    for (index_t j = local_id * vec_size; j < cfg.workgroup_work_size;
         j += cfg.workgroup_size * vec_size) {
      index_t plane_offset = group_id_foreach * cfg.workgroup_work_size + j;
      if (plane_offset < (index_t)cfg.problem_size) {
        vec_t dY_val = *(reinterpret_cast<const vec_t*>(
            NB::dY_data + group_offset + plane_offset));
        vec_t X_val = *(reinterpret_cast<const vec_t*>(
            NB::X_data + group_offset + plane_offset));
        weight_vec_t gamma_val;
        if (NB::gamma_data != nullptr) {
          gamma_val = *(reinterpret_cast<const weight_vec_t*>(
              NB::gamma_data + plane_offset));
        }

        vec_t dX_val;
        for (int v = 0; v < vec_size; ++v) {
          accscalar_t f_grad_input = (NB::gamma_data == nullptr)
              ? static_cast<accscalar_t>(fH * dY_val[v])
              : static_cast<accscalar_t>(fH * gamma_val[v] * dY_val[v]);
          if constexpr (!rms_norm) {
            f_grad_input -= (X_val[v] - mean_val) * var_val * sum2;
            f_grad_input -= sum1;
          } else {
            f_grad_input -= X_val[v] * var_val * sum2;
          }
          dX_val[v] = static_cast<scalar_t>(f_grad_input * term1);
        }
        *(reinterpret_cast<vec_t*>(NB::dX_data + group_offset + plane_offset)) =
            dX_val;
      }
    }
  };

  int64_t M;
  int64_t N;
};

// we could make it dependent on dtype, but that would lead to different results
// between float and low-p types
constexpr int vec_size = 4;

// Checks alignment of buffers for using vectorized loads / stores
template <typename T>
bool can_vectorize(const T* ptr, int alignment) {
  uint64_t addr = reinterpret_cast<uint64_t>(ptr);
  return addr % alignment == 0;
};

template <typename T, typename T_ACC, bool rms_norm>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::sub_group_size<SIMD>))
void row_wise_moments_kernel(
    int64_t N,
    T_ACC eps,
    const T* X,
    T_ACC* mean,
    T_ACC* rstd) {
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int64_t, std::pair<T_ACC, T_ACC>>;

  auto item_id = syclext::this_work_item::get_nd_item<1>();
  WelfordType* shared =
      reinterpret_cast<WelfordType*>(syclexp::get_work_group_scratch_memory());
  const int64_t i = item_id.get_group(0);
  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);
  for (int64_t j = item_id.get_local_id(0); j < N;
       j += item_id.get_local_range(0)) {
    const int64_t index = i * N + j;
    val = welford_op.reduce(val, static_cast<T_ACC>(X[index]), index);
  }

  val = GroupReduceWithoutBroadcast<WelfordType, WelfordOp, SIMD>(
      item_id, val, welford_op, shared);

  if (item_id.get_local_id(0) == 0) {
    auto [m2, m1] = welford_op.project(val);
    if constexpr (!rms_norm) {
      mean[i] = m1;
      rstd[i] = c10::xpu::compat::rsqrt(m2 + eps);
    } else {
      rstd[i] = c10::xpu::compat::rsqrt(m2 + m1 * m1 + eps);
    }
  }
}

template <typename T, typename T_ACC, bool rms_norm>
void launch_rowwise_moments_kernel(
    int64_t N,
    int64_t M,
    T_ACC eps,
    const T* X_data,
    T_ACC* mean_data,
    T_ACC* rstd_data) {
  using WelfordType = WelfordData<T_ACC, int64_t>;

  int64_t sg_size = SIMD;
  int64_t wg_size = get_group_reduce_group_size(sg_size);
  sycl::range<1> local_range{size_t(wg_size)};
  sycl::range<1> global_range{size_t(M * wg_size)};
  auto queue = getCurrentSYCLQueue();

  int slm_sz = sizeof(WelfordType) * SIMD;
  sycl_kernel_submit<row_wise_moments_kernel<T, T_ACC, rms_norm>>(
      global_range,
      local_range,
      queue,
      slm_sz,
      N,
      eps,
      X_data,
      mean_data,
      rstd_data);
}

template <typename T, typename T_ACC, bool rms_norm>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void layer_norm_forward_kernel(
    int64_t N,
    const T* X,
    const T_ACC* mean,
    const T_ACC* rstd,
    const T* gamma,
    const T* beta,
    T* Y) {
  auto item_id = syclext::this_work_item::get_nd_item<1>();
  const int64_t i = item_id.get_group(0);
  for (int64_t j = item_id.get_local_id(0); j < N;
       j += item_id.get_local_range(0)) {
    const int64_t index = i * N + j;
    const T_ACC gamma_v =
        gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[j]);
    if constexpr (!rms_norm) {
      const T_ACC beta_v =
          beta == nullptr ? T_ACC(0) : static_cast<T_ACC>(beta[j]);
      Y[index] = (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i])) *
              static_cast<T_ACC>(rstd[i]) * gamma_v +
          beta_v;
    } else {
      Y[index] = (static_cast<T_ACC>(X[index])) * static_cast<T_ACC>(rstd[i]) *
          gamma_v;
    }
  }
}

template <typename T, typename T_ACC, bool rms_norm>
void launch_layer_norm_forward_kernel(
    int64_t N,
    int64_t M,
    const T* X_data,
    const T_ACC* mean_data,
    const T_ACC* rstd_data,
    const T* gamma_data,
    const T* beta_data,
    T* Y_data) {
  int64_t sg_size = SIMD;
  int64_t wg_size = get_group_reduce_group_size(sg_size);
  sycl::range<1> local_range{size_t(wg_size)};
  sycl::range<1> global_range(M * size_t(wg_size));
  auto queue = getCurrentSYCLQueue();

  sycl_kernel_submit<layer_norm_forward_kernel<T, T_ACC, rms_norm>>(
      global_range,
      local_range,
      queue,
      0,
      N,
      X_data,
      mean_data,
      rstd_data,
      gamma_data,
      beta_data,
      Y_data);
}

struct WelfordDataLN {
  float mean;
  float sigma2;
  float count;
  WelfordDataLN() : mean(0.f), sigma2(0.f), count(0.f) {}
  WelfordDataLN(float mean, float sigma2, float count)
      : mean(mean), sigma2(sigma2), count(count) {}
};

template <typename U, bool rms_norm>
WelfordDataLN WelfordOnlineSum(const U val, const WelfordDataLN& curr_sum) {
  if constexpr (!rms_norm) {
    U delta = val - curr_sum.mean;
    U new_count = curr_sum.count + 1.f;
    // proper division is slow, this is less accurate but noticeably faster
    U new_mean = curr_sum.mean + delta * sycl::native::recip(new_count);
    return {
        static_cast<float>(new_mean),
        static_cast<float>(curr_sum.sigma2 + delta * (val - new_mean)),
        static_cast<float>(new_count)};
  } else {
    return {0.f, static_cast<float>(curr_sum.sigma2 + val * val), 0.f};
  }
}

template <bool rms_norm>
WelfordDataLN WelfordCombine(
    const WelfordDataLN dataB,
    const WelfordDataLN dataA) {
  if constexpr (!rms_norm) {
    using U = decltype(dataB.count);
    U delta = dataB.mean - dataA.mean;
    U count = dataA.count + dataB.count;
    U mean, sigma2;
    if (count > decltype(dataB.count){0}) {
      auto coef = sycl::native::recip(count);
      auto nA = dataA.count * coef;
      auto nB = dataB.count * coef;
      mean = nA * dataA.mean + nB * dataB.mean;
      sigma2 = dataA.sigma2 + dataB.sigma2 + delta * delta * dataA.count * nB;
    } else {
      mean = U(0);
      sigma2 = U(0);
    }
    return {mean, sigma2, count};
  } else {
    return {0.f, dataB.sigma2 + dataA.sigma2, 0.f};
  }
}

template <typename T, typename T_ACC, bool rms_norm>
WelfordDataLN compute_stats(
    const T* RESTRICT X,
    const int N,
    T_ACC* buf,
    sycl::nd_item<2>& item_id) {
  // X points to the row to read
  using vec_t = aligned_vector<T, vec_size>;
  using acc_t = acc_type_device<T, kXPU>;
  const vec_t* X_vec = reinterpret_cast<const vec_t*>(X);
  const int numx = item_id.get_local_range(1) * item_id.get_local_range(0);
  const int thrx = item_id.get_local_linear_id();
  const int n_vec_to_read = N / vec_size;
  WelfordDataLN wd(0.f, 0.f, 0.f);
  // no tail, we check that N is multiple of vec_size
  for (int i = thrx; i < n_vec_to_read; i += numx) {
    vec_t data = X_vec[i];
#pragma unroll
    for (int ii = 0; ii < vec_size; ii++) {
      wd = WelfordOnlineSum<acc_t, rms_norm>(
          static_cast<acc_t>(data.val[ii]), wd);
    }
  }
  // intra-warp reduction
  auto sg = item_id.get_sub_group();
  for (int offset = (SIMD >> 1); offset > 0; offset >>= 1) {
    WelfordDataLN wdB{
        sycl::shift_group_left(sg, wd.mean, offset),
        sycl::shift_group_left(sg, wd.sigma2, offset),
        sycl::shift_group_left(sg, wd.count, offset)};
    wd = WelfordCombine<rms_norm>(wd, wdB);
  }

  // threadIdx.x == 0 has correct values for each warp
  // inter-warp reductions
  if (item_id.get_local_range(0) > 1) {
    auto addr_offset = item_id.get_local_range(0);
    for (int offset = item_id.get_local_range(0) / 2; offset > 0; offset /= 2) {
      // upper half of warps write to shared
      if (item_id.get_local_id(1) == 0 && item_id.get_local_id(0) >= offset &&
          item_id.get_local_id(0) < 2 * offset) {
        const int wrt_y = item_id.get_local_id(0) - offset;
        buf[2 * wrt_y] = wd.mean;
        buf[2 * wrt_y + 1] = wd.sigma2;
        buf[wrt_y + addr_offset] = wd.count;
      }
      sycl::group_barrier(item_id.get_group());

      // lower half merges
      if (item_id.get_local_id(1) == 0 && item_id.get_local_id(0) < offset) {
        const int rd_y = item_id.get_local_id(0);
        WelfordDataLN wdB{
            static_cast<float>(buf[2 * rd_y]),
            static_cast<float>(buf[2 * rd_y + 1]),
            static_cast<float>(buf[rd_y + addr_offset])};
        wd = WelfordCombine<rms_norm>(wd, wdB);
      }
      sycl::group_barrier(item_id.get_group());
    }

    if (item_id.get_local_id(1) == 0 && item_id.get_local_id(0) == 0) {
      buf[0] = wd.mean;
      buf[1] = wd.sigma2 / float(N);
    }
    sycl::group_barrier(item_id.get_group());
    return WelfordDataLN{
        static_cast<float>(buf[0]), static_cast<float>(buf[1]), 0.f};
  } else {
    return WelfordDataLN{
        sycl::select_from_group(sg, wd.mean, 0),
        sycl::select_from_group(sg, wd.sigma2, 0) / float(N),
        0.f};
  }
}

template <typename T, typename T_ACC, bool rms_norm>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::sub_group_size<SIMD>))
void vectorized_layer_norm_kernel(
    const int N,
    T_ACC eps,
    const T* RESTRICT X,
    const T* gamma,
    const T* beta,
    T_ACC* mean,
    T_ACC* rstd,
    T* Y,
    int64_t wg_size) {
  auto item_id = syclext::this_work_item::get_nd_item<2>();
  T_ACC* buf =
      reinterpret_cast<T_ACC*>(syclexp::get_work_group_scratch_memory());

  auto i1 = item_id.get_group(1);
  const T* block_row = X + i1 * N;
  WelfordDataLN wd =
      compute_stats<T, T_ACC, rms_norm>(block_row, N, buf, item_id);

  using vec_t = aligned_vector<T, vec_size>;
  const vec_t* X_vec = reinterpret_cast<const vec_t*>(block_row);
  const vec_t* gamma_vec =
      (gamma != nullptr) ? reinterpret_cast<const vec_t*>(gamma) : nullptr;
  const vec_t* beta_vec =
      (beta != nullptr) ? reinterpret_cast<const vec_t*>(beta) : nullptr;
  vec_t* Y_vec = reinterpret_cast<vec_t*>(Y + i1 * N);

  const int numx = item_id.get_local_range(1) * item_id.get_local_range(0);
  const int thrx = item_id.get_local_linear_id();
  const int n_vec_to_read = N / vec_size;

  T_ACC rstd_val = c10::xpu::compat::rsqrt(wd.sigma2 + eps);

  // No tail, N is guaranteed to be multiple of vec size
  for (int i = thrx; i < n_vec_to_read; i += numx) {
    vec_t data = X_vec[i];
    vec_t out;

    // Computation is performed in T_ACC, X is cast to T_ACC and result is
    // implicitly cast to T
    if (gamma_vec != nullptr && beta_vec != nullptr) {
      vec_t gamma_data = gamma_vec[i];
      if constexpr (!rms_norm) {
        vec_t beta_data = beta_vec[i];
#pragma unroll
        for (int ii = 0; ii < vec_size; ii++) {
          out.val[ii] = static_cast<T_ACC>(gamma_data.val[ii]) *
                  (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean)) +
              static_cast<T_ACC>(beta_data.val[ii]);
        }
      } else {
#pragma unroll
        for (int ii = 0; ii < vec_size; ii++) {
          out.val[ii] = static_cast<T_ACC>(gamma_data.val[ii]) *
              (rstd_val * static_cast<T_ACC>(data.val[ii]));
        }
      }
    } else if (gamma_vec != nullptr) {
      vec_t gamma_data = gamma_vec[i];
#pragma unroll
      for (int ii = 0; ii < vec_size; ii++) {
        if constexpr (!rms_norm) {
          out.val[ii] = static_cast<T_ACC>(gamma_data.val[ii]) *
              (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean));
        } else {
          out.val[ii] = static_cast<T_ACC>(gamma_data.val[ii]) *
              (rstd_val * static_cast<T_ACC>(data.val[ii]));
        }
      }
    } else if (beta_vec != nullptr) {
      vec_t beta_data = beta_vec[i];
#pragma unroll
      for (int ii = 0; ii < vec_size; ii++) {
        out.val[ii] =
            (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean)) +
            static_cast<T_ACC>(beta_data.val[ii]);
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < vec_size; ii++) {
        if constexpr (!rms_norm) {
          out.val[ii] = rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean);
        } else {
          out.val[ii] = rstd_val * static_cast<T_ACC>(data.val[ii]);
        }
      }
    }
    Y_vec[i] = out;
  }
  if (thrx == 0) {
    if constexpr (!rms_norm) {
      mean[i1] = wd.mean;
    }
    rstd[i1] = rstd_val;
  }
}

int64_t layer_norm_wg_size_select(
    const int64_t max_wg_size,
    const int64_t M,
    const int n) {
  if (n > max_wg_size)
    return max_wg_size;

  int64_t wg_size = max_wg_size;
  while (wg_size > n && wg_size > SIMD) {
    wg_size >>= 1;
  }

  // To reduce the barrier overhead during tree-reduce
  // with 4 subgroups per workgroup
  constexpr int64_t threads_per_wg = 4;
  constexpr int64_t preferred_wg_size = SIMD * threads_per_wg;

  // keep wg_size when n is not large enough to utilize preferred_wg_size
  if (wg_size <= preferred_wg_size)
    return wg_size;

  // (XeCore count * EUs per XeCore) * HW threads per EU
  int64_t total_hw_threads = at::xpu::getDeviceHWThreads();
  // Only use preferred_wg_size when less than 50% HW threads would be left idle
  if (M * threads_per_wg > total_hw_threads / 2)
    return preferred_wg_size;

  return wg_size;
}

template <typename T, typename T_ACC, bool rms_norm>
void launch_vectorized_layer_norm_kernel(
    int N,
    int64_t M,
    T_ACC eps,
    const T* X_data,
    const T* gamma_data,
    const T* beta_data,
    T* Y_data,
    T_ACC* mean_data,
    T_ACC* rstd_data) {
  auto wg_size = layer_norm_wg_size_select(
      at::xpu::getKernelMaxWorkGroupSize<
          vectorized_layer_norm_kernel<T, T_ACC, rms_norm>>(),
      M,
      N / vec_size);
  sycl::range<2> local_range{size_t(wg_size / SIMD), SIMD};
  sycl::range<2> global_range(size_t(wg_size / SIMD), M * SIMD);
  auto queue = getCurrentSYCLQueue();
  size_t slm_sz = sizeof(T_ACC) * (wg_size / SIMD) * 2;

  sycl_kernel_submit<vectorized_layer_norm_kernel<T, T_ACC, rms_norm>>(
      global_range,
      local_range,
      queue,
      slm_sz,
      N,
      eps,
      X_data,
      gamma_data,
      beta_data,
      mean_data,
      rstd_data,
      Y_data,
      wg_size);
}

template <typename T, typename T_ACC, bool rms_norm = false>
void layer_norm_kernel_impl(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    T_ACC eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  const T* X_data = X.const_data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.const_data_ptr<T>() : nullptr;
  T* Y_data = Y->data_ptr<T>();
  T_ACC* mean_data = nullptr;
  if constexpr (!rms_norm) {
    mean_data = mean->data_ptr<T_ACC>();
  }
  T_ACC* rstd_data = rstd->data_ptr<T_ACC>();
  constexpr int num_vec_elems = vec_size;
  constexpr int alignment = num_vec_elems * sizeof(T);
  bool can_vec_X = can_vectorize(X_data, alignment);
  bool can_vec_Y = can_vectorize(Y_data, alignment);
  bool can_vec_gamma =
      gamma.defined() ? can_vectorize(gamma_data, alignment) : true;
  bool can_vec_beta =
      beta.defined() ? can_vectorize(beta_data, alignment) : true;

  if ((std::is_same_v<T, float> || std::is_same_v<T, at::Half> ||
       std::is_same_v<T, at::BFloat16>) &&
      N <= static_cast<int64_t>(1ULL << std::numeric_limits<float>::digits) &&
      N % num_vec_elems == 0 && can_vec_X && can_vec_Y && can_vec_gamma &&
      can_vec_beta) {
    launch_vectorized_layer_norm_kernel<T, T_ACC, rms_norm>(
        static_cast<int>(N),
        M,
        eps,
        X_data,
        gamma_data,
        beta_data,
        Y_data,
        mean_data,
        rstd_data);
  } else {
    launch_rowwise_moments_kernel<T, T_ACC, rms_norm>(
        N, M, eps, X_data, mean_data, rstd_data);
    launch_layer_norm_forward_kernel<T, T_ACC, rms_norm>(
        N, M, X_data, mean_data, rstd_data, gamma_data, beta_data, Y_data);
  }
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename mean_t,
    typename weight_t,
    bool have_gamma = true,
    bool have_beta = true,
    bool rms_norm = false>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<3>))
void gamma_beta_reduce_kernel(
    const mean_t* mean_data,
    const mean_t* var_data,
    const scalar_t* dY_data,
    const scalar_t* X_data,
    weight_t* dg_data,
    weight_t* db_data,
    int64_t num_tile_m,
    int64_t num_tile_n,
    int64_t tile_size_m,
    int64_t tile_size_n,
    int64_t elements_per_thread,
    int64_t num_subgroup,
    int64_t M,
    int64_t N) {
  auto item = syclext::this_work_item::get_nd_item<3>();

  accscalar_t* lsm =
      reinterpret_cast<accscalar_t*>(syclexp::get_work_group_scratch_memory());
  size_t local_sum_beta_size = tile_size_n * tile_size_m / elements_per_thread;
  accscalar_t* local_sum_beta = lsm;
  accscalar_t* local_sum_gamma = local_sum_beta + local_sum_beta_size;

  auto local_n = item.get_local_id(2); // [0, 32)
  auto local_m = item.get_local_id(1); // [0, 8)
  for (auto tile_id = item.get_global_id(0); tile_id < num_tile_n * num_tile_m;
       tile_id += item.get_group_range(0)) {
    auto tile_id_n = tile_id % num_tile_n;
    auto tile_id_m = tile_id / num_tile_n;
    auto tile_actual_row_base = tile_id_m * tile_size_m;
    auto tile_actual_col_base = tile_id_n * tile_size_n;
    auto actual_column = tile_actual_col_base + local_n;
    if (actual_column < N) {
      // slm_row 0, 8, 16...56
      for (auto slm_row = 0; slm_row < tile_size_m / elements_per_thread;
           slm_row += num_subgroup) {
        accscalar_t sum_beta = accscalar_t(0);
        accscalar_t sum_gamma = accscalar_t(0);
        // row 0, 128, 256, ...896
        auto row = tile_actual_row_base + slm_row * elements_per_thread;
        for (int i = 0; i < elements_per_thread; i++) {
          // row_local: row + 0, 8, 16, ...120
          auto row_local = row + i * num_subgroup;
          auto actual_row = row_local + local_m;
          // TODO: try tree reduction here if accuracy loss
          if (actual_row < M) {
            if constexpr (have_beta && !rms_norm) {
              sum_beta += static_cast<accscalar_t>(
                  dY_data[actual_row * N + actual_column]);
            }
            if constexpr (have_gamma) {
              if constexpr (!rms_norm) {
                sum_gamma += static_cast<accscalar_t>(
                                 dY_data[actual_row * N + actual_column]) *
                    (static_cast<accscalar_t>(
                         X_data[actual_row * N + actual_column]) -
                     static_cast<accscalar_t>(mean_data[actual_row])) *
                    static_cast<accscalar_t>(var_data[actual_row]);
              } else {
                sum_gamma += static_cast<accscalar_t>(
                                 dY_data[actual_row * N + actual_column]) *
                    (static_cast<accscalar_t>(
                        X_data[actual_row * N + actual_column])) *
                    static_cast<accscalar_t>(var_data[actual_row]);
              }
            }
          }
        }
        if constexpr (have_beta && !rms_norm) {
          local_sum_beta[(slm_row + local_m) * tile_size_n + local_n] =
              sum_beta;
        }
        if constexpr (have_gamma) {
          local_sum_gamma[(slm_row + local_m) * tile_size_n + local_n] =
              sum_gamma;
        }
      }

      // sycl::group_barrier(item.get_group());
      accscalar_t slm_sum_beta = accscalar_t(0);
      accscalar_t slm_sum_gamma = accscalar_t(0);
      // slm row 64, 8 subgroup, i = 0,2,4,6
      // slm row 32, 8 subgroup, i = 0,2
      // slm row 16, 8 subgroup, i = 0
      for (int i = 0; i < tile_size_m / elements_per_thread / num_subgroup;
           i = i + 1) {
        if constexpr (have_beta && !rms_norm) {
          slm_sum_beta += local_sum_beta
              [(i * num_subgroup + local_m) * tile_size_n + local_n];
        }
        if constexpr (have_gamma) {
          slm_sum_gamma += local_sum_gamma
              [(i * num_subgroup + local_m) * tile_size_n + local_n];
        }
      }
      if constexpr (have_beta && !rms_norm) {
        local_sum_beta[local_m * tile_size_n + local_n] = slm_sum_beta;
      }
      if constexpr (have_gamma) {
        local_sum_gamma[local_m * tile_size_n + local_n] = slm_sum_gamma;
      }
    }
    sycl::group_barrier(item.get_group());
    accscalar_t output_sum_beta = accscalar_t(0);
    accscalar_t output_sum_gamma = accscalar_t(0);
    if (local_m == 0 && actual_column < N) {
      for (int i = 0; i < num_subgroup; i = i + 1) {
        if constexpr (have_beta && !rms_norm) {
          output_sum_beta += local_sum_beta[i * tile_size_n + local_n];
        }
        if constexpr (have_gamma) {
          output_sum_gamma += local_sum_gamma[i * tile_size_n + local_n];
        }
      }
      if constexpr (have_beta && !rms_norm) {
        db_data[tile_id_m * N + actual_column] =
            static_cast<weight_t>(output_sum_beta);
      }

      if constexpr (have_gamma) {
        dg_data[tile_id_m * N + actual_column] =
            static_cast<weight_t>(output_sum_gamma);
      }
    }
  }
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename mean_t,
    typename weight_t,
    int vec_size,
    bool rms_norm>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<3>))
void gamma_beta_backward_simple_kernel(
    const mean_t* mean_data,
    const mean_t* var_data,
    NormConfig cfg,
    const scalar_t* dY_data,
    const scalar_t* X_data,
    weight_t* dg_data,
    weight_t* db_data) {
  using vec_t = at::native::memory::aligned_vector<scalar_t, vec_size>;
  using weight_vec_t = at::native::memory::aligned_vector<weight_t, vec_size>;
  auto item_id = syclext::this_work_item::get_nd_item<3>();

  accscalar_t* lsm =
      reinterpret_cast<accscalar_t*>(syclexp::get_work_group_scratch_memory());

  size_t local_sum1_size = cfg.block_row * cfg.workgroup_size * vec_size;

  accscalar_t* local_sum1 = lsm;
  accscalar_t* local_sum2 = local_sum1 + local_sum1_size;

  auto local_row_id = item_id.get_local_id(1);
  auto local_col_id = item_id.get_local_id(2);
  auto group_id = item_id.get_group(0);

  accscalar_t dg_sum1[vec_size], db_sum1[vec_size];
#pragma unroll(vec_size)
  for (int v = 0; v < vec_size; ++v) {
    dg_sum1[v] = 0;
    if constexpr (!rms_norm) {
      db_sum1[v] = 0;
    }
  }

  for (int row_id = local_row_id; row_id < cfg.batch_size;
       row_id += cfg.block_row) {
    accscalar_t mean_val = accscalar_t(0);
    if constexpr (!rms_norm) {
      mean_val = mean_data[row_id];
    }
    accscalar_t rstd_val = var_data[row_id];
    auto plane_offset =
        (group_id * cfg.workgroup_size + local_col_id) * vec_size;
    if (plane_offset < cfg.problem_size) {
      auto offset = row_id * cfg.problem_size + plane_offset;
      vec_t X_val = *(reinterpret_cast<const vec_t*>(X_data + offset));
      vec_t dY_val = *(reinterpret_cast<const vec_t*>(dY_data + offset));
#pragma unroll(vec_size)
      for (int v = 0; v < vec_size; ++v) {
        if constexpr (!rms_norm) {
          dg_sum1[v] += (dg_data == nullptr)
              ? accscalar_t(0)
              : static_cast<accscalar_t>(dY_val[v]) *
                  (static_cast<accscalar_t>(X_val[v]) - mean_val) * rstd_val;
        } else {
          dg_sum1[v] += (dg_data == nullptr)
              ? accscalar_t(0)
              : static_cast<accscalar_t>(dY_val[v]) *
                  (static_cast<accscalar_t>(X_val[v])) * rstd_val;
        }
        if constexpr (!rms_norm) {
          db_sum1[v] += (db_data == nullptr)
              ? accscalar_t(0)
              : static_cast<accscalar_t>(dY_val[v]);
        }
      }
    }
  }

  if (cfg.block_row > 1) {
    norm_group_reduce_row<vec_size, accscalar_t, rms_norm>(
        item_id,
        dg_sum1,
        db_sum1,
        local_sum1,
        local_sum2,
        cfg.block_row,
        cfg.workgroup_size,
        [](accscalar_t a, accscalar_t b) { return a + b; });
  }

  if (local_row_id == 0) {
    auto plane_offset =
        (group_id * cfg.workgroup_size + local_col_id) * vec_size;
    if (plane_offset < cfg.problem_size) {
      weight_vec_t dg_val, db_val;
      if (cfg.block_row > 1) {
#pragma unroll(vec_size)
        for (int v = 0; v < vec_size; ++v) {
          dg_val[v] =
              static_cast<weight_t>(local_sum1[local_col_id * vec_size + v]);
          if constexpr (!rms_norm) {
            db_val[v] =
                static_cast<weight_t>(local_sum2[local_col_id * vec_size + v]);
          }
        }
      } else {
#pragma unroll(vec_size)
        for (int v = 0; v < vec_size; ++v) {
          dg_val[v] = static_cast<weight_t>(dg_sum1[v]);
          if constexpr (!rms_norm) {
            db_val[v] = static_cast<weight_t>(db_sum1[v]);
          }
        }
      }
      if (dg_data != nullptr) {
        *(reinterpret_cast<weight_vec_t*>(dg_data + plane_offset)) = dg_val;
      }
      if constexpr (!rms_norm) {
        if (db_data != nullptr) {
          *(reinterpret_cast<weight_vec_t*>(db_data + plane_offset)) = db_val;
        }
      }
    }
  }
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename mean_t,
    typename weight_t,
    int vec_size,
    bool rms_norm>
void vec_gamma_beta_bwd_simple_kernel(
    const Tensor& dY,
    const Tensor& X,
    const mean_t* mean_data,
    const mean_t* var_data,
    Tensor* dgamma,
    Tensor* dbeta,
    NormConfig& cfg) {
  const scalar_t* dY_data = dY.const_data_ptr<scalar_t>();
  const scalar_t* X_data = X.const_data_ptr<scalar_t>();
  weight_t* dg_data =
      dgamma->defined() ? dgamma->data_ptr<weight_t>() : nullptr;
  weight_t* db_data = dbeta->defined() ? dbeta->data_ptr<weight_t>() : nullptr;

  sycl::range<3> local_range{
      1, (size_t)cfg.block_row, (size_t)cfg.workgroup_size};
  sycl::range<3> global_range{
      (size_t)cfg.workgroup_num,
      (size_t)cfg.block_row,
      (size_t)cfg.workgroup_size};
  size_t lsm_size =
      2 * cfg.block_row * cfg.workgroup_size * vec_size * sizeof(accscalar_t);

  sycl_kernel_submit<gamma_beta_backward_simple_kernel<
      scalar_t,
      accscalar_t,
      mean_t,
      weight_t,
      vec_size,
      rms_norm>>(
      global_range,
      local_range,
      getCurrentSYCLQueue(),
      lsm_size,
      mean_data,
      var_data,
      cfg,
      dY_data,
      X_data,
      dg_data,
      db_data);
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename mean_t,
    typename weight_t,
    bool rms_norm>
void gamma_beta_bwd_simple_kernel(
    const Tensor& dY,
    const Tensor& X,
    const mean_t* mean_data,
    const mean_t* var_data,
    Tensor* dgamma,
    Tensor* dbeta,
    NormConfig& config) {
#define VECTORIZE_KERNEL(vec_size)                                  \
  vec_gamma_beta_bwd_simple_kernel<                                 \
      scalar_t,                                                     \
      accscalar_t,                                                  \
      mean_t,                                                       \
      weight_t,                                                     \
      vec_size,                                                     \
      rms_norm>(dY, X, mean_data, var_data, dgamma, dbeta, config); \
  break;

  switch (config.max_vec_size) {
    case 8: {
      VECTORIZE_KERNEL(8);
    }
    case 4: {
      VECTORIZE_KERNEL(4);
    }
    case 2: {
      VECTORIZE_KERNEL(2);
    }
    case 1: {
      VECTORIZE_KERNEL(1);
    }
  }
#undef VECTORIZE_KERNEL
}

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    bool rms_norm = false>
void layer_norm_backward_kernel_impl(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor* dX,
    Tensor* dgamma,
    Tensor* dbeta) {
  TORCH_CHECK(dY.numel() == M * N);
  if constexpr (!rms_norm) {
    TORCH_CHECK(mean.numel() == M);
  }
  TORCH_CHECK(rstd.numel() == M);

  using accscalar_t = acc_type_device<scalar_t, kXPU>;
  const mean_t* mean_data = mean.const_data_ptr<mean_t>();
  const mean_t* var_data = rstd.const_data_ptr<mean_t>();
  const weight_t* gamma_data =
      gamma.defined() ? gamma.const_data_ptr<weight_t>() : nullptr;

  if (dX->defined()) {
    // backward data
    const scalar_t* X_data = X.const_data_ptr<scalar_t>();
    const scalar_t* dY_data = dY.const_data_ptr<scalar_t>();
    scalar_t* dX_data = dX->data_ptr<scalar_t>();

    auto config = NormConfig(M, N, 1, sizeof(scalar_t));
    bool can_use_32bit_index = canUse32BitIndexMath(X) &&
        canUse32BitIndexMath(dY) && canUse32BitIndexMath(*dX);
    if (config.workgroup_num_foreach == 1) {
      LayerNormBackward<scalar_t, mean_t, weight_t, rms_norm> norm(
          X_data, dY_data, dX_data, mean_data, var_data, gamma_data, M, N);
      vectorized_fused_norm_kernel<
          scalar_t,
          mean_t,
          weight_t,
          LayerNormBackward,
          rms_norm>(norm, config, can_use_32bit_index);
    } else {
      const auto kAccType =
          (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
          ? kFloat
          : X.scalar_type();
      Tensor a =
          rms_norm ? Tensor() : at::empty({M}, X.options().dtype(kAccType));
      accscalar_t* a_data = rms_norm ? nullptr : a.data_ptr<accscalar_t>();
      Tensor b = at::empty({M}, X.options().dtype(kAccType));
      accscalar_t* b_data = b.data_ptr<accscalar_t>();

      LayerNormBackward<scalar_t, mean_t, weight_t, rms_norm> norm(
          X_data,
          dY_data,
          dX_data,
          mean_data,
          var_data,
          gamma_data,
          a_data,
          b_data,
          M,
          N);
      Tensor semaphores, scratchpad;
      config.template init_global_reduce<rms_norm>(X, semaphores, scratchpad);
      rowwise_moments_kernel<
          scalar_t,
          mean_t,
          weight_t,
          LayerNormBackward,
          rms_norm>(norm, config, can_use_32bit_index);
      norm_update_kernel<
          scalar_t,
          mean_t,
          weight_t,
          LayerNormBackward,
          rms_norm>(norm, config, can_use_32bit_index);
    }
  }
  auto config_w = NormConfig(M, N, 0, sizeof(scalar_t));
  auto norm_config_global_size =
      config_w.workgroup_num * config_w.block_row * config_w.workgroup_size;
  int thread_slots = at::xpu::getDeviceHWThreads();
  // use two stage col reduction if norm config occupancy < 50%
  // TODO: we can relax this restriction in future for better perf
  bool use_two_stage_col_reduction =
      (dY.dtype() == kFloat || dY.dtype() == kBFloat16 ||
       dY.dtype() == kHalf) &&
      norm_config_global_size / at::xpu::getDeviceMaxSubGroupSize() * 2 <=
          thread_slots;
  // cuda uses condition M > 64 * 1024 && N / 32 < sm_count / 2 to parallelize
  // in the M dimension
  int xe_core_count = at::xpu::getDeviceXeCoreCount();
  int tile_n = N / 32;
  if (use_two_stage_col_reduction && M > xe_core_count * 1024 &&
      tile_n < xe_core_count * 2) {
    const size_t local_size_x = 8;
    const size_t SIMD = 32;
    // workgroup size is 256
    // slm is 16KB, 64*32 float * 2
    // elements_per_thread is at least 16
    const int elements_per_thread = 16;
    int tile_size_m = 1024;
    int tile_size_n = N < 32 ? N : 32;
    int num_tile_m = (M + tile_size_m - 1) / tile_size_m;
    int num_tile_n = (N + tile_size_n - 1) / tile_size_n;
    bool adjust_m = true;
    // for M = 64*1024, N = 1, we choose tile size (256, 16) on pvc
    // TODO: Consider tuning the tile size selection logic (tile_size_m,
    // tile_size_n) and occupancy calculation
    for (auto i = 0; i < 3; i++) {
      // occupancy <= 50%
      if (num_tile_m * num_tile_n * local_size_x * SIMD /
              at::xpu::getDeviceMaxSubGroupSize() * 2 <=
          thread_slots) {
        if (adjust_m) {
          tile_size_m /= 2;
          num_tile_m = (M + tile_size_m - 1) / tile_size_m;
          adjust_m = false;
        } else {
          tile_size_n /= 2;
          num_tile_n = (N + tile_size_n - 1) / tile_size_n;
          adjust_m = true;
        }
      } else {
        break;
      }
    }
    // tile size can be (1024,32), (512,32), (512,16), (256, 16)
    // Modifying these parameters (num_subgroup, workgroup_size, tile_size,
    // elements_per_thread) will alter the kernel configuration, potentially
    // affecting performance and behavior.
    const scalar_t* dY_data = dY.const_data_ptr<scalar_t>();
    const scalar_t* X_data = X.const_data_ptr<scalar_t>();
    Tensor dgamma_blocks;
    Tensor dbeta_blocks;
    weight_t* dgamma_blocks_ptr = nullptr;
    weight_t* dbeta_blocks_ptr = nullptr;
    if (dgamma->defined()) {
      auto options = dgamma->options();
      // TODO: how to set dgamma_blocks dtype = float32?
      dgamma_blocks = at::empty({num_tile_m, N}, options);
      dgamma_blocks_ptr = dgamma_blocks.data_ptr<weight_t>();
    }
    if (dbeta->defined()) {
      if constexpr (!rms_norm) {
        auto options = dbeta->options();
        dbeta_blocks = at::empty({num_tile_m, N}, options);
        dbeta_blocks_ptr = dbeta_blocks.data_ptr<weight_t>();
      }
    }

    size_t num_workgroup = std::min(
        num_tile_m * num_tile_n, static_cast<int>(thread_slots / local_size_x));
    if (dgamma->defined() && dbeta->defined()) {
      size_t lsm_size = 2 * tile_size_n * tile_size_m / elements_per_thread *
          sizeof(accscalar_t);

      sycl_kernel_submit<gamma_beta_reduce_kernel<
          scalar_t,
          accscalar_t,
          mean_t,
          weight_t,
          true,
          true,
          rms_norm>>(
          sycl::range<3>(
              num_workgroup,
              local_size_x,
              static_cast<size_t>(tile_size_n < SIMD ? tile_size_n : SIMD)),
          sycl::range<3>(
              1,
              local_size_x,
              static_cast<size_t>(tile_size_n < SIMD ? tile_size_n : SIMD)),
          getCurrentSYCLQueue(),
          lsm_size,
          mean_data,
          var_data,
          dY_data,
          X_data,
          dgamma_blocks_ptr,
          dbeta_blocks_ptr,
          num_tile_m,
          num_tile_n,
          tile_size_m,
          tile_size_n,
          elements_per_thread,
          local_size_x,
          M,
          N);
      *dgamma = dgamma_blocks.sum(0);
      if constexpr (!rms_norm) {
        *dbeta = dbeta_blocks.sum(0);
      }
    } else if (dgamma->defined() && !dbeta->defined()) {
      size_t lsm_size = 2 * tile_size_n * tile_size_m / elements_per_thread *
          sizeof(accscalar_t);
      sycl_kernel_submit<gamma_beta_reduce_kernel<
          scalar_t,
          accscalar_t,
          mean_t,
          weight_t,
          true,
          false,
          rms_norm>>(
          sycl::range<3>(
              num_workgroup,
              local_size_x,
              static_cast<size_t>(tile_size_n < SIMD ? tile_size_n : SIMD)),
          sycl::range<3>(
              1,
              local_size_x,
              static_cast<size_t>(tile_size_n < SIMD ? tile_size_n : SIMD)),
          getCurrentSYCLQueue(),
          lsm_size,
          mean_data,
          var_data,
          dY_data,
          X_data,
          dgamma_blocks_ptr,
          dbeta_blocks_ptr,
          num_tile_m,
          num_tile_n,
          tile_size_m,
          tile_size_n,
          elements_per_thread,
          local_size_x,
          M,
          N);
      *dgamma = dgamma_blocks.sum(0);
    } else if (!dgamma->defined() && dbeta->defined()) {
      size_t lsm_size = 2 * tile_size_n * tile_size_m / elements_per_thread *
          sizeof(accscalar_t);
      sycl_kernel_submit<gamma_beta_reduce_kernel<
          scalar_t,
          accscalar_t,
          mean_t,
          weight_t,
          false,
          true,
          rms_norm>>(
          sycl::range<3>(
              num_workgroup,
              local_size_x,
              static_cast<size_t>(tile_size_n < SIMD ? tile_size_n : SIMD)),
          sycl::range<3>(
              1,
              local_size_x,
              static_cast<size_t>(tile_size_n < SIMD ? tile_size_n : SIMD)),
          getCurrentSYCLQueue(),
          lsm_size,
          mean_data,
          var_data,
          dY_data,
          X_data,
          dgamma_blocks_ptr,
          dbeta_blocks_ptr,
          num_tile_m,
          num_tile_n,
          tile_size_m,
          tile_size_n,
          elements_per_thread,
          local_size_x,
          M,
          N);
      *dbeta = dbeta_blocks.sum(0);
    } else {
      return;
    }

  } else {
    gamma_beta_bwd_simple_kernel<
        scalar_t,
        accscalar_t,
        mean_t,
        weight_t,
        rms_norm>(dY, X, mean_data, var_data, dgamma, dbeta, config_w);
  }
}

void layer_norm_kernel(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    double eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "layer_norm_xpu",
      [&]() {
        using acc_t = acc_type_device<scalar_t, kXPU>;
        layer_norm_kernel_impl<scalar_t, acc_t>(
            X, gamma, beta, M, N, static_cast<acc_t>(eps), Y, mean, rstd);
      });
}

void layer_norm_backward_kernel(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor* dX,
    Tensor* dgamma,
    Tensor* dbeta) {
  if (M > 0 && N > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        X.scalar_type(),
        "layer_norm_backward_xpu",
        [&]() {
          using accscalar_t = acc_type_device<scalar_t, kXPU>;
          layer_norm_backward_kernel_impl<scalar_t, accscalar_t, scalar_t>(
              dY.contiguous(), X, mean, rstd, gamma, M, N, dX, dgamma, dbeta);
        });
  }
}

void rms_norm_kernel(
    const Tensor& X,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    double eps,
    Tensor* Y,
    Tensor* rstd) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "rms_norm_xpu",
      [&]() {
        using acc_t = acc_type_device<scalar_t, kXPU>;
        layer_norm_kernel_impl<scalar_t, acc_t, true>(
            X,
            gamma,
            at::Tensor(),
            M,
            N,
            static_cast<acc_t>(eps),
            Y,
            nullptr,
            rstd);
      });
}

void rms_norm_backward_kernel(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor* dX,
    Tensor* dgamma) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "rms_norm_backward_xpu",
      [&]() {
        using accscalar_t = acc_type_device<scalar_t, kXPU>;
        Tensor unused_dbeta;
        layer_norm_backward_kernel_impl<scalar_t, accscalar_t, scalar_t, true>(
            dY.contiguous(),
            X,
            rstd,
            rstd,
            gamma,
            M,
            N,
            dX,
            dgamma,
            &unused_dbeta);
      });
}

} // namespace xpu
} // namespace native
} // namespace at
