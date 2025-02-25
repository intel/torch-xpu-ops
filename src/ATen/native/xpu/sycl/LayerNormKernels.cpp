#include <ATen/Dispatch.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/GroupReduceUtils.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/Norm.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/LayerNormKernels.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t, typename mean_t, typename weight_t>
class LayerNormBackward : public NormBackward<scalar_t, mean_t, weight_t> {
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
      : NormBackward<scalar_t, mean_t, weight_t>(
            X_data,
            dY_data,
            dX_data,
            mean_data,
            var_data,
            gamma_data,
            nullptr,
            nullptr),
        M(M),
        N(N) {
    numel = M * N;
  }

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
      : NormBackward<scalar_t, mean_t, weight_t>(
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
  typedef NormBackward<scalar_t, mean_t, weight_t> NB;

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
          sum1 += value;
          sum2 +=
              value * static_cast<accscalar_t>(X_val[v] - mean_val) * rstd_val;
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
      sum1 = NB::a_data[group_id];
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
          f_grad_input -= (X_val[v] - mean_val) * var_val * sum2;
          dX_val[v] = static_cast<scalar_t>((f_grad_input - sum1) * term1);
        }
        *(reinterpret_cast<vec_t*>(NB::dX_data + group_offset + plane_offset)) =
            dX_val;
      }
    }
  };

  int64_t M;
  int64_t N;
  int64_t numel;
};

constexpr int vec_size =
    4; // we could make it dependent on dtype, but that would lead to different
       // results between float and low-p types

// Checks alignment of buffers for using vectorized loads / stores
template <typename T>
bool can_vectorize(const T* ptr, int alignment) {
  uint64_t addr = reinterpret_cast<uint64_t>(ptr);
  return addr % alignment == 0;
};

template <typename T, typename T_ACC>
struct RowwiseMomentsFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int64_t, std::pair<T_ACC, T_ACC>>;

  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<1> item_id) const {
    const int64_t i = item_id.get_group(0);
    WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
    WelfordType val(0, 0, 0, 0);
    for (int64_t j = item_id.get_local_id(0); j < N_;
         j += item_id.get_local_range(0)) {
      const int64_t index = i * N_ + j;
      val = welford_op.reduce(val, static_cast<T_ACC>(X_[index]), index);
    }

    val = GroupReduceWithoutBroadcast<WelfordType, WelfordOp, SIMD>(
        item_id, val, welford_op, shared_);

    if (item_id.get_local_id(0) == 0) {
      T_ACC m1;
      T_ACC m2;
      std::tie(m2, m1) = welford_op.project(val);
      mean_[i] = m1;
      rstd_[i] = c10::xpu::compat::rsqrt(m2 + eps_);
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<WelfordType>(SIMD, cgh);
  }

  RowwiseMomentsFunctor(
      int64_t N,
      T_ACC eps,
      const T* X,
      T_ACC* mean,
      T_ACC* rstd)
      : N_(N), eps_(eps), X_(X), mean_(mean), rstd_(rstd) {}

 private:
  int64_t N_;
  T_ACC eps_;
  const T* X_;
  T_ACC* mean_;
  T_ACC* rstd_;
  sycl_local_acc_t<WelfordType> shared_;
};

template <typename T, typename T_ACC>
void launch_rowwise_moments_kernel(
    int64_t N,
    int64_t M,
    T_ACC eps,
    const T* X_data,
    T_ACC* mean_data,
    T_ACC* rstd_data) {
  RowwiseMomentsFunctor<T, T_ACC> kfn(N, eps, X_data, mean_data, rstd_data);

  int64_t sg_size = SIMD;
  int64_t wg_size = get_group_reduce_group_size(sg_size);
  sycl::range<1> local_range{size_t(wg_size)};
  sycl::range<1> global_range{size_t(M * wg_size)};
  auto queue = getCurrentSYCLQueue();

  sycl_kernel_submit(global_range, local_range, queue, kfn);
}

template <typename T, typename T_ACC>
struct LayerNormForwardKernelFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    const int64_t i = item_id.get_group(0);
    for (int64_t j = item_id.get_local_id(0); j < N_;
         j += item_id.get_local_range(0)) {
      const int64_t index = i * N_ + j;
      const T_ACC gamma_v =
          gamma_ == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma_[j]);
      const T_ACC beta_v =
          beta_ == nullptr ? T_ACC(0) : static_cast<T_ACC>(beta_[j]);
      Y_[index] =
          (static_cast<T_ACC>(X_[index]) - static_cast<T_ACC>(mean_[i])) *
              static_cast<T_ACC>(rstd_[i]) * gamma_v +
          beta_v;
    }
  }
  LayerNormForwardKernelFunctor(
      int64_t N,
      const T* X,
      const T_ACC* mean,
      const T_ACC* rstd,
      const T* gamma,
      const T* beta,
      T* Y)
      : N_(N),
        X_(X),
        mean_(mean),
        rstd_(rstd),
        gamma_(gamma),
        beta_(beta),
        Y_(Y) {}

 private:
  int64_t N_;
  const T* X_;
  const T_ACC* mean_;
  const T_ACC* rstd_;
  const T* gamma_;
  const T* beta_;
  T* Y_;
};

template <typename T, typename T_ACC>
void launch_layer_norm_forward_kernel(
    int64_t N,
    int64_t M,
    const T* X_data,
    const T_ACC* mean_data,
    const T_ACC* rstd_data,
    const T* gamma_data,
    const T* beta_data,
    T* Y_data) {
  LayerNormForwardKernelFunctor<T, T_ACC> kfn(
      N, X_data, mean_data, rstd_data, gamma_data, beta_data, Y_data);

  int64_t sg_size = SIMD;
  int64_t wg_size = get_group_reduce_group_size(sg_size);
  sycl::range<1> local_range{size_t(wg_size)};
  sycl::range<1> global_range(M * size_t(wg_size));
  auto queue = getCurrentSYCLQueue();

  sycl_kernel_submit(global_range, local_range, queue, kfn);
}

struct WelfordDataLN {
  float mean;
  float sigma2;
  float count;
  WelfordDataLN() : mean(0.f), sigma2(0.f), count(0.f) {}
  WelfordDataLN(float mean, float sigma2, float count)
      : mean(mean), sigma2(sigma2), count(count) {}
};

template <typename U>
WelfordDataLN WelfordOnlineSum(const U val, const WelfordDataLN& curr_sum) {
  U delta = val - curr_sum.mean;
  U new_count = curr_sum.count + 1.f;
  U new_mean = curr_sum.mean +
      delta * (1.f / new_count); // proper division is slow, this is less
                                 // accurate but noticeably faster
  return {
      static_cast<float>(new_mean),
      static_cast<float>(curr_sum.sigma2 + delta * (val - new_mean)),
      static_cast<float>(new_count)};
}

WelfordDataLN WelfordCombine(
    const WelfordDataLN dataB,
    const WelfordDataLN dataA) {
  using U = decltype(dataB.count);
  U delta = dataB.mean - dataA.mean;
  U count = dataA.count + dataB.count;
  U mean, sigma2;
  if (count > decltype(dataB.count){0}) {
    auto coef = 1.f / count; // NB we don't use --use_fast_math, but this is
                             // emulation, 1./count goes to intrinsic, `* coef`
                             // is multiplication, instead of slow fp division
    auto nA = dataA.count * coef;
    auto nB = dataB.count * coef;
    mean = nA * dataA.mean + nB * dataB.mean;
    sigma2 = dataA.sigma2 + dataB.sigma2 + delta * delta * dataA.count * nB;
  } else {
    mean = U(0);
    sigma2 = U(0);
  }
  return {mean, sigma2, count};
}

template <typename T, typename T_ACC>
WelfordDataLN compute_stats(
    const T* RESTRICT X,
    const int N,
    T_ACC& buf,
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
      wd = WelfordOnlineSum(static_cast<acc_t>(data.val[ii]), wd);
    }
  }
  // intra-warp reduction
  auto sg = item_id.get_sub_group();
  for (int offset = (SIMD >> 1); offset > 0; offset >>= 1) {
    WelfordDataLN wdB{
        sycl::shift_group_left(sg, wd.mean, offset),
        sycl::shift_group_left(sg, wd.sigma2, offset),
        sycl::shift_group_left(sg, wd.count, offset)};
    wd = WelfordCombine(wd, wdB);
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
      item_id.barrier(sycl_local_fence);

      // lower half merges
      if (item_id.get_local_id(1) == 0 && item_id.get_local_id(0) < offset) {
        const int rd_y = item_id.get_local_id(0);
        WelfordDataLN wdB{
            static_cast<float>(buf[2 * rd_y]),
            static_cast<float>(buf[2 * rd_y + 1]),
            static_cast<float>(buf[rd_y + addr_offset])};
        wd = WelfordCombine(wd, wdB);
      }
      item_id.barrier(sycl_local_fence);
    }

    if (item_id.get_local_id(1) == 0 && item_id.get_local_id(0) == 0) {
      buf[0] = wd.mean;
      buf[1] = wd.sigma2 / float(N);
    }
    item_id.barrier(sycl_local_fence);
    return WelfordDataLN{
        static_cast<float>(buf[0]), static_cast<float>(buf[1]), 0.f};
  } else {
    return WelfordDataLN{
        sycl::select_from_group(sg, wd.mean, 0),
        sycl::select_from_group(sg, wd.sigma2, 0) / float(N),
        0.f};
  }
}

template <typename T, typename T_ACC>
struct VectorizedLayerNormKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<2> item_id) const {
    auto i1 = item_id.get_group(1);
    const T* block_row = X_ + i1 * N_;
    WelfordDataLN wd = compute_stats<T>(block_row, N_, buf_, item_id);

    using vec_t = aligned_vector<T, vec_size>;
    const vec_t* X_vec = reinterpret_cast<const vec_t*>(block_row);
    const vec_t* gamma_vec =
        (gamma_ != nullptr) ? reinterpret_cast<const vec_t*>(gamma_) : nullptr;
    const vec_t* beta_vec =
        (beta_ != nullptr) ? reinterpret_cast<const vec_t*>(beta_) : nullptr;
    vec_t* Y_vec = reinterpret_cast<vec_t*>(Y_ + i1 * N_);

    const int numx = item_id.get_local_range(1) * item_id.get_local_range(0);
    const int thrx = item_id.get_local_linear_id();
    const int n_vec_to_read = N_ / vec_size;

    T_ACC rstd_val = c10::xpu::compat::rsqrt(wd.sigma2 + eps_);

    // No tail, N is guaranteed to be multiple of vec size
    for (int i = thrx; i < n_vec_to_read; i += numx) {
      vec_t data = X_vec[i];
      vec_t out;

      // Computation is performed in T_ACC, X is cast to T_ACC and result is
      // implicitly cast to T
      if (gamma_vec != nullptr && beta_vec != nullptr) {
#pragma unroll
        for (int ii = 0; ii < vec_size; ii++) {
          out.val[ii] = static_cast<T_ACC>(gamma_vec[i].val[ii]) *
                  (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean)) +
              static_cast<T_ACC>(beta_vec[i].val[ii]);
        }
      } else if (gamma_vec != nullptr) {
#pragma unroll
        for (int ii = 0; ii < vec_size; ii++) {
          out.val[ii] = static_cast<T_ACC>(gamma_vec[i].val[ii]) *
              (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean));
        }
      } else if (beta_vec != nullptr) {
#pragma unroll
        for (int ii = 0; ii < vec_size; ii++) {
          out.val[ii] =
              (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean)) +
              static_cast<T_ACC>(beta_vec[i].val[ii]);
        }
      } else {
#pragma unroll
        for (int ii = 0; ii < vec_size; ii++) {
          out.val[ii] = rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean);
        }
      }
      Y_vec[i] = out;
    }
    if (thrx == 0) {
      mean_[i1] = wd.mean;
      rstd_[i1] = rstd_val;
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    buf_ = sycl_local_acc_t<T_ACC>((wg_size_ / SIMD) * 2, cgh);
  }

  VectorizedLayerNormKernelFunctor(
      const int N,
      T_ACC eps,
      const T* RESTRICT X,
      const T* gamma,
      const T* beta,
      T_ACC* mean,
      T_ACC* rstd,
      T* Y,
      int64_t wg_size)
      : N_(N),
        eps_(eps),
        X_(X),
        gamma_(gamma),
        beta_(beta),
        mean_(mean),
        rstd_(rstd),
        Y_(Y),
        wg_size_(wg_size) {}

 private:
  const int N_;
  T_ACC eps_;
  const T* RESTRICT X_;
  const T* gamma_;
  const T* beta_;
  T_ACC* mean_;
  T_ACC* rstd_;
  T* Y_;
  int64_t sg_size_;
  int64_t wg_size_;
  sycl_local_acc_t<T_ACC> buf_;
};

template <typename T, typename T_ACC>
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
  using KernelClass = VectorizedLayerNormKernelFunctor<T, T_ACC>;
  int64_t wg_size = syclMaxWorkGroupSize<KernelClass>();
  KernelClass kfn(
      N,
      eps,
      X_data,
      gamma_data,
      beta_data,
      mean_data,
      rstd_data,
      Y_data,
      wg_size);
  sycl::range<2> local_range{size_t(wg_size / SIMD), SIMD};
  sycl::range<2> global_range(size_t(wg_size / SIMD), M * SIMD);
  auto queue = getCurrentSYCLQueue();
  sycl_kernel_submit(global_range, local_range, queue, kfn);
}

template <typename T, typename T_ACC>
void _layer_norm_kernel(
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
  T_ACC* mean_data = mean->data_ptr<T_ACC>();
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
       std::is_same_v<T, at::BFloat16>)&&N <=
          static_cast<int64_t>(1ULL << std::numeric_limits<float>::digits) &&
      N % num_vec_elems == 0 && can_vec_X && can_vec_Y && can_vec_gamma &&
      can_vec_beta) {
    launch_vectorized_layer_norm_kernel(
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
    launch_rowwise_moments_kernel(N, M, eps, X_data, mean_data, rstd_data);
    launch_layer_norm_forward_kernel(
        N, M, X_data, mean_data, rstd_data, gamma_data, beta_data, Y_data);
  }
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename mean_t,
    typename weight_t,
    int vec_size,
    typename vec_t,
    typename weight_vec_t>
struct GammaBetaBackwardSimpleKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<3> item_id) const {
    auto local_row_id = item_id.get_local_id(1);
    auto local_col_id = item_id.get_local_id(2);
    auto group_id = item_id.get_group(0);

    accscalar_t dg_sum1[vec_size], db_sum1[vec_size];
#pragma unroll(vec_size)
    for (int v = 0; v < vec_size; ++v) {
      dg_sum1[v] = 0;
      db_sum1[v] = 0;
    }

    for (int row_id = local_row_id; row_id < cfg.batch_size;
         row_id += cfg.block_row) {
      accscalar_t mean_val = mean_data[row_id];
      accscalar_t rstd_val = var_data[row_id];
      auto plane_offset =
          (group_id * cfg.workgroup_size + local_col_id) * vec_size;
      if (plane_offset < cfg.problem_size) {
        auto offset = row_id * cfg.problem_size + plane_offset;
        vec_t X_val = *(reinterpret_cast<const vec_t*>(X_data + offset));
        vec_t dY_val = *(reinterpret_cast<const vec_t*>(dY_data + offset));
#pragma unroll(vec_size)
        for (int v = 0; v < vec_size; ++v) {
          dg_sum1[v] += (dg_data == nullptr)
              ? accscalar_t(0)
              : static_cast<accscalar_t>(dY_val[v]) *
                  (static_cast<accscalar_t>(X_val[v]) - mean_val) * rstd_val;
          db_sum1[v] += (db_data == nullptr)
              ? accscalar_t(0)
              : static_cast<accscalar_t>(dY_val[v]);
        }
      }
    }

    if (cfg.block_row > 1) {
      norm_group_reduce_row<vec_size, accscalar_t>(
          item_id,
          dg_sum1,
          db_sum1,
          local_sum1,
          local_sum2,
          cfg.block_row,
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
            dg_val[v] = static_cast<weight_t>(local_sum1[0][local_col_id][v]);
            db_val[v] = static_cast<weight_t>(local_sum2[0][local_col_id][v]);
          }
        } else {
#pragma unroll(vec_size)
          for (int v = 0; v < vec_size; ++v) {
            dg_val[v] = static_cast<weight_t>(dg_sum1[v]);
            db_val[v] = static_cast<weight_t>(db_sum1[v]);
          }
        }
        if (dg_data != nullptr) {
          *(reinterpret_cast<weight_vec_t*>(dg_data + plane_offset)) = dg_val;
        }
        if (db_data != nullptr) {
          *(reinterpret_cast<weight_vec_t*>(db_data + plane_offset)) = db_val;
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    local_sum1 = sycl_local_acc_t<accscalar_t, 3>(
        sycl::range<3>(
            (size_t)cfg.block_row,
            (size_t)cfg.workgroup_size,
            (size_t)vec_size),
        cgh);
    local_sum2 = sycl_local_acc_t<accscalar_t, 3>(
        sycl::range<3>(
            (size_t)cfg.block_row,
            (size_t)cfg.workgroup_size,
            (size_t)vec_size),
        cgh);
  }

  GammaBetaBackwardSimpleKernelFunctor(
      const mean_t* mean_data_,
      const mean_t* var_data_,
      NormConfig cfg_,
      const scalar_t* dY_data_,
      const scalar_t* X_data_,
      weight_t* dg_data_,
      weight_t* db_data_)
      : mean_data(mean_data_),
        var_data(var_data_),
        cfg(cfg_),
        dY_data(dY_data_),
        X_data(X_data_),
        dg_data(dg_data_),
        db_data(db_data_),
        local_sum1(),
        local_sum2() {}

 private:
  const mean_t* mean_data;
  const mean_t* var_data;
  NormConfig cfg;
  const scalar_t* dY_data;
  const scalar_t* X_data;
  weight_t* dg_data;
  weight_t* db_data;
  sycl_local_acc_t<accscalar_t, 3> local_sum1;
  sycl_local_acc_t<accscalar_t, 3> local_sum2;
};

template <
    typename scalar_t,
    typename accscalar_t,
    typename mean_t,
    typename weight_t,
    int vec_size>
void vec_gamma_beta_bwd_simple_kernel(
    const Tensor& dY,
    const Tensor& X,
    const mean_t* mean_data,
    const mean_t* var_data,
    Tensor& dgamma,
    Tensor& dbeta,
    NormConfig& cfg) {
  const scalar_t* dY_data = dY.const_data_ptr<scalar_t>();
  const scalar_t* X_data = X.const_data_ptr<scalar_t>();
  weight_t* dg_data = dgamma.defined() ? dgamma.data_ptr<weight_t>() : nullptr;
  weight_t* db_data = dbeta.defined() ? dbeta.data_ptr<weight_t>() : nullptr;

  using vec_t = aligned_vector<scalar_t, vec_size>;
  using weight_vec_t = aligned_vector<weight_t, vec_size>;

  sycl::range<3> local_range{
      1, (size_t)cfg.block_row, (size_t)cfg.workgroup_size};
  sycl::range<3> global_range{
      (size_t)cfg.workgroup_num,
      (size_t)cfg.block_row,
      (size_t)cfg.workgroup_size};

  GammaBetaBackwardSimpleKernelFunctor<
      scalar_t,
      accscalar_t,
      mean_t,
      weight_t,
      vec_size,
      vec_t,
      weight_vec_t>
      kfn(mean_data, var_data, cfg, dY_data, X_data, dg_data, db_data);

  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), kfn);
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename mean_t,
    typename weight_t>
void gamma_beta_bwd_simple_kernel(
    const Tensor& dY,
    const Tensor& X,
    const mean_t* mean_data,
    const mean_t* var_data,
    Tensor& dgamma,
    Tensor& dbeta,
    NormConfig& config) {
#define VECTORIZE_KERNEL(vec_size)                                  \
  vec_gamma_beta_bwd_simple_kernel<                                 \
      scalar_t,                                                     \
      accscalar_t,                                                  \
      mean_t,                                                       \
      weight_t,                                                     \
      vec_size>(dY, X, mean_data, var_data, dgamma, dbeta, config); \
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

template <typename scalar_t, typename mean_t, typename weight_t>
void _layer_norm_backward_kernel(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor& dX,
    Tensor& dgamma,
    Tensor& dbeta,
    std::array<bool, 3> grad_input_mask) {
  TORCH_CHECK(dY.numel() == M * N);
  TORCH_CHECK(mean.numel() == M);
  TORCH_CHECK(rstd.numel() == M);

  using accscalar_t = acc_type_device<scalar_t, kXPU>;
  const mean_t* mean_data = mean.const_data_ptr<mean_t>();
  const mean_t* var_data = rstd.const_data_ptr<mean_t>();
  const weight_t* gamma_data =
      gamma.defined() ? gamma.const_data_ptr<weight_t>() : nullptr;

  if (grad_input_mask[0]) {
    // backward data
    const scalar_t* X_data = X.const_data_ptr<scalar_t>();
    const scalar_t* dY_data = dY.const_data_ptr<scalar_t>();
    scalar_t* dX_data = dX.data_ptr<scalar_t>();

    auto config = NormConfig(M, N, 1, sizeof(scalar_t));
    bool can_use_32bit_index = canUse32BitIndexMath(X) &&
        canUse32BitIndexMath(dY) && canUse32BitIndexMath(dX);
    if (config.workgroup_num_foreach == 1) {
      LayerNormBackward<scalar_t, mean_t, weight_t> norm(
          X_data, dY_data, dX_data, mean_data, var_data, gamma_data, M, N);
      vectorized_fused_norm_kernel<
          scalar_t,
          mean_t,
          weight_t,
          LayerNormBackward>(norm, config, can_use_32bit_index);
    } else {
      const auto kAccType =
          (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
          ? kFloat
          : X.scalar_type();
      Tensor a = at::empty({M}, X.options().dtype(kAccType));
      Tensor b = at::empty({M}, X.options().dtype(kAccType));
      accscalar_t* a_data = a.data_ptr<accscalar_t>();
      accscalar_t* b_data = b.data_ptr<accscalar_t>();

      LayerNormBackward<scalar_t, mean_t, weight_t> norm(
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
      config.template init_global_reduce<accscalar_t>(
          X, semaphores, scratchpad);
      rowwise_moments_kernel<scalar_t, mean_t, weight_t, LayerNormBackward>(
          norm, config, can_use_32bit_index);
      norm_update_kernel<scalar_t, mean_t, weight_t, LayerNormBackward>(
          norm, config, can_use_32bit_index);
    }
  }

  auto config_w = NormConfig(M, N, 0, sizeof(scalar_t));
  gamma_beta_bwd_simple_kernel<scalar_t, accscalar_t, mean_t, weight_t>(
      dY, X, mean_data, var_data, dgamma, dbeta, config_w);
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
        _layer_norm_kernel<scalar_t, acc_t>(
            X, gamma, beta, M, N, static_cast<acc_t>(eps), Y, mean, rstd);
      });
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_kernel(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor& dX,
    Tensor& dgamma,
    Tensor& dbeta,
    std::array<bool, 3> grad_input_mask) {
  if (M > 0 && N > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        X.scalar_type(),
        "layer_norm_backward_xpu",
        [&]() {
          using accscalar_t = acc_type_device<scalar_t, kXPU>;
          _layer_norm_backward_kernel<scalar_t, accscalar_t, scalar_t>(
              dY.contiguous(),
              X,
              mean,
              rstd,
              gamma,
              M,
              N,
              dX,
              dgamma,
              dbeta,
              grad_input_mask);
        });
  }

  return std::make_tuple(dX, dgamma, dbeta);
}

} // namespace xpu
} // namespace native
} // namespace at
