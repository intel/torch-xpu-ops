#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/GroupUtils.h>
#include <aten/sycl/Loops.h>
#include <comm/MemoryFormat.h>
#include <comm/XPUMathCompat.h>

namespace at::native::xpu {

template <
    typename scalar_t,
    typename acc_scalar_t,
    typename index_t,
    typename res_t>
struct WelfordOps {
  sycl::nd_item<1>& item;
  acc_scalar_t correction;
  bool take_sqrt;

 public:
  using acc_t = at::native::WelfordData<acc_scalar_t, index_t>;
  inline acc_t reduce(acc_t acc, scalar_t data, index_t /*idx*/) const {
    // We accumulate n in index_t to avoid cumulative rounding error, but still
    // need nf for use in combine where int32 may overflow.
    index_t new_n = acc.n + 1;
    acc_scalar_t new_nf = static_cast<acc_scalar_t>(new_n);
    acc_scalar_t delta = data - acc.mean;
    acc_scalar_t new_mean = acc.mean + delta / new_nf;
    acc_scalar_t new_delta = data - new_mean;
    return {
        new_mean,
        acc.m2 + delta * new_delta,
        new_n,
        new_nf,
    };
  }
  inline acc_t combine(acc_t a, acc_t b) const {
    if (a.nf == 0) {
      return b;
    }
    if (b.nf == 0) {
      return a;
    }
    acc_scalar_t delta = b.mean - a.mean;
    acc_scalar_t new_count = a.nf + b.nf;
    acc_scalar_t nb_over_n = b.nf / new_count;
    return {
        a.mean + delta * nb_over_n,
        a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
        // setting acc.n as -1 since acc.n might not be able to represent the
        // count correctly within its range, setting it to -1 to avoid confusion
        -1,
        new_count};
  }
  inline res_t project(acc_t acc) const __ubsan_ignore_float_divide_by_zero__ {
    const auto mean = static_cast<scalar_t>(acc.mean);
    const auto divisor = acc.nf > correction ? acc.nf - correction : 0;
    const auto var = acc.m2 / divisor;
    res_t results(take_sqrt ? device_sqrt(var) : var, mean);
    return results;
  }

  static acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

  inline acc_t shfl_down(acc_t acc, int offset) const {
    auto sg = item.get_sub_group();
    return {
        sg.shuffle_down(acc.mean, offset),
        sg.shuffle_down(acc.m2, offset),
        sg.shuffle_down(acc.n, offset),
        sg.shuffle_down(acc.nf, offset)};
  }

  WelfordOps(sycl::nd_item<1>& item, acc_scalar_t correction, bool take_sqrt)
      : item(item), correction(correction), take_sqrt(take_sqrt) {}
};

template <typename T, int SIMD>
struct GNRowwiseMomentsFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  using T_ACC = acc_type<T, true>;
  using WelfordType = at::native::WelfordData<T_ACC, int64_t>;
  using WelfordOp = at::native::xpu::
      WelfordOps<T_ACC, T_ACC, int64_t, std::pair<T_ACC, T_ACC>>;

  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<1> item) const {
    const int64_t i = item.get_group(0);
    WelfordOp welford_op = {item, /*correction=*/0, /*take_sqrt=*/false};
    WelfordType val(0, 0, 0, 0);
    for (int64_t j = item.get_local_id(0); j < N_;
         j += item.get_local_range(0)) {
      const int64_t index = i * N_ + j;
      val = welford_op.reduce(val, static_cast<T_ACC>(X_[index]), index);
    }

    val = GroupReduce<WelfordType, WelfordOp, SIMD>(
        item, val, welford_op, shared_);

    if (item.get_local_id(0) == 0) {
      T_ACC m1;
      T_ACC m2;
      std::tie(m2, m1) = welford_op.project(val);
      mean_[i] = m1;
      rstd_[i] = c10::xpu::compat::rsqrt(m2 + static_cast<T_ACC>(eps_));
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<WelfordType>(SIMD, cgh);
  }

  GNRowwiseMomentsFunctor(int64_t N, T eps, const T* X, T* mean, T* rstd)
      : N_(N), eps_(eps), X_(X), mean_(mean), rstd_(rstd) {}

 private:
  int64_t N_;
  T eps_;
  const T* X_;
  T* mean_;
  T* rstd_;
  sycl_local_acc_t<WelfordType> shared_;
};

template <typename T, typename T_ACC>
struct ComputeFusedParamsFunctor {
  void operator()(sycl::item<1> item) const {
    auto index = item.get_id(0);
    const int64_t ng = index / (C_ / group_);
    const int64_t c = index % C_;
    const T_ACC scale = (gamma_ == nullptr)
        ? static_cast<T_ACC>(rstd_[ng])
        : static_cast<T_ACC>(rstd_[ng]) * static_cast<T_ACC>(gamma_[c]);
    a_[index] = scale;
    b_[index] = -scale * static_cast<T_ACC>(mean_[ng]) +
        ((beta_ == nullptr) ? T_ACC(0) : static_cast<T_ACC>(beta_[c]));
  }
  ComputeFusedParamsFunctor(
      int64_t C,
      int64_t group,
      const T* mean,
      const T* rstd,
      const T* gamma,
      const T* beta,
      T_ACC* a,
      T_ACC* b)
      : C_(C),
        group_(group),
        mean_(mean),
        rstd_(rstd),
        gamma_(gamma),
        beta_(beta),
        a_(a),
        b_(b) {}

 private:
  int64_t C_;
  int64_t group_;
  const T* mean_;
  const T* rstd_;
  const T* gamma_;
  const T* beta_;
  T_ACC* a_;
  T_ACC* b_;
};

template <typename T, typename T_ACC>
struct GroupNorm1dGammaBetaFunctor {
  T operator()(T x, T mean, T rstd, T gamma, T beta) const {
    return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
        static_cast<T_ACC>(rstd) * static_cast<T_ACC>(gamma) +
        static_cast<T_ACC>(beta);
  }
};

template <typename T, typename T_ACC>
struct GroupNorm1dGammaFunctor {
  T operator()(T x, T mean, T rstd, T gamma) const {
    return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
        static_cast<T_ACC>(rstd) * static_cast<T_ACC>(gamma);
  }
};

template <typename T, typename T_ACC>
struct GroupNorm1dBetaFunctor {
  T operator()(T x, T mean, T rstd, T beta) const {
    return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
        static_cast<T_ACC>(rstd) +
        static_cast<T_ACC>(beta);
  }
};

template <typename T, typename T_ACC>
struct GroupNorm1dFunctor {
  T operator()(T x, T mean, T rstd) const {
    return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
        static_cast<T_ACC>(rstd);
  }
};

template <typename T>
void group_norm_1d_forward(
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t group,
    Tensor& Y) {
  using T_ACC = acc_type<T, true>;
  const int64_t G = group;
  const int64_t D = C / G;
  if (gamma.defined() && beta.defined()) {
    auto iter = TensorIteratorConfig()
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N, G, D}))
                    .add_owned_const_input(X.view({N, G, D}))
                    .add_owned_input(mean.view({N, G, 1}))
                    .add_owned_input(rstd.view({N, G, 1}))
                    .add_owned_const_input(gamma.view({1, G, D}))
                    .add_owned_const_input(beta.view({1, G, D}))
                    .build();
    gpu_kernel(iter, GroupNorm1dGammaBetaFunctor<T, T_ACC>());
  } else if (gamma.defined()) {
    auto iter = TensorIteratorConfig()
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N, G, D}))
                    .add_owned_const_input(X.view({N, G, D}))
                    .add_owned_input(mean.view({N, G, 1}))
                    .add_owned_input(rstd.view({N, G, 1}))
                    .add_owned_const_input(gamma.view({1, G, D}))
                    .build();
    gpu_kernel(iter, GroupNorm1dGammaFunctor<T, T_ACC>());
  } else if (beta.defined()) {
    auto iter = TensorIteratorConfig()
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N, G, D}))
                    .add_owned_const_input(X.view({N, G, D}))
                    .add_owned_input(mean.view({N, G, 1}))
                    .add_owned_input(rstd.view({N, G, 1}))
                    .add_owned_const_input(beta.view({1, G, D}))
                    .build();
    gpu_kernel(iter, GroupNorm1dBetaFunctor<T, T_ACC>());
  } else {
    auto iter = TensorIteratorConfig()
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N * G, D}))
                    .add_owned_const_input(X.view({N * G, D}))
                    .add_owned_input(mean.view({N * G, 1}))
                    .add_owned_input(rstd.view({N * G, 1}))
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
void group_norm_kernel_impl(
    const Tensor& X_,
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
  auto X = X_.contiguous();

  using T_ACC = acc_type<T, true>;
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
  constexpr int SIMD = 32; // TODO: cross-platform support
  auto caller = GNRowwiseMomentsFunctor<T, SIMD>(
      D * HxW, eps, X_data, mean_data, rstd_data);
  const int64_t wg_size =
      D * HxW < GROUP_REDUCE_WORK_SIZE ? SIMD : GROUP_REDUCE_WORK_SIZE;
  int64_t nwg = N * G;
  sycl_kernel_submit(
      sycl::range<1>(nwg * wg_size), sycl::range<1>(wg_size), queue, caller);

  if (HxW == 1) {
    group_norm_1d_forward<T>(X, mean, rstd, gamma, beta, N, C, G, Y);
  } else if (!gamma.defined() && !beta.defined()) {
    auto iter = TensorIteratorConfig()
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N * G, D * HxW}))
                    .add_owned_const_input(X.view({N * G, D * HxW}))
                    .add_owned_input(mean.view({N * G, 1}))
                    .add_owned_input(rstd.view({N * G, 1}))
                    .build();
    gpu_kernel(iter, GroupNorm1dFunctor<T, T_ACC>());
  } else {
    const auto kAccType =
        (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
        ? kFloat
        : X.scalar_type();
    Tensor a = at::empty({N, C}, X.options().dtype(kAccType));
    Tensor b = at::empty({N, C}, X.options().dtype(kAccType));
    const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
    const T* beta_data = beta.defined() ? beta.const_data_ptr<T>() : nullptr;
    T_ACC* a_data = a.mutable_data_ptr<T_ACC>();
    T_ACC* b_data = b.mutable_data_ptr<T_ACC>();

    // TODO: Since there is some issues in gpu_kernel_multiple_outputs, we are
    // using manual kernel here. Make it using gpu_kernel_multiple_outputs once
    // the issue fixed.
    auto caller = ComputeFusedParamsFunctor<T, T_ACC>(
        C, G, mean_data, rstd_data, gamma_data, beta_data, a_data, b_data);
    sycl_kernel_submit(sycl::range<1>(N * C), queue, caller);

    auto iter = TensorIteratorConfig()
                    .check_all_same_dtype(std::is_same<T, T_ACC>::value)
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
    Tensor& rstd,
    ScalarType dtype) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "group_norm_kernel_xpu",
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
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<2> item) const {
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
    sum1 = GroupReduceSum<T_ACC, SIMD>(item, sum1, ds_shared_);
    sum2 = GroupReduceSum<T_ACC, SIMD>(item, sum2, db_shared_);
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
    ds_shared_ = sycl_local_acc_t<T_ACC>(GROUP_REDUCE_WORK_SIZE, cgh);
    db_shared_ = sycl_local_acc_t<T_ACC>(GROUP_REDUCE_WORK_SIZE, cgh);
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
    using T_ACC = acc_type<T, true>;
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
  using T_ACC = acc_type<T, true>;

  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<2> item) const {
    const int64_t c =
        item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
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
    item.barrier(sycl_local_fence);

    // Do subgroup reduce for the 1st 16 cols in the tile.
    T_ACC sum1 = g_shared_[tid_x][tid_y];
    T_ACC sum2 = b_shared_[tid_x][tid_y];
    sum1 = SubgroupReduceSum<T_ACC, SIMD>(item, sum1);
    sum2 = SubgroupReduceSum<T_ACC, SIMD>(item, sum2);
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
    sum1 = SubgroupReduceSum<T_ACC, SIMD>(item, sum1);
    sum2 = SubgroupReduceSum<T_ACC, SIMD>(item, sum2);
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
  using T_ACC = acc_type<T, true>;
  const int64_t G = group;
  const int64_t D = C / G;
  const T* dY_data = dY.const_data_ptr<T>();
  const T* X_data = X.const_data_ptr<T>();
  const T* mean_data = mean.const_data_ptr<T>();
  const T* rstd_data = rstd.const_data_ptr<T>();

  auto& queue = getCurrentSYCLQueue();
  constexpr int SIMD = 32; // TODO: cross-platform support

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

    const int64_t wg_size =
        (C / G) < GROUP_REDUCE_WORK_SIZE ? SIMD : GROUP_REDUCE_WORK_SIZE;
    auto caller = Compute1dBackwardFusedParamsFunctor<T, T_ACC, SIMD>(
        C,
        G,
        dY_data,
        X_data,
        mean_data,
        rstd_data,
        gamma_data,
        c2_data,
        c3_data);
    auto global_range = sycl::range<2>(G, N * wg_size);
    auto local_range = sycl::range<2>(1, wg_size);
    sycl_kernel_submit(global_range, local_range, queue, caller);

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
      const int64_t wg_size = GROUP_REDUCE_WORK_SIZE;
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
      const int kReduceTileSize = 32;
      const int64_t B = (C + kReduceTileSize - 1) / kReduceTileSize;
      // The algorithm for colwise reduction here is to accumulate each 32 cols
      // to a 32 * 32 tile and write the tile to shared memory. Then do subgroup
      // reduce for each col in the tile. So here the group size must be (32,
      // 16).
      auto global_range =
          sycl::range<2>(kReduceTileSize / 2, B * kReduceTileSize);
      auto local_range = sycl::range<2>(kReduceTileSize / 2, kReduceTileSize);
      auto caller = GammaBeta1dBackwardLargeKernel<T, SIMD, kReduceTileSize>(
          N,
          C,
          G,
          dY_data,
          X_data,
          mean_data,
          rstd_data,
          dgamma_data,
          dbeta_data);
      sycl_kernel_submit(global_range, local_range, queue, caller);
    }
  }
}

template <typename T, int SIMD>
struct ComputeInternalGradientsFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  using T_ACC = acc_type<T, true>;

  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<1> item) const {
    const int64_t nc = item.get_group(0);
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t hw = item.get_local_id(0); hw < HxW_;
         hw += item.get_local_range(0)) {
      const int64_t index = nc * HxW_ + hw;
      sum1 += static_cast<T_ACC>(dY_[index]) * static_cast<T_ACC>(X_[index]);
      sum2 += static_cast<T_ACC>(dY_[index]);
    }
    sum1 = GroupReduceSum<T_ACC, SIMD>(item, sum1, ds_shared_);
    sum2 = GroupReduceSum<T_ACC, SIMD>(item, sum2, db_shared_);
    if (item.get_local_id(0) == 0) {
      ds_[nc] = sum1;
      db_[nc] = sum2;
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    ds_shared_ = sycl_local_acc_t<T_ACC>(GROUP_REDUCE_WORK_SIZE, cgh);
    db_shared_ = sycl_local_acc_t<T_ACC>(GROUP_REDUCE_WORK_SIZE, cgh);
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
  using T_ACC = acc_type<T, true>;

  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<2> item) const {
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
    sum1 = GroupReduceSum<T_ACC, SIMD>(item, sum1, ds_shared_);
    sum2 = GroupReduceSum<T_ACC, SIMD>(item, sum2, db_shared_);
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
    ds_shared_ = sycl_local_acc_t<T_ACC>(GROUP_REDUCE_WORK_SIZE, cgh);
    db_shared_ = sycl_local_acc_t<T_ACC>(GROUP_REDUCE_WORK_SIZE, cgh);
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
  using T_ACC = acc_type<T, true>;

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
  using T_ACC = acc_type<T, true>;

  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<2> item) const {
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
    item.barrier(sycl_local_fence);

    // Do subgroup reduce for the 1st 16 cols in the tile.
    T_ACC sum1 = g_shared_[tid_x][tid_y];
    T_ACC sum2 = b_shared_[tid_x][tid_y];
    sum1 = SubgroupReduceSum<T_ACC, SIMD>(item, sum1);
    sum2 = SubgroupReduceSum<T_ACC, SIMD>(item, sum2);
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
    sum1 = SubgroupReduceSum<T_ACC, SIMD>(item, sum1);
    sum2 = SubgroupReduceSum<T_ACC, SIMD>(item, sum2);
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

  using T_ACC = acc_type<T, true>;
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

  constexpr int SIMD = 32; // TODO: cross-platform support
  int64_t wg_size =
      HxW < GROUP_REDUCE_WORK_SIZE ? SIMD : GROUP_REDUCE_WORK_SIZE;
  auto caller = ComputeInternalGradientsFunctor<T, SIMD>(
      HxW, dY_data, X_data, ds_data, db_data);
  sycl_kernel_submit(
      sycl::range<1>(N * C * wg_size), sycl::range<1>(wg_size), queue, caller);

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

    wg_size = (C / G) < GROUP_REDUCE_WORK_SIZE ? SIMD : GROUP_REDUCE_WORK_SIZE;
    auto caller = ComputeBackwardFusedParamsFunctor<T, SIMD>(
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
    sycl_kernel_submit(
        sycl::range<2>(G, N * wg_size),
        sycl::range<2>(1, wg_size),
        queue,
        caller);

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
      const int kReduceTileSize = 32;
      const int64_t B = (C + kReduceTileSize - 1) / kReduceTileSize;
      // The algorithm for colwise reduction here is to accumulate each 32 cols
      // to a 32 * 32 tile and write the tile to shared memory. Then do subgroup
      // reduce for each col in the tile. So here the gorup size must be (32,
      // 16).
      auto global_range =
          sycl::range<2>(kReduceTileSize / 2, B * kReduceTileSize);
      auto local_range = sycl::range<2>(kReduceTileSize / 2, kReduceTileSize);
      auto caller = GammaBetaBackwardFunctor<T, SIMD, kReduceTileSize>(
          N,
          C,
          G,
          mean_data,
          rstd_data,
          ds_data,
          db_data,
          dgamma_data,
          dbeta_data);
      sycl_kernel_submit(global_range, local_range, queue, caller);
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
      "group_norm_backward_kernel_xpu",
      [&]() {
        group_norm_backward_kernel_impl<scalar_t>(
            dY, X, mean, rstd, gamma, N, C, HxW, group, dX, dgamma, dbeta);
      });
}

} // namespace at::native::xpu
