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
struct GNRowwiseMomentsKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
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

    if (item.get_group_range(0) <= SIMD) {
      val = SubgroupReduce<WelfordType, WelfordOp, SIMD>(val, welford_op);
    } else {
      val = GroupReduce<WelfordType, WelfordOp, SIMD>(
          item,
          val,
          welford_op,
          /*identity_element=*/WelfordType(0, 0, 0, 0),
          shared_);
    }

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

  GNRowwiseMomentsKernelFunctor(int64_t N, T eps, const T* X, T* mean, T* rstd)
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
struct ComputeFusedParamsKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    const int64_t index = item.get_global_linear_id();
    if (index < N_ * C_) {
      const int64_t ng = index / (C_ / group_);
      const int64_t c = index % C_;
      const T_ACC scale = (gamma_ == nullptr)
          ? static_cast<T_ACC>(rstd_[ng])
          : static_cast<T_ACC>(rstd_[ng]) * static_cast<T_ACC>(gamma_[c]);
      a_[index] = scale;
      b_[index] = -scale * static_cast<T_ACC>(mean_[ng]) +
          ((beta_ == nullptr) ? 0 : static_cast<T_ACC>(beta_[c]));
    }
  }
  ComputeFusedParamsKernelFunctor(
      int64_t N,
      int64_t C,
      int64_t group,
      const T* mean,
      const T* rstd,
      const T* gamma,
      const T* beta,
      T_ACC* a,
      T_ACC* b)
      : N_(N),
        C_(C),
        group_(group),
        mean_(mean),
        rstd_(rstd),
        gamma_(gamma),
        beta_(beta),
        a_(a),
        b_(b) {}

 private:
  int64_t N_;
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
    return a * static_cast<T_ACC>(x) + b;
  }
};

template <typename T>
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
  constexpr int group_reduce_wg_size = 512;
  auto caller = GNRowwiseMomentsKernelFunctor<T, SIMD>(
      D * HxW, eps, X_data, mean_data, rstd_data);
  const int64_t wg_size =
      D * HxW < group_reduce_wg_size ? SIMD : group_reduce_wg_size;
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
    const int64_t nwg =
        (N * C + group_reduce_wg_size - 1) / group_reduce_wg_size;
    const int64_t wg_size = group_reduce_wg_size;
    auto caller = ComputeFusedParamsKernelFunctor<T, T_ACC>(
        N, C, G, mean_data, rstd_data, gamma_data, beta_data, a_data, b_data);
    sycl_kernel_submit(
        sycl::range<1>(nwg * wg_size), sycl::range<1>(wg_size), queue, caller);

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

} // namespace at::native::xpu
