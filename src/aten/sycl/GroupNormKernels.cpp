#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/Norm.h>
#include <comm/MemoryFormat.h>

namespace at::native::xpu {

template <typename scalar_t, typename mean_t, typename weight_t>
class GroupNormForward : public NormForward<scalar_t, mean_t, weight_t> {
 public:
  using accscalar_t = acc_type<scalar_t, true>;
  GroupNormForward() = delete;
  GroupNormForward(
      scalar_t* X_data,
      mean_t* mean_data,
      mean_t* var_data,
      weight_t* gamma_data,
      weight_t* beta_data,
      accscalar_t eps,
      int64_t N,
      int64_t C,
      int64_t group,
      int64_t HxW)
      : NormForward<scalar_t, mean_t, weight_t>(
            X_data,
            nullptr,
            mean_data,
            var_data,
            gamma_data,
            beta_data,
            eps),
        N_(N),
        C_(C),
        group_(group),
        HxW_(HxW) {
    numel_ = N_ * C_ * HxW_;
    D_ = C_ / group_;
  }
  typedef NormForward<scalar_t, mean_t, weight_t> NF;

  void set_eltwise_update_parameter(
      scalar_t* X_ptr,
      scalar_t* Y_ptr,
      accscalar_t* a_ptr,
      accscalar_t* b_ptr,
      bool is_channels_last) {
    NF::X_data = X_ptr;
    NF::Y_data = Y_ptr;
    a_data_ = a_ptr;
    b_data_ = b_ptr;
    channels_last_ = is_channels_last;
  }

  template <int vec_size, typename index_t, typename vec_t>
  void eltwise_update(index_t i) const {
    index_t remaining = numel_ - i * vec_size;
    if (remaining < vec_size) {
      for (int j = 0; j < remaining; ++j) {
        index_t offset = i * vec_size + j;

        index_t nc;
        if (channels_last_) {
          nc = offset / (C_ * HxW_) * C_ + offset % C_;
        } else {
          nc = offset / HxW_;
        }
        NF::Y_data[offset] = static_cast<scalar_t>(
            a_data_[nc] * static_cast<accscalar_t>(NF::X_data[offset]) +
            b_data_[nc]);
      }
    } else {
      index_t offset = i * vec_size;

      vec_t in_val = *(reinterpret_cast<vec_t*>(NF::X_data + offset));
      vec_t out_val;
#pragma unroll(vec_size)
      for (int v = 0; v < vec_size; ++v) {
        index_t nc;
        if (channels_last_) {
          nc = (offset + v) / (C_ * HxW_) * C_ + (offset + v) % C_;
        } else {
          nc = (offset + v) / HxW_;
        }
        out_val[v] = static_cast<scalar_t>(
            a_data_[nc] * static_cast<accscalar_t>(in_val[v]) + b_data_[nc]);
      }
      *(reinterpret_cast<vec_t*>(NF::Y_data + offset)) = out_val;
    }
  };

  int numel() const {
    return numel_;
  }

 private:
  int N_;
  int C_;
  int group_;
  int HxW_;
  int D_;
  int numel_;
  accscalar_t* a_data_;
  accscalar_t* b_data_;
  bool channels_last_;
};

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    typename accscalar_t>
struct ComputeFusedParamsKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    auto id = item_id.get_id(0);

    const int64_t ng = id / (C_ / group_);
    const int64_t c = id % C_;
    const accscalar_t x = (gamma_data_ == nullptr)
        ? static_cast<accscalar_t>(rstd_data_[ng])
        : static_cast<accscalar_t>(rstd_data_[ng]) *
            static_cast<accscalar_t>(gamma_data_[c]);
    a_data_[id] = x;
    b_data_[id] = -x * static_cast<accscalar_t>(mean_data_[ng]) +
        (beta_data_ == nullptr ? accscalar_t(0)
                               : static_cast<accscalar_t>(beta_data_[c]));
  }
  ComputeFusedParamsKernelFunctor(
      int64_t C,
      int64_t group,
      const mean_t* mean_data,
      const mean_t* rstd_data,
      const weight_t* gamma_data,
      const weight_t* beta_data,
      accscalar_t* a_data,
      accscalar_t* b_data)
      : C_(C),
        group_(group),
        mean_data_(mean_data),
        rstd_data_(rstd_data),
        gamma_data_(gamma_data),
        beta_data_(beta_data),
        a_data_(a_data),
        b_data_(b_data) {}

 private:
  int64_t C_;
  int64_t group_;
  const mean_t* mean_data_;
  const mean_t* rstd_data_;
  const weight_t* gamma_data_;
  const weight_t* beta_data_;
  accscalar_t* a_data_;
  accscalar_t* b_data_;
};

template <typename scalar_t, typename mean_t, typename weight_t>
void compute_fused_params_kernel(
    int64_t N,
    int64_t C,
    int64_t group,
    const mean_t* mean_data,
    const mean_t* rstd_data,
    const weight_t* gamma_data,
    const weight_t* beta_data,
    acc_type<scalar_t, true>* a_data,
    acc_type<scalar_t, true>* b_data) {
  using accscalar_t = acc_type<scalar_t, true>;
  auto& q = getCurrentSYCLQueue();
  auto global_range = sycl::range<1>(N * C);
  auto caller =
      ComputeFusedParamsKernelFunctor<scalar_t, mean_t, weight_t, accscalar_t>(
          C,
          group,
          mean_data,
          rstd_data,
          gamma_data,
          beta_data,
          a_data,
          b_data);
  sycl_kernel_submit(global_range, q, caller);
}

template <typename scalar_t, typename mean_t, typename weight_t>
void group_norm_kernel_impl(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    scalar_t eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  using accscalar_t = acc_type<scalar_t, true>;
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  TORCH_CHECK(!beta.defined() || beta.numel() == C);
  if (N == 0) {
    return;
  }
  Tensor X_cont = X.contiguous();
  scalar_t* X_data = X_cont.data_ptr<scalar_t>();

  mean_t* mean_data = mean.data_ptr<mean_t>();
  mean_t* rstd_data = rstd.data_ptr<mean_t>();
  weight_t* gamma_data = gamma.defined() ? gamma.data_ptr<weight_t>() : nullptr;
  weight_t* beta_data = beta.defined() ? beta.data_ptr<weight_t>() : nullptr;

  auto config = NormConfig(N * group, C / group * HxW, 1, sizeof(scalar_t));
  GroupNormForward<scalar_t, mean_t, weight_t> group_norm_forward(
      X_data,
      mean_data,
      rstd_data,
      gamma_data,
      beta_data,
      eps,
      N,
      C,
      group,
      HxW);
  bool can_use_32bit_index = canUse32BitIndexMath(X);
  Tensor semaphores, scratchpad;
  config.template init_global_reduce<scalar_t>(X, semaphores, scratchpad);
  rowwise_moments_kernel<scalar_t, mean_t, weight_t, GroupNormForward>(
      group_norm_forward, config, can_use_32bit_index);

  const auto kAccType =
      (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
      ? kFloat
      : X.scalar_type();
  Tensor a = at::empty({N, C}, X.options().dtype(kAccType));
  Tensor b = at::empty({N, C}, X.options().dtype(kAccType));
  accscalar_t* a_data = a.data_ptr<accscalar_t>();
  accscalar_t* b_data = b.data_ptr<accscalar_t>();
  compute_fused_params_kernel<scalar_t, mean_t, weight_t>(
      N, C, group, mean_data, rstd_data, gamma_data, beta_data, a_data, b_data);

  // propagate channels_last format from X to Y
  bool is_channels_last = is_smf_channels_last(X);
  if (is_channels_last) {
    X_cont = X;
    auto smf = X.suggest_memory_format();
    Y = at::empty_like(X, smf);
  } else {
    Y = at::empty_like(X_cont);
  }
  group_norm_forward.set_eltwise_update_parameter(
      X_cont.data_ptr<scalar_t>(),
      Y.data_ptr<scalar_t>(),
      a_data,
      b_data,
      is_channels_last);
  norm_eltwise_update_kernel<scalar_t, mean_t, weight_t, GroupNormForward>(
      group_norm_forward, config, can_use_32bit_index);
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
        if (dtype == kFloat) {
          group_norm_kernel_impl<scalar_t, float, float>(
              X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
        } else {
          group_norm_kernel_impl<scalar_t, scalar_t, scalar_t>(
              X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
        }
      });
}

} // namespace at::native::xpu
