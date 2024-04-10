#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/Loops.h>
#include <aten/sycl/Norm.h>
#include <comm/SYCLContext.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t, typename mean_t, typename weight_t>
class LayerNormForward : public NormForward<scalar_t, mean_t, weight_t> {
 public:
  using accscalar_t = acc_type<scalar_t, true>;
  typedef NormForward<scalar_t, mean_t, weight_t> NF;
  LayerNormForward() = delete;
  LayerNormForward(
      scalar_t* X_data,
      scalar_t* Y_data,
      mean_t* mean_data,
      mean_t* var_data,
      weight_t* gamma_data,
      weight_t* beta_data,
      accscalar_t eps,
      int64_t M,
      int64_t N)
      : NormForward<scalar_t, mean_t, weight_t>(
            X_data,
            Y_data,
            mean_data,
            var_data,
            gamma_data,
            beta_data,
            eps),
        M(M),
        N(N) {
    numel = M * N;
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
    auto group_id = item_id.get_group(0);
    auto group_id_foreach = item_id.get_group(1);
    auto local_id = item_id.get_local_id(2);

    index_t group_offset = group_id * cfg.Plane;
    if (cfg.workgroup_num_foreach == 1) {
      if (local_id == 0) {
        NF::reduce_project(item_id, sum1, sum2, cfg);
      }
      item_id.barrier(sycl_global_fence);
    }

    mean_t mean_val = NF::mean_data[group_id];
    mean_t var_val = NF::var_data[group_id];
    for (index_t j = local_id * vec_size; j < (index_t)cfg.WGPlane;
         j += cfg.workgroup_size * vec_size) {
      index_t plane_offset = group_id_foreach * cfg.WGPlane + j;
      if (plane_offset < (index_t)cfg.Plane) {
        vec_t X_val = *(
            reinterpret_cast<vec_t*>(NF::X_data + group_offset + plane_offset));
        weight_vec_t gamma_val, beta_val;
        vec_t Y_val;
        if (NF::gamma_data != nullptr) {
          gamma_val =
              *(reinterpret_cast<weight_vec_t*>(NF::gamma_data + plane_offset));
        }
        if (NF::beta_data != nullptr) {
          beta_val =
              *(reinterpret_cast<weight_vec_t*>(NF::beta_data + plane_offset));
        }

        for (int v = 0; v < vec_size; ++v) {
          if (NF::gamma_data != nullptr && NF::beta_data != nullptr) {
            Y_val[v] = static_cast<accscalar_t>(gamma_val[v]) *
                    (var_val * static_cast<accscalar_t>(X_val[v] - mean_val)) +
                static_cast<accscalar_t>(beta_val[v]);
          } else if (NF::gamma_data != nullptr) {
            Y_val[v] = static_cast<accscalar_t>(gamma_val[v]) *
                (var_val * static_cast<accscalar_t>(X_val[v] - mean_val));
          } else if (NF::beta_data != nullptr) {
            Y_val[v] =
                (var_val * static_cast<accscalar_t>(X_val[v] - mean_val)) +
                static_cast<accscalar_t>(beta_val[v]);
          } else {
            Y_val[v] =
                (var_val * static_cast<accscalar_t>(X_val[v] - mean_val));
          }
        }
        *(reinterpret_cast<vec_t*>(NF::Y_data + group_offset + plane_offset)) =
            Y_val;
      }
    }
  };

  int64_t M;
  int64_t N;
  int64_t numel;
};

template <typename scalar_t, typename mean_t, typename weight_t>
class LayerNormBackward : public NormBackward<scalar_t, weight_t, weight_t> {
 public:
  using accscalar_t = acc_type<scalar_t, true>;
  LayerNormBackward() = delete;
  LayerNormBackward(
      scalar_t* X_data,
      scalar_t* dY_data,
      scalar_t* dX_data,
      mean_t* mean_data,
      mean_t* var_data,
      weight_t* gamma_data,
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
      scalar_t* X_data,
      scalar_t* dY_data,
      scalar_t* dX_data,
      mean_t* mean_data,
      mean_t* var_data,
      weight_t* gamma_data,
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
    index_t group_offset = group_id * cfg.Plane;

    mean_t mean_val = NB::mean_data[group_id];
    mean_t rstd_val = NB::var_data[group_id];
    for (index_t j = local_id * vec_size; j < (index_t)cfg.WGPlane;
         j += cfg.workgroup_size * vec_size) {
      index_t plane_offset = group_id_foreach * cfg.WGPlane + j;
      if (plane_offset < (index_t)cfg.Plane) {
        weight_vec_t gamma_val;
        if (NB::gamma_data != nullptr) {
          gamma_val =
              *(reinterpret_cast<weight_vec_t*>(NB::gamma_data + plane_offset));
        }
        vec_t dY_val = *(reinterpret_cast<vec_t*>(
            NB::dY_data + group_offset + plane_offset));
        vec_t X_val = *(
            reinterpret_cast<vec_t*>(NB::X_data + group_offset + plane_offset));
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

    index_t group_offset = group_id * cfg.Plane;
    mean_t mean_val = NB::mean_data[group_id];
    mean_t var_val = NB::var_data[group_id];

    int fH = cfg.Plane;
    accscalar_t term1 = (accscalar_t(1) / fH) * var_val;
    for (index_t j = local_id * vec_size; j < (index_t)cfg.WGPlane;
         j += cfg.workgroup_size * vec_size) {
      index_t plane_offset = group_id_foreach * cfg.WGPlane + j;
      if (plane_offset < (index_t)cfg.Plane) {
        vec_t dY_val = *(reinterpret_cast<vec_t*>(
            NB::dY_data + group_offset + plane_offset));
        vec_t X_val = *(
            reinterpret_cast<vec_t*>(NB::X_data + group_offset + plane_offset));
        weight_vec_t gamma_val;
        if (NB::gamma_data != nullptr) {
          gamma_val =
              *(reinterpret_cast<weight_vec_t*>(NB::gamma_data + plane_offset));
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

template <typename scalar_t, typename mean_t, typename weight_t>
void LayerNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    acc_type<scalar_t, true> eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  TORCH_CHECK(X.numel() == M * N);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == N);
  TORCH_CHECK(!beta.defined() || beta.numel() == N);

  scalar_t* X_data = X.data_ptr<scalar_t>();
  scalar_t* Y_data = Y.data_ptr<scalar_t>();
  mean_t* mean_data = mean.data_ptr<mean_t>();
  mean_t* var_data = rstd.data_ptr<mean_t>();
  weight_t* gamma_data = gamma.defined() ? gamma.data_ptr<weight_t>() : nullptr;
  weight_t* beta_data = beta.defined() ? beta.data_ptr<weight_t>() : nullptr;

  auto config = NormConfig(M, N, 1, sizeof(scalar_t));
  bool can_use_32bit_index = canUse32BitIndexMath(X);
  LayerNormForward<scalar_t, mean_t, weight_t> layer_norm_forward(
      X_data, Y_data, mean_data, var_data, gamma_data, beta_data, eps, M, N);

  if (config.workgroup_num_foreach == 1) {
    launch_vectorized_fused_norm_kernel<
        scalar_t,
        mean_t,
        weight_t,
        LayerNormForward>(layer_norm_forward, config, can_use_32bit_index);
  } else {
    Tensor semaphores, scratchpad;
    config.template init_global_reduce<scalar_t>(X, semaphores, scratchpad);
    RowwiseMomentsSYCLKernelImpl<scalar_t, mean_t, weight_t, LayerNormForward>(
        layer_norm_forward, config, can_use_32bit_index);
    NormUpdateKernelImpl<scalar_t, mean_t, weight_t, LayerNormForward>(
        layer_norm_forward, config, can_use_32bit_index);
  }
}

void LayerNormKernelImpl(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    double eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "LayerNormKernelImpl",
      [&]() {
        if (gamma.scalar_type() == kFloat) {
          mean = at::empty({M}, X.options().dtype(kFloat));
          rstd = at::empty({M}, X.options().dtype(kFloat));
          LayerNormKernelImplInternal<scalar_t, float, float>(
              X,
              gamma,
              beta,
              M,
              N,
              static_cast<acc_type<scalar_t, true>>(eps),
              Y,
              mean,
              rstd);
        } else {
          mean = at::empty({M}, X.options());
          rstd = at::empty({M}, X.options());
          LayerNormKernelImplInternal<scalar_t, scalar_t, scalar_t>(
              X,
              gamma,
              beta,
              M,
              N,
              static_cast<acc_type<scalar_t, true>>(eps),
              Y,
              mean,
              rstd);
        }
      });
}

} // namespace xpu
} // namespace native
} // namespace at
