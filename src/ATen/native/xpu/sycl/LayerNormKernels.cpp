#include <ATen/Dispatch.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/Norm.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/LayerNormKernels.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t, typename mean_t, typename weight_t>
class LayerNormForward : public NormForward<scalar_t, mean_t, weight_t> {
 public:
  using accscalar_t = acc_type_device<scalar_t, kXPU>;
  typedef NormForward<scalar_t, mean_t, weight_t> NF;
  LayerNormForward() = delete;
  LayerNormForward(
      const scalar_t* X_data,
      scalar_t* Y_data,
      mean_t* mean_data,
      mean_t* var_data,
      const weight_t* gamma_data,
      const weight_t* beta_data,
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

    index_t group_offset = group_id * cfg.problem_size;
    if (cfg.workgroup_num_foreach == 1) {
      if (local_id == 0) {
        NF::reduce_project(item_id, sum1, sum2, cfg);
      }
      item_id.barrier(sycl_global_fence);
    }

    mean_t mean_val = NF::mean_data[group_id];
    mean_t var_val = NF::var_data[group_id];
    for (index_t j = local_id * vec_size; j < cfg.workgroup_work_size;
         j += cfg.workgroup_size * vec_size) {
      index_t plane_offset = group_id_foreach * cfg.workgroup_work_size + j;
      if (plane_offset < (index_t)cfg.problem_size) {
        vec_t X_val = *(reinterpret_cast<const vec_t*>(
            NF::X_data + group_offset + plane_offset));
        weight_vec_t gamma_val, beta_val;
        vec_t Y_val;
        if (NF::gamma_data != nullptr) {
          gamma_val = *(reinterpret_cast<const weight_vec_t*>(
              NF::gamma_data + plane_offset));
        }
        if (NF::beta_data != nullptr) {
          beta_val = *(reinterpret_cast<const weight_vec_t*>(
              NF::beta_data + plane_offset));
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

template <typename scalar_t, typename mean_t, typename weight_t>
void _layer_norm_kernel(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    acc_type_device<scalar_t, kXPU> eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  TORCH_CHECK(X.numel() == M * N);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == N);
  TORCH_CHECK(!beta.defined() || beta.numel() == N);

  const scalar_t* X_data = X.const_data_ptr<scalar_t>();
  scalar_t* Y_data = Y.data_ptr<scalar_t>();
  mean_t* mean_data = mean.data_ptr<mean_t>();
  mean_t* var_data = rstd.data_ptr<mean_t>();
  const weight_t* gamma_data =
      gamma.defined() ? gamma.const_data_ptr<weight_t>() : nullptr;
  const weight_t* beta_data =
      beta.defined() ? beta.const_data_ptr<weight_t>() : nullptr;

  auto config = NormConfig(M, N, 1, sizeof(scalar_t));
  bool can_use_32bit_index = canUse32BitIndexMath(X);
  LayerNormForward<scalar_t, mean_t, weight_t> norm(
      X_data, Y_data, mean_data, var_data, gamma_data, beta_data, eps, M, N);

  if (config.workgroup_num_foreach == 1) {
    vectorized_fused_norm_kernel<scalar_t, mean_t, weight_t, LayerNormForward>(
        norm, config, can_use_32bit_index);
  } else {
    Tensor semaphores, scratchpad;
    config.template init_global_reduce<scalar_t>(X, semaphores, scratchpad);
    rowwise_moments_kernel<scalar_t, mean_t, weight_t, LayerNormForward>(
        norm, config, can_use_32bit_index);
    norm_update_kernel<scalar_t, mean_t, weight_t, LayerNormForward>(
        norm, config, can_use_32bit_index);
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

std::tuple<Tensor, Tensor, Tensor> layer_norm_kernel(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    double eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  if (M > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        X.scalar_type(),
        "layer_norm_xpu",
        [&]() {
          using acc_t = acc_type_device<scalar_t, kXPU>;
          _layer_norm_kernel<scalar_t, acc_t, scalar_t>(
              X, gamma, beta, M, N, static_cast<acc_t>(eps), Y, mean, rstd);
        });
  }

  return std::make_tuple(Y, mean, rstd);
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
