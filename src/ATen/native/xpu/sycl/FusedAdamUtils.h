#pragma once

#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>

#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>

#include <comm/SYCLHelpers.h>

namespace at::native::xpu {

enum class ADAM_MODE : uint8_t { ORIGINAL = 0, ADAMW = 1 };

// index in TensorList for params
constexpr uint8_t kParamIdx = 0;
constexpr uint8_t kGradIdx = 1;
constexpr uint8_t kExpAvgIdx = 2;
constexpr uint8_t kExpAvgSqIdx = 3;
constexpr uint8_t kMaxExpAvgSqIdx = 4;

template <
    typename scalar_type,
    typename opmath_t,
    int depth,
    ADAM_MODE adam_mode,
    bool amsgrad>
inline void adam_math(
    scalar_type r_args[depth][kILP],
    const double& lr,
    const double& beta1,
    const double& beta2,
    const double& weight_decay,
    const double& eps,
    const bool& maximize,
    const float* grad_scale_ptr,
    const float* found_inf_ptr,
    const opmath_t& bias_correction1,
    const opmath_t& bias_correction2_sqrt) {
  static_assert(depth == 4 || depth == 5);
#pragma unroll
  for (int ii = 0; ii < kILP; ii++) {
    // Load values.
    opmath_t param = static_cast<opmath_t>(r_args[kParamIdx][ii]);
    opmath_t grad = static_cast<opmath_t>(r_args[kGradIdx][ii]);
    if (grad_scale_ptr) {
      grad /= (static_cast<double>(*grad_scale_ptr));
    }
    const opmath_t grad_to_store = grad;
    if (maximize) {
      grad = -grad;
    }
    opmath_t exp_avg = static_cast<opmath_t>(r_args[kExpAvgIdx][ii]);
    opmath_t exp_avg_sq = static_cast<opmath_t>(r_args[kExpAvgSqIdx][ii]);
    opmath_t max_exp_avg_sq;
    if (amsgrad) {
      max_exp_avg_sq = static_cast<opmath_t>(r_args[kMaxExpAvgSqIdx][ii]);
    }
    // Update param, grad, 1st and 2nd order momentum.
    if (weight_decay != 0) {
      if constexpr (adam_mode == ADAM_MODE::ORIGINAL) {
        grad += param * weight_decay;
      } else if constexpr (adam_mode == ADAM_MODE::ADAMW) {
        param -= lr * weight_decay * param;
      }
    }

    exp_avg = beta1 * exp_avg + (1 - beta1) * grad;
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad;
    const opmath_t step_size = lr / bias_correction1;
    opmath_t denom;
    if (amsgrad) {
      max_exp_avg_sq = std::max(max_exp_avg_sq, exp_avg_sq);
      denom = (std::sqrt(max_exp_avg_sq) / bias_correction2_sqrt) + eps;
    } else {
      denom = (std::sqrt(exp_avg_sq) / bias_correction2_sqrt) + eps;
    }
    param -= step_size * exp_avg / denom;

    // Store results.
    r_args[kParamIdx][ii] = param;
    if (grad_scale_ptr) {
      r_args[kGradIdx][ii] = grad_to_store;
    }
    r_args[kExpAvgIdx][ii] = exp_avg;
    r_args[kExpAvgSqIdx][ii] = exp_avg_sq;
    if (amsgrad) {
      r_args[kMaxExpAvgSqIdx][ii] = max_exp_avg_sq;
    }
  }
}

template <typename scalar_type, int depth, ADAM_MODE adam_mode, bool amsgrad>
struct FusedAdamMathFunctor {
  static_assert(
      depth == 4 || depth == 5,
      "depth of 4 for Adam, depth of 5 for Adam with AMSGrad.");
  using opmath_t = at::opmath_type<scalar_type>;
  template <typename TLA, typename TLW>
  inline void operator()(
      int chunk_size,
      TLA tlAddress,
      TLW tlWGMeta,
      sycl::nd_item<1> item,
      const float* lr_ptr,
      const double& lr,
      const double& beta1,
      const double& beta2,
      const double& weight_decay,
      const double& eps,
      const bool& maximize,
      const float* grad_scale_ptr,
      const float* found_inf_ptr) const {
    auto group_id = item.get_group(0);
    auto item_id = item.get_local_id(0);
    auto local_range = item.get_local_range(0);

    const auto tensor_loc = tlWGMeta[group_id].wg_to_tensor;
    const auto chunk_idx = tlWGMeta[group_id].wg_to_chunk;
    const double lr_double = lr_ptr ? *lr_ptr : lr;

    if (found_inf_ptr && *found_inf_ptr == 1) {
      return;
    }
    const auto [bias_correction1, bias_correction2_sqrt] =
        [&]() -> std::pair<double, double> {
      auto* step_count = reinterpret_cast<const float*>(
          tlAddress[tensor_loc].state_steps_addresses);
      const auto bias_correction1 = 1 - std::pow(beta1, *step_count);
      const auto bias_correction2 = 1 - std::pow(beta2, *step_count);
      const auto bias_correction2_sqrt = std::sqrt(bias_correction2);
      return {bias_correction1, bias_correction2_sqrt};
    }();

    scalar_type* args[depth];
    scalar_type r_args[depth][kILP];
    const auto n =
        tlAddress[tensor_loc].numel_to_tensor - chunk_idx * chunk_size;

    const bool all_aligned{
        init_args<depth>(args, tlAddress, chunk_idx, chunk_size, tensor_loc)};
    if ((n % kILP == 0) && (chunk_size % kILP == 0) && all_aligned) {
      for (int64_t i_start = item_id;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += local_range) {
#pragma unroll
        for (int i = 0; i < depth; i++) {
          load_store(r_args[i], args[i], 0, i_start);
        }
        adam_math<scalar_type, opmath_t, depth, adam_mode, amsgrad>(
            r_args,
            lr_double,
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr,
            bias_correction1,
            bias_correction2_sqrt);
#pragma unroll
        for (int i = 0; i < depth; i++) {
          if (i != kGradIdx || grad_scale_ptr) {
            load_store(args[i], r_args[i], i_start, 0);
          }
        }
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += local_range * kILP) {
        load_args<depth>(
            r_args, args, i_start, chunk_size, n, item_id, local_range);
        adam_math<scalar_type, opmath_t, depth, adam_mode, amsgrad>(
            r_args,
            lr_double,
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr,
            bias_correction1,
            bias_correction2_sqrt);
#pragma unroll
        for (int i = 0; i < depth; i++) {
          if (i != kGradIdx || grad_scale_ptr) {
            store_args(
                args[i],
                r_args[i],
                i_start,
                chunk_size,
                n,
                item_id,
                local_range);
          }
        }
      }
    }
  }
};

} // namespace at::native::xpu
