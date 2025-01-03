#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>

#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/FusedSgdKernels.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>

#include <comm/SYCLHelpers.h>

namespace at::native::xpu {

namespace {

template <typename scalar_t, int depth>
void sgd_math(
    scalar_t r_args[depth][kILP],
    const double weight_decay,
    const double momentum,
    const float* lr_ptr,
    const double lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const float* grad_scale_ptr) {
  using opmath_t = at::opmath_type<scalar_t>;
  const double double_lr = lr_ptr != nullptr ? *lr_ptr : lr;
#pragma unroll
  for (int ii = 0; ii < kILP; ii++) {
    auto p = static_cast<opmath_t>(r_args[0][ii]);
    auto g = static_cast<opmath_t>(r_args[1][ii]);
    if (grad_scale_ptr) {
      g /= static_cast<double>(*grad_scale_ptr);
      r_args[1][ii] = g;
    }
    if (maximize) {
      g *= -1.0;
    }
    if (weight_decay != 0) {
      g += weight_decay * p;
    }
    if (depth > 2) {
      const auto momentum_buffer = is_first_step
          ? g
          : (momentum * static_cast<opmath_t>(r_args[2][ii]) +
             (1 - dampening) * g);
      r_args[2][ii] = momentum_buffer;

      if (nesterov) {
        g = g + momentum * momentum_buffer;
      } else {
        g = momentum_buffer;
      }
    }
    p -= double_lr * g;
    r_args[0][ii] = p;
  }
}

template <typename scalar_t, int depth>
struct FusedSgdMathFunctor {
  static_assert(
      depth == 2 || depth == 3,
      "depth of 2 for SGD w/ momentum == 0, 3 for SGD w/ momentum != 0");

  template <typename TLA, typename TLW>
  void operator()(
      const int chunk_size,
      TLA tlAddress,
      TLW tlWGMeta,
      sycl::nd_item<1> item,
      const double weight_decay,
      const double momentum,
      const float* lr_ptr,
      const double lr,
      const double dampening,
      const bool nesterov,
      const bool maximize,
      const bool is_first_step,
      const float* grad_scale_ptr,
      const float* found_inf_ptr) const {
    if (found_inf_ptr && *found_inf_ptr == 1) {
      return;
    }

    auto workgroup_id = item.get_group(0);
    auto item_id = item.get_local_id(0);
    auto local_range = item.get_local_range(0);

    const auto tensor_loc = tlWGMeta[workgroup_id].wg_to_tensor;
    const auto chunk_idx = tlWGMeta[workgroup_id].wg_to_chunk;

    scalar_t* args[depth];
    scalar_t r_args[depth][kILP];
    const auto all_aligned{
        init_args<depth>(args, tlAddress, chunk_idx, chunk_size, tensor_loc)};
    const auto n =
        tlAddress[tensor_loc].numel_to_tensor - chunk_idx * chunk_size;

    const auto use_faster_load_store =
        (n % kILP == 0) && (chunk_size % kILP == 0) && all_aligned;
    if (use_faster_load_store) {
      for (auto i_start = item_id;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += local_range) {
#pragma unroll
        for (auto i = 0; i < depth; i++) {
          load_store(r_args[i], args[i], 0, i_start);
        }
        sgd_math<scalar_t, depth>(
            r_args,
            weight_decay,
            momentum,
            lr_ptr,
            lr,
            dampening,
            nesterov,
            maximize,
            is_first_step,
            grad_scale_ptr);
        load_store(args[0], r_args[0], i_start, 0);
        if (grad_scale_ptr) {
          load_store(args[1], r_args[1], i_start, 0);
        }
        if (depth > 2) {
          load_store(args[2], r_args[2], i_start, 0);
        }
      }
    } else {
      for (auto i_start = 0; i_start < n && i_start < chunk_size;
           i_start += local_range * kILP) {
        load_args<depth>(
            r_args, args, i_start, chunk_size, n, item_id, local_range);
        sgd_math<scalar_t, depth>(
            r_args,
            weight_decay,
            momentum,
            lr_ptr,
            lr,
            dampening,
            nesterov,
            maximize,
            is_first_step,
            grad_scale_ptr);
        store_args(
            args[0], r_args[0], i_start, chunk_size, n, item_id, local_range);
        if (grad_scale_ptr) {
          store_args(
              args[1], r_args[1], i_start, chunk_size, n, item_id, local_range);
        }
        if (depth > 2) {
          store_args(
              args[2], r_args[2], i_start, chunk_size, n, item_id, local_range);
        }
      }
    }
  }
};

} // namespace

void fused_sgd_kernel(
    at::TensorList params,
    at::TensorList grads,
    const double weight_decay,
    const double momentum,
    const float* lr_ptr,
    const double lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const float* grad_scale_ptr,
    const float* found_inf_ptr) {
  std::vector<std::vector<at::Tensor>> tensor_lists{params.vec(), grads.vec()};
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, params[0].scalar_type(), "fused_sgd_kernel_xpu", [&]() {
        multi_tensor_apply<2>(
            tensor_lists,
            FusedSgdMathFunctor<scalar_t, 2>(),
            weight_decay,
            momentum,
            lr_ptr,
            lr,
            dampening,
            nesterov,
            maximize,
            is_first_step,
            grad_scale_ptr,
            found_inf_ptr);
      });
};

void fused_sgd_with_momentum_kernel(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const double weight_decay,
    const double momentum,
    const float* lr_ptr,
    const double lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const float* grad_scale_ptr,
    const float* found_inf_ptr) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      params.vec(), grads.vec(), momentum_buffer_list.vec()};
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      params[0].scalar_type(),
      "fused_sgd_with_momentum_kernel_xpu",
      [&]() {
        multi_tensor_apply<3>(
            tensor_lists,
            FusedSgdMathFunctor<scalar_t, 3>(),
            weight_decay,
            momentum,
            lr_ptr,
            lr,
            dampening,
            nesterov,
            maximize,
            is_first_step,
            grad_scale_ptr,
            found_inf_ptr);
      });
};

} // namespace at::native::xpu
