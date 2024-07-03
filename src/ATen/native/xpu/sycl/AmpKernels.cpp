#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>
#include <comm/SYCLHelpers.h>

namespace at::native::xpu {

template <typename scalar_t>
struct AmpNonFiniteCheckUnscaleFunctor {
  using opmath_t = at::opmath_type<scalar_t>;

  scalar_t operator()(scalar_t val_in) const {
    auto val = static_cast<opmath_t>(val_in);
    if (std::isinf(val) || std::isnan(val)) {
      *found_inf_ptr_ = 1.f;
    }
    const auto inv_scale_val = *inv_scale_ptr_;
    return static_cast<scalar_t>(
        inv_scale_val == 1.f ? val : val * inv_scale_val);
  }

  AmpNonFiniteCheckUnscaleFunctor(float* found_inf_ptr, float* inv_scale_ptr)
      : found_inf_ptr_(found_inf_ptr), inv_scale_ptr_(inv_scale_ptr) {}

 private:
  float* found_inf_ptr_;
  float* inv_scale_ptr_;
};

void amp_non_finite_check_and_unscale_kernel(
    Tensor& scaled_grad,
    Tensor& found_inf,
    const Tensor& inv_scale) {
  auto iter = TensorIterator::unary_op(scaled_grad, scaled_grad);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "amp_non_finite_check_and_unscale_xpu",
      [&iter, &found_inf, &inv_scale] {
        auto* found_inf_ptr = found_inf.data_ptr<float>();
        auto* inv_scale_ptr = inv_scale.data_ptr<float>();

        AmpNonFiniteCheckUnscaleFunctor<scalar_t> f(
            found_inf_ptr, inv_scale_ptr);
        gpu_kernel(iter, f);
      });
}

template <typename opmath_t>
struct AmpForeachNonFiniteCheckUnscaleFunctor {
  opmath_t operator()(opmath_t val) const {
    if (std::isinf(val) || std::isnan(val)) {
      *found_inf_ptr_ = 1.f;
    }
    const auto inv_scale_val = *inv_scale_ptr_;
    return static_cast<opmath_t>(
        inv_scale_val == 1.f ? val : val * inv_scale_val);
  }

  AmpForeachNonFiniteCheckUnscaleFunctor(
      float* found_inf_ptr,
      float* inv_scale_ptr)
      : found_inf_ptr_(found_inf_ptr), inv_scale_ptr_(inv_scale_ptr) {}

 private:
  float* found_inf_ptr_;
  float* inv_scale_ptr_;
};

void amp_foreach_non_finite_check_and_unscale_kernel(
    std::vector<std::vector<at::Tensor>> scaled_grads,
    Tensor& found_inf,
    const Tensor& inv_scale) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scaled_grads[0][0].scalar_type(),
      "amp_foreach_non_finite_check_and_unscale_xpu",
      [&scaled_grads, &found_inf, &inv_scale] {
        auto* found_inf_ptr = found_inf.data_ptr<float>();
        auto* inv_scale_ptr = inv_scale.data_ptr<float>();

        using opmath_t = at::opmath_type<scalar_t>;

        AmpForeachNonFiniteCheckUnscaleFunctor<opmath_t> f(
            found_inf_ptr, inv_scale_ptr);
        multi_tensor_apply<1>(
            scaled_grads,
            UnaryOpFunctor<
                scalar_t,
                /* depth */ 1,
                /* r_args_depth */ 1,
                /* res_arg_index */ 0>(),
            f);
      });
}

struct AmpUpdateScaleKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    // There is only single item/task scheduled.
    if (item.get_global_linear_id() != 0)
      return;

    if (*found_inf_) {
      *current_scale_ *= backoff_factor_;
      *growth_tracker_ = 0;
    } else {
      // Entering this branch means we just carried out a successful step,
      // so growth_tracker is incremented before comparing to growth_interval.
      auto successful = (*growth_tracker_) + 1;
      if (successful == growth_interval_) {
        auto new_scale = static_cast<float>((*current_scale_) * growth_factor_);
        if (!std::isinf(new_scale)) {
          *current_scale_ = new_scale;
        }
        *growth_tracker_ = 0;
      } else {
        *growth_tracker_ = successful;
      }
    }
  }
  AmpUpdateScaleKernelFunctor(
      float* current_scale,
      int* growth_tracker,
      const float* found_inf,
      double growth_factor,
      double backoff_factor,
      int growth_interval)
      : current_scale_(current_scale),
        growth_tracker_(growth_tracker),
        found_inf_(found_inf),
        growth_factor_(growth_factor),
        backoff_factor_(backoff_factor),
        growth_interval_(growth_interval) {}

 private:
  float* current_scale_;
  int* growth_tracker_;
  const float* found_inf_;
  double growth_factor_;
  double backoff_factor_;
  int growth_interval_;
};

Tensor& amp_update_scale_kernel(
    Tensor& current_scale,
    Tensor& growth_tracker,
    const Tensor& found_inf,
    double growth_factor,
    double backoff_factor,
    int64_t growth_interval) {
  AmpUpdateScaleKernelFunctor kfn(
      current_scale.mutable_data_ptr<float>(),
      growth_tracker.mutable_data_ptr<int>(),
      found_inf.const_data_ptr<float>(),
      growth_factor,
      backoff_factor,
      growth_interval);
  sycl_kernel_submit(
      sycl::range<1>(1), sycl::range<1>(1), getCurrentSYCLQueue(), kfn);

  return current_scale;
}

} // namespace at::native::xpu
