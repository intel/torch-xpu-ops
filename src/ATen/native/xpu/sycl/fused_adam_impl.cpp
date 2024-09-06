#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>

#include <ATen/native/xpu/sycl/MultiTensorApply.h>
#include <ATen/native/xpu/sycl/fused_adam_utils.h>

namespace at::native::xpu {

void _fused_adam_xpu_impl_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      params.vec(), grads.vec(), exp_avgs.vec(), exp_avg_sqs.vec()};

  if (grad_scale.has_value())
    TORCH_WARN(" grad_scale not supported with ipex fused_adam.")
  if (found_inf.has_value() && found_inf->data_ptr<float>()[0] == true)
    TORCH_WARN(" found_inf not supported with ipex fused_adam.")

  const float beta1_value = static_cast<float>(beta1);
  const float beta2_value = static_cast<float>(beta2);
  const float weight_decay_value = static_cast<float>(weight_decay);
  const float eps_value = static_cast<float>(eps);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      params[0].scalar_type(),
      "fused_adam_kernel_xpu",
      [&]() {
        multi_tensor_apply_for_fused_optimizer<4>(
            tensor_lists,
            state_steps,
            FusedAdamMathFunctor<scalar_t, 4, ADAM_MODE::ORIGINAL, false>(),
            (const float*)nullptr,
            lr,
            beta1_value,
            beta2_value,
            weight_decay_value,
            eps_value,
            maximize);
      });
}

void _fused_adam_xpu_impl_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList state_steps,
    const Tensor& lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      params.vec(), grads.vec(), exp_avgs.vec(), exp_avg_sqs.vec()};

  if (grad_scale.has_value())
    TORCH_WARN(" grad_scale not supported with ipex fused_adam.")
  if (found_inf.has_value() && found_inf->data_ptr<float>()[0] == true)
    TORCH_WARN(" found_inf not supported with ipex fused_adam.")

  const float beta1_value = static_cast<float>(beta1);
  const float beta2_value = static_cast<float>(beta2);
  const float weight_decay_value = static_cast<float>(weight_decay);
  const float eps_value = static_cast<float>(eps);
  float* lr_ptr = lr.data_ptr<float>();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      params[0].scalar_type(),
      "fused_adam_kernel_xpu",
      [&]() {
        multi_tensor_apply_for_fused_optimizer<4>(
            tensor_lists,
            state_steps,
            FusedAdamMathFunctor<scalar_t, 4, ADAM_MODE::ORIGINAL, false>(),
            lr_ptr,
            0.0f,
            beta1_value,
            beta2_value,
            weight_decay_value,
            eps_value,
            maximize);
      });
}

} // namespace at::native::xpu
