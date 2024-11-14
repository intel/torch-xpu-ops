#include <ATen/core/Tensor.h>
#include <ATen/native/ForeachUtils.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <comm/SYCLHelpers.h>

#include <ATen/native/xpu/sycl/FusedSgdKernels.h>

namespace at::native {

void _fused_sgd_with_momentum_kernel_xpu_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const double weight_decay,
    const double momentum,
    const double lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  TORCH_CHECK_GT(momentum, 0);
  TORCH_CHECK(at::native::check_fast_path_restrictions(
      {params, grads, momentum_buffer_list}));
  if (grad_scale != std::nullopt) {
    TORCH_CHECK(
        grad_scale->device() == params[0].device(),
        "grad_scale must be on the same XPU device as the params");
  }
  if (found_inf != std::nullopt) {
    TORCH_CHECK(
        found_inf->device() == params[0].device(),
        "found_inf must be on the same XPU device as the params");
  }

  float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  float* lr_ptr = nullptr;

  xpu::fused_sgd_with_momentum_kernel(
      params,
      grads,
      momentum_buffer_list,
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
}

void _fused_sgd_with_momentum_kernel_xpu_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const double weight_decay,
    const double momentum,
    const at::Tensor& lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (lr.is_cpu()) {
    _fused_sgd_with_momentum_kernel_xpu_(
        params,
        grads,
        momentum_buffer_list,
        weight_decay,
        momentum,
        lr.item<double>(),
        dampening,
        nesterov,
        maximize,
        is_first_step,
        grad_scale,
        found_inf);
    return;
  }
  TORCH_CHECK_GT(momentum, 0);
  TORCH_CHECK(at::native::check_fast_path_restrictions(
      {params, grads, momentum_buffer_list}));
  if (grad_scale != std::nullopt) {
    TORCH_CHECK(
        grad_scale->device() == params[0].device(),
        "grad_scale must be on the same XPU device as the params");
  }
  if (found_inf != std::nullopt) {
    TORCH_CHECK(
        found_inf->device() == params[0].device(),
        "found_inf must be on the same XPU device as the params");
  }
  TORCH_CHECK(
      lr.device() == params[0].device(),
      "found_inf must be on the same XPU device as the params");

  float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;

  xpu::fused_sgd_with_momentum_kernel(
      params,
      grads,
      momentum_buffer_list,
      weight_decay,
      momentum,
      lr.data_ptr<float>(),
      1.0,
      dampening,
      nesterov,
      maximize,
      is_first_step,
      grad_scale_ptr,
      found_inf_ptr);
}

void _fused_sgd_kernel_xpu_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const double weight_decay,
    const double momentum,
    const double lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (!momentum_buffer_list.empty()) {
    _fused_sgd_with_momentum_kernel_xpu_(
        params,
        grads,
        momentum_buffer_list,
        weight_decay,
        momentum,
        lr,
        dampening,
        nesterov,
        maximize,
        is_first_step,
        grad_scale,
        found_inf);
    return;
  }
  TORCH_CHECK_EQ(momentum, 0);
  TORCH_CHECK(at::native::check_fast_path_restrictions({params, grads}));
  if (is_first_step) {
    TORCH_WARN_ONCE(
        "`is_first_step` argument has no effect when `momentum_buffer_list` is empty");
  }
  if (grad_scale.has_value()) {
    TORCH_CHECK(
        grad_scale->device() == params[0].device(),
        "grad_scale must be on the same XPU device as the params");
  }
  if (found_inf.has_value()) {
    TORCH_CHECK(
        found_inf->device() == params[0].device(),
        "found_inf must be on the same XPU device as the params");
  }

  float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  float* lr_ptr = nullptr;

  xpu::fused_sgd_kernel(
      params,
      grads,
      weight_decay,
      momentum,
      lr_ptr,
      lr,
      dampening,
      nesterov,
      maximize,
      /* is_first_step */ false,
      grad_scale_ptr,
      found_inf_ptr);
}

void _fused_sgd_kernel_xpu_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const double weight_decay,
    const double momentum,
    const at::Tensor& lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (!momentum_buffer_list.empty()) {
    _fused_sgd_with_momentum_kernel_xpu_(
        params,
        grads,
        momentum_buffer_list,
        weight_decay,
        momentum,
        lr,
        dampening,
        nesterov,
        maximize,
        is_first_step,
        grad_scale,
        found_inf);
    return;
  }
  if (lr.is_cpu()) {
    _fused_sgd_kernel_xpu_(
        params,
        grads,
        momentum_buffer_list,
        weight_decay,
        momentum,
        lr.item<double>(),
        dampening,
        nesterov,
        maximize,
        is_first_step,
        grad_scale,
        found_inf);
    return;
  }
  TORCH_CHECK_EQ(momentum, 0);
  TORCH_CHECK(at::native::check_fast_path_restrictions({params, grads}));
  if (is_first_step) {
    TORCH_WARN_ONCE(
        "`is_first_step` argument has no effect when `momentum_buffer_list` is empty");
  }
  if (grad_scale.has_value()) {
    TORCH_CHECK(
        grad_scale->device() == params[0].device(),
        "grad_scale must be on the same XPU device as the params");
  }
  if (found_inf.has_value()) {
    TORCH_CHECK(
        found_inf->device() == params[0].device(),
        "found_inf must be on the same XPU device as the params");
  }
  TORCH_CHECK(
      lr.device() == params[0].device(),
      "lr must be on the same XPU device as the params");

  float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;

  xpu::fused_sgd_kernel(
      params,
      grads,
      weight_decay,
      momentum,
      lr.data_ptr<float>(),
      1.0,
      dampening,
      nesterov,
      maximize,
      /* is_first_step */ false,
      grad_scale_ptr,
      found_inf_ptr);
}

} // namespace at::native
