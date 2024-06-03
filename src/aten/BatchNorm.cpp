#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <aten/sycl/BatchNormKernels.h>

namespace at {

::std::tuple<Tensor, Tensor> XPUNativeFunctions::batch_norm_stats(
    const Tensor& input,
    double eps) {
  return native::xpu::batch_norm_stats_kernel(input, eps);
}

Tensor XPUNativeFunctions::batch_norm_elemt(
    const Tensor& input,
    const ::std::optional<Tensor>& weight,
    const ::std::optional<Tensor>& bias,
    const Tensor& mean,
    const Tensor& invstd,
    double eps) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::batch_norm_elemt", "input");
  c10::impl::check_and_update_common_device(
      common_device, weight, "xpu::batch_norm_elemt", "weight");
  c10::impl::check_and_update_common_device(
      common_device, bias, "xpu::batch_norm_elemt", "bias");
  c10::impl::check_and_update_common_device(
      common_device, mean, "xpu::batch_norm_elemt", "mean");
  c10::impl::check_and_update_common_device(
      common_device, invstd, "xpu::batch_norm_elemt", "invstd");
  auto output = at::empty_like(input);
  native::xpu::batch_norm_elemt_kernel(
      output, input, weight, bias, mean, invstd);
  return output;
}

Tensor& XPUNativeFunctions::batch_norm_elemt_out(
    const Tensor& input,
    const ::std::optional<Tensor>& weight,
    const ::std::optional<Tensor>& bias,
    const Tensor& mean,
    const Tensor& invstd,
    double eps,
    Tensor& out) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "xpu::batch_norm_elemt_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::batch_norm_elemt_out", "input");
  c10::impl::check_and_update_common_device(
      common_device, weight, "xpu::batch_norm_elemt_out", "weight");
  c10::impl::check_and_update_common_device(
      common_device, bias, "xpu::batch_norm_elemt_out", "bias");
  c10::impl::check_and_update_common_device(
      common_device, mean, "xpu::batch_norm_elemt_out", "mean");
  c10::impl::check_and_update_common_device(
      common_device, invstd, "xpu::batch_norm_elemt_out", "invstd");
  native::xpu::batch_norm_elemt_kernel(out, input, weight, bias, mean, invstd);
  return out;
}

::std::tuple<Tensor, Tensor> XPUNativeFunctions::batch_norm_gather_stats(
    const Tensor& input,
    const Tensor& mean,
    const Tensor& invstd,
    const ::std::optional<Tensor>& running_mean,
    const ::std::optional<Tensor>& running_var,
    double momentum,
    double eps,
    int64_t count) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::batch_norm_gather_stats", "input");
  c10::impl::check_and_update_common_device(
      common_device, mean, "xpu::batch_norm_gather_stats", "mean");
  c10::impl::check_and_update_common_device(
      common_device, invstd, "xpu::batch_norm_gather_stats", "invstd");
  c10::impl::check_and_update_common_device(
      common_device,
      running_mean,
      "xpu::batch_norm_gather_stats",
      "running_mean");
  c10::impl::check_and_update_common_device(
      common_device,
      running_var,
      "xpu::batch_norm_gather_stats",
      "running_var");
  return native::xpu::batch_norm_gather_stats_kernel(
      input, mean, invstd, running_mean, running_var, momentum, eps, count);
}

::std::tuple<Tensor, Tensor> XPUNativeFunctions::
    batch_norm_gather_stats_with_counts(
        const Tensor& input,
        const Tensor& mean,
        const Tensor& invstd,
        const ::std::optional<Tensor>& running_mean,
        const ::std::optional<Tensor>& running_var,
        double momentum,
        double eps,
        const Tensor& counts) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device,
      input,
      "xpu::batch_norm_gather_stats_with_counts",
      "input");
  c10::impl::check_and_update_common_device(
      common_device, mean, "xpu::batch_norm_gather_stats_with_counts", "mean");
  c10::impl::check_and_update_common_device(
      common_device,
      invstd,
      "xpu::batch_norm_gather_stats_with_counts",
      "invstd");
  c10::impl::check_and_update_common_device(
      common_device,
      running_mean,
      "xpu::batch_norm_gather_stats_with_counts",
      "running_mean");
  c10::impl::check_and_update_common_device(
      common_device,
      running_var,
      "xpu::batch_norm_gather_stats_with_counts",
      "running_var");
  c10::impl::check_and_update_common_device(
      common_device,
      counts,
      "xpu::batch_norm_gather_stats_with_counts",
      "counts");
  return native::xpu::batch_norm_gather_stats_with_counts_kernel(
      input, mean, invstd, running_mean, running_var, momentum, eps, counts);
}

::std::tuple<Tensor, Tensor, Tensor, Tensor> XPUNativeFunctions::
    batch_norm_backward_reduce(
        const Tensor& grad_out,
        const Tensor& input,
        const Tensor& mean,
        const Tensor& invstd,
        const ::std::optional<Tensor>& weight,
        bool input_g,
        bool weight_g,
        bool bias_g) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, grad_out, "xpu::batch_norm_backward_reduce", "grad_out");
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::batch_norm_backward_reduce", "input");
  c10::impl::check_and_update_common_device(
      common_device, mean, "xpu::batch_norm_backward_reduce", "mean");
  c10::impl::check_and_update_common_device(
      common_device, invstd, "xpu::batch_norm_backward_reduce", "invstd");
  c10::impl::check_and_update_common_device(
      common_device, weight, "xpu::batch_norm_backward_reduce", "weight");
  return native::xpu::batch_norm_backward_reduce_kernel(
      grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
}

Tensor XPUNativeFunctions::batch_norm_backward_elemt(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& invstd,
    const ::std::optional<Tensor>& weight,
    const Tensor& sum_dy,
    const Tensor& sum_dy_xmu,
    const Tensor& count) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, grad_out, "xpu::batch_norm_backward_elemt", "grad_out");
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::batch_norm_backward_elemt", "input");
  c10::impl::check_and_update_common_device(
      common_device, mean, "xpu::batch_norm_backward_elemt", "mean");
  c10::impl::check_and_update_common_device(
      common_device, invstd, "xpu::batch_norm_backward_elemt", "invstd");
  c10::impl::check_and_update_common_device(
      common_device, weight, "xpu::batch_norm_backward_elemt", "weight");
  c10::impl::check_and_update_common_device(
      common_device, sum_dy, "xpu::batch_norm_backward_elemt", "sum_dy");
  c10::impl::check_and_update_common_device(
      common_device,
      sum_dy_xmu,
      "xpu::batch_norm_backward_elemt",
      "sum_dy_xmu");
  c10::impl::check_and_update_common_device(
      common_device, count, "xpu::batch_norm_backward_elemt", "count");
  return native::xpu::batch_norm_backward_elemt_kernel(
      grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
}

::std::tuple<Tensor, Tensor> XPUNativeFunctions::batch_norm_update_stats(
    const Tensor& input,
    const ::std::optional<Tensor>& running_mean,
    const ::std::optional<Tensor>& running_var,
    double momentum) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::batch_norm_update_stats", "input");
  c10::impl::check_and_update_common_device(
      common_device,
      running_mean,
      "xpu::batch_norm_update_stats",
      "running_mean");
  c10::impl::check_and_update_common_device(
      common_device,
      running_var,
      "xpu::batch_norm_update_stats",
      "running_var");
  return native::xpu::batch_norm_update_stats_kernel(
      input, running_mean, running_var, momentum);
}

} // namespace at
