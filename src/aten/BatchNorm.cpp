#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
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

::std::tuple<Tensor, Tensor, Tensor> XPUNativeFunctions::native_batch_norm(
    const Tensor& input,
    const ::std::optional<Tensor>& weight,
    const ::std::optional<Tensor>& bias,
    const ::std::optional<Tensor>& running_mean,
    const ::std::optional<Tensor>& running_var,
    bool training,
    double momentum,
    double eps) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::native_batch_norm", "input");
  c10::impl::check_and_update_common_device(
      common_device, weight, "xpu::native_batch_norm", "weight");
  c10::impl::check_and_update_common_device(
      common_device, bias, "xpu::native_batch_norm", "bias");
  c10::impl::check_and_update_common_device(
      common_device, running_mean, "xpu::native_batch_norm", "running_mean");
  c10::impl::check_and_update_common_device(
      common_device, running_var, "xpu::native_batch_norm", "running_var");

  auto output = at::empty_like(input);
  int64_t n_input = input.size(1);
  auto options =
      input.options().dtype(at::toAccumulateType(input.scalar_type(), true));
  auto save_mean = at::empty({n_input}, options);
  auto save_invstd = at::empty({n_input}, options);

  native::xpu::batch_norm_out_kernel(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      training,
      momentum,
      eps,
      output,
      save_mean,
      save_invstd);

  return std::make_tuple(output, save_mean, save_invstd);
}

::std::tuple<Tensor&, Tensor&, Tensor&> XPUNativeFunctions::
    native_batch_norm_out(
        const Tensor& input,
        const ::std::optional<Tensor>& weight,
        const ::std::optional<Tensor>& bias,
        const ::std::optional<Tensor>& running_mean,
        const ::std::optional<Tensor>& running_var,
        bool training,
        double momentum,
        double eps,
        Tensor& out,
        Tensor& save_mean,
        Tensor& save_invstd) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "xpu::native_batch_norm_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, save_mean, "xpu::native_batch_norm_out", "save_mean");
  c10::impl::check_and_update_common_device(
      common_device, save_invstd, "xpu::native_batch_norm_out", "save_invstd");
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::native_batch_norm_out", "input");
  c10::impl::check_and_update_common_device(
      common_device, weight, "xpu::native_batch_norm_out", "weight");
  c10::impl::check_and_update_common_device(
      common_device, bias, "xpu::native_batch_norm_out", "bias");
  c10::impl::check_and_update_common_device(
      common_device,
      running_mean,
      "xpu::native_batch_norm_out",
      "running_mean");
  c10::impl::check_and_update_common_device(
      common_device,
      running_var,
      "xpu::out_native_batch_norm_out",
      "running_var");
  return native::xpu::batch_norm_out_kernel(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      training,
      momentum,
      eps,
      out,
      save_mean,
      save_invstd);
}

::std::tuple<Tensor, Tensor, Tensor> XPUNativeFunctions::
    native_batch_norm_backward(
        const Tensor& grad_out,
        const Tensor& input,
        const ::std::optional<Tensor>& weight,
        const ::std::optional<Tensor>& running_mean,
        const ::std::optional<Tensor>& running_var,
        const ::std::optional<Tensor>& save_mean,
        const ::std::optional<Tensor>& save_invstd,
        bool train,
        double eps,
        ::std::array<bool, 3> output_mask) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, grad_out, "xpu::native_batch_norm_backward", "grad_out");
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::native_batch_norm_backward", "input");
  c10::impl::check_and_update_common_device(
      common_device, weight, "xpu::native_batch_norm_backward", "weight");
  c10::impl::check_and_update_common_device(
      common_device,
      running_mean,
      "xpu::native_batch_norm_backward",
      "running_mean");
  c10::impl::check_and_update_common_device(
      common_device,
      running_var,
      "xpu::native_batch_norm_backward",
      "running_var");
  c10::impl::check_and_update_common_device(
      common_device, save_mean, "xpu::native_batch_norm_backward", "save_mean");
  c10::impl::check_and_update_common_device(
      common_device,
      save_invstd,
      "xpu::native_batch_norm_backward",
      "save_invstd");
  return native::xpu::batch_norm_backward_kernel(
      grad_out,
      input,
      weight,
      running_mean,
      running_var,
      save_mean,
      save_invstd,
      train,
      eps,
      output_mask);
}

} // namespace at
