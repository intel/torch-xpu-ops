#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/xpu/sycl/BatchNormKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>

namespace at {

std::tuple<Tensor, Tensor> XPUNativeFunctions::batch_norm_stats(
    const Tensor& input,
    double eps) {
  return native::xpu::batch_norm_stats_kernel(input, eps);
}

Tensor XPUNativeFunctions::batch_norm_elemt(
    const Tensor& input,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const Tensor& mean,
    const Tensor& invstd,
    double eps) {
  auto output = at::empty_like(input);
  native::xpu::batch_norm_elemt_kernel(
      output, input, weight, bias, mean, invstd);
  return output;
}

Tensor& XPUNativeFunctions::batch_norm_elemt_out(
    const Tensor& input,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const Tensor& mean,
    const Tensor& invstd,
    double eps,
    Tensor& out) {
  native::xpu::batch_norm_elemt_kernel(out, input, weight, bias, mean, invstd);
  return out;
}

std::tuple<Tensor, Tensor, Tensor, Tensor> XPUNativeFunctions::
    batch_norm_backward_reduce(
        const Tensor& grad_out,
        const Tensor& input,
        const Tensor& mean,
        const Tensor& invstd,
        const std::optional<Tensor>& weight,
        bool input_g,
        bool weight_g,
        bool bias_g) {
  return native::xpu::batch_norm_backward_reduce_kernel(
      grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
}

Tensor XPUNativeFunctions::batch_norm_backward_elemt(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& invstd,
    const std::optional<Tensor>& weight,
    const Tensor& sum_dy,
    const Tensor& sum_dy_xmu,
    const Tensor& count) {
  return native::xpu::batch_norm_backward_elemt_kernel(
      grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::batch_norm_update_stats(
    const Tensor& input,
    const std::optional<Tensor>& running_mean,
    const std::optional<Tensor>& running_var,
    double momentum) {
  return native::xpu::batch_norm_update_stats_kernel(
      input, running_mean, running_var, momentum);
}

std::tuple<Tensor, Tensor, Tensor> XPUNativeFunctions::native_batch_norm(
    const Tensor& input,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& running_mean,
    const std::optional<Tensor>& running_var,
    bool training,
    double momentum,
    double eps) {
  auto output = at::empty_like(input);
  int64_t n_input = input.size(1);
  auto options =
      input.options().dtype(at::toAccumulateType(input.scalar_type(), true));
  auto save_mean = at::empty({n_input}, options);
  auto save_invstd = at::empty({n_input}, options);

  native::xpu::batch_norm_kernel(
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

std::tuple<Tensor&, Tensor&, Tensor&> XPUNativeFunctions::native_batch_norm_out(
    const Tensor& input,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& running_mean,
    const std::optional<Tensor>& running_var,
    bool training,
    double momentum,
    double eps,
    Tensor& out,
    Tensor& save_mean,
    Tensor& save_invstd) {
  return native::xpu::batch_norm_kernel(
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

std::tuple<Tensor, Tensor, Tensor> XPUNativeFunctions::
    native_batch_norm_backward(
        const Tensor& grad_out,
        const Tensor& input,
        const std::optional<Tensor>& weight,
        const std::optional<Tensor>& running_mean,
        const std::optional<Tensor>& running_var,
        const std::optional<Tensor>& save_mean,
        const std::optional<Tensor>& save_invstd,
        bool train,
        double eps,
        std::array<bool, 3> output_mask) {
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

std::tuple<Tensor, Tensor, Tensor> XPUNativeFunctions::_native_batch_norm_legit(
    const Tensor& input,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    Tensor& running_mean,
    Tensor& running_var,
    bool training,
    double momentum,
    double eps) {
  return XPUNativeFunctions::native_batch_norm(
      input, weight, bias, running_mean, running_var, training, momentum, eps);
}

std::tuple<Tensor&, Tensor&, Tensor&> XPUNativeFunctions::
    _native_batch_norm_legit_out(
        const Tensor& input,
        const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias,
        Tensor& running_mean,
        Tensor& running_var,
        bool training,
        double momentum,
        double eps,
        Tensor& out,
        Tensor& save_mean,
        Tensor& save_invstd) {
  return XPUNativeFunctions::native_batch_norm_out(
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

std::tuple<Tensor, Tensor, Tensor> XPUNativeFunctions::_native_batch_norm_legit(
    const Tensor& input,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    bool training,
    double momentum,
    double eps) {
  return XPUNativeFunctions::native_batch_norm(
      input, weight, bias, Tensor(), Tensor(), training, momentum, eps);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> XPUNativeFunctions::
    _native_batch_norm_legit_out(
        const at::Tensor& input,
        const std::optional<at::Tensor>& weight,
        const std::optional<at::Tensor>& bias,
        bool training,
        double momentum,
        double eps,
        at::Tensor& out,
        at::Tensor& save_mean,
        at::Tensor& save_invstd) {
  return XPUNativeFunctions::native_batch_norm_out(
      input,
      weight,
      bias,
      Tensor(),
      Tensor(),
      training,
      momentum,
      eps,
      out,
      save_mean,
      save_invstd);
}

inline std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_with_update(
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    Tensor& running_mean,
    Tensor& running_var,
    double momentum,
    double eps) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& bias = c10::value_or_else(bias_opt, [] { return Tensor(); });
  Tensor reserve;

  reserve = at::empty({0}, input.options().dtype(kByte));

  auto output = at::empty_like(input);
  int64_t n_input = input.size(1);
  auto options =
      input.options().dtype(at::toAccumulateType(input.scalar_type(), true));
  auto save_mean = at::empty({n_input}, options);
  auto save_invstd = at::empty({n_input}, options);

  native::xpu::batch_norm_kernel(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      /*training*/ true,
      momentum,
      eps,
      output,
      save_mean,
      save_invstd);

  return std::tuple<Tensor, Tensor, Tensor, Tensor>(
      output, save_mean, save_invstd, reserve);
}

inline std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> batch_norm_with_update_out(
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    Tensor& running_mean,
    Tensor& running_var,
    double momentum,
    double eps,
    Tensor& out,
    Tensor& save_mean,
    Tensor& save_var,
    Tensor& reserve) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& bias = c10::value_or_else(bias_opt, [] { return Tensor(); });

  std::tie(out, save_mean, save_var) = native::xpu::batch_norm_kernel(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      /*update*/ true,
      momentum,
      eps,
      out,
      save_mean,
      save_var);

  return std::tuple<Tensor&, Tensor&, Tensor&, Tensor&>(
      out, save_mean, save_var, reserve);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> XPUNativeFunctions::
    _batch_norm_with_update(
        const Tensor& input,
        const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias,
        Tensor& running_mean,
        Tensor& running_var,
        double momentum,
        double eps) {
  return batch_norm_with_update(
      input, weight, bias, running_mean, running_var, momentum, eps);
}

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> XPUNativeFunctions::
    _batch_norm_with_update_out(
        const Tensor& input,
        const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias,
        Tensor& running_mean,
        Tensor& running_var,
        double momentum,
        double eps,
        Tensor& out,
        Tensor& save_mean,
        Tensor& save_invstd,
        Tensor& reserve) {
  return batch_norm_with_update_out(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      momentum,
      eps,
      out,
      save_mean,
      save_invstd,
      reserve);
}

std::tuple<Tensor, Tensor, Tensor> XPUNativeFunctions::batch_norm_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    const std::optional<Tensor>& save_mean_opt,
    const std::optional<Tensor>& save_var_opt,
    bool update,
    double eps,
    std::array<bool, 3> grad_input_mask,
    const Tensor& reserve) {
  const Tensor& running_mean =
      c10::value_or_else(running_mean_opt, [] { return Tensor(); });
  const Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return Tensor(); });
  const Tensor& save_mean =
      c10::value_or_else(save_mean_opt, [] { return Tensor(); });
  const Tensor& save_var =
      c10::value_or_else(save_var_opt, [] { return Tensor(); });
  return native::xpu::batch_norm_backward_kernel(
      grad_output,
      input,
      weight,
      running_mean,
      running_var,
      save_mean,
      save_var,
      update,
      eps,
      grad_input_mask);
}

} // namespace at
