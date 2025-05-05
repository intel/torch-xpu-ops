#include <ATen/AccumulateType.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/xpu/sycl/BatchNormKernels.h>
#include <comm/xpu_aten.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor> batch_norm_stats_xpu(
    const Tensor& input,
    double eps) {
  return xpu::batch_norm_stats_kernel(input, eps);
}

Tensor batch_norm_elemt_xpu(
    const Tensor& input,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const Tensor& mean,
    const Tensor& invstd,
    double eps) {
  auto output = at::empty_like(input);
  xpu::batch_norm_elemt_kernel(output, input, weight, bias, mean, invstd);
  return output;
}

Tensor& batch_norm_elemt_xpu_out(
    const Tensor& input,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const Tensor& mean,
    const Tensor& invstd,
    double eps,
    Tensor& out) {
  xpu::batch_norm_elemt_kernel(out, input, weight, bias, mean, invstd);
  return out;
}

std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_backward_reduce_xpu(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& invstd,
    const std::optional<Tensor>& weight,
    bool input_g,
    bool weight_g,
    bool bias_g) {
  return xpu::batch_norm_backward_reduce_kernel(
      grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
}

Tensor batch_norm_backward_elemt_xpu(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& invstd,
    const std::optional<Tensor>& weight,
    const Tensor& sum_dy,
    const Tensor& sum_dy_xmu,
    const Tensor& count) {
  return xpu::batch_norm_backward_elemt_kernel(
      grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
}

std::tuple<Tensor, Tensor> batch_norm_update_stats_xpu(
    const Tensor& input,
    const std::optional<Tensor>& running_mean,
    const std::optional<Tensor>& running_var,
    double momentum) {
  return xpu::batch_norm_update_stats_kernel(
      input, running_mean, running_var, momentum);
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_xpu(
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

  xpu::batch_norm_kernel(
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

std::tuple<Tensor&, Tensor&, Tensor&> batch_norm_xpu_out(
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
  return xpu::batch_norm_kernel(
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

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_xpu(
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
  return xpu::batch_norm_backward_kernel(
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

std::tuple<Tensor, Tensor, Tensor> _batch_norm_legit_xpu(
    const Tensor& input,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    Tensor& running_mean,
    Tensor& running_var,
    bool training,
    double momentum,
    double eps) {
  return batch_norm_xpu(
      input, weight, bias, running_mean, running_var, training, momentum, eps);
}

std::tuple<Tensor&, Tensor&, Tensor&> _batch_norm_legit_xpu_out(
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
  return batch_norm_xpu_out(
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

std::tuple<Tensor, Tensor, Tensor> _batch_norm_legit_no_stats_xpu(
    const Tensor& input,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    bool training,
    double momentum,
    double eps) {
  return batch_norm_xpu(
      input, weight, bias, Tensor(), Tensor(), training, momentum, eps);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>
_batch_norm_legit_no_stats_xpu_out(
    const at::Tensor& input,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    bool training,
    double momentum,
    double eps,
    at::Tensor& out,
    at::Tensor& save_mean,
    at::Tensor& save_invstd) {
  return batch_norm_xpu_out(
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

std::tuple<Tensor, Tensor, Tensor, Tensor> _batch_norm_with_update_xpu(
    const Tensor& input,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    Tensor& running_mean,
    Tensor& running_var,
    double momentum,
    double eps) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& bias = bias_opt.value_or(Tensor());
  Tensor reserve;

  reserve = at::empty({0}, input.options().dtype(kByte));

  auto output = at::empty_like(input);
  int64_t n_input = input.size(1);
  auto options =
      input.options().dtype(at::toAccumulateType(input.scalar_type(), true));
  auto save_mean = at::empty({n_input}, options);
  auto save_invstd = at::empty({n_input}, options);

  xpu::batch_norm_kernel(
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

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> _batch_norm_with_update_xpu_out(
    const Tensor& input,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
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
  const Tensor& bias = bias_opt.value_or(Tensor());

  std::tie(out, save_mean, save_var) = xpu::batch_norm_kernel(
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

std::tuple<Tensor, Tensor, Tensor> _new_batch_norm_backward_xpu(
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
  const Tensor& running_mean = running_mean_opt.value_or(Tensor());
  const Tensor& running_var = running_var_opt.value_or(Tensor());
  const Tensor& save_mean = save_mean_opt.value_or(Tensor());
  const Tensor& save_var = save_var_opt.value_or(Tensor());
  return xpu::batch_norm_backward_kernel(
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

std::tuple<Tensor, Tensor> batch_norm_gather_stats_xpu(
    const Tensor& input,
    const Tensor& mean,
    const Tensor& invstd,
    const std::optional<Tensor>& running_mean,
    const std::optional<Tensor>& running_var,
    double momentum,
    double eps,
    int64_t count) {
  return xpu::batch_norm_gather_stats_kernel(
      input, mean, invstd, running_mean, running_var, momentum, eps, count);
}

std::tuple<Tensor, Tensor> batch_norm_gather_stats_with_counts_xpu(
    const Tensor& input,
    const Tensor& mean,
    const Tensor& invstd,
    const std::optional<Tensor>& running_mean,
    const std::optional<Tensor>& running_var,
    double momentum,
    double eps,
    const Tensor& counts) {
  return xpu::batch_norm_gather_stats_with_counts_kernel(
      input, mean, invstd, running_mean, running_var, momentum, eps, counts);
}

} // namespace native
} // namespace at
