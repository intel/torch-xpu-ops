#include <ATen/ATen.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <ATen/native/xpu/sycl/GroupNormKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>

namespace at {

template <typename T>
void check_group_norm_inputs(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    T C,
    int64_t num_groups) {
  TORCH_CHECK(
      num_groups > 0,
      "Expected num groups to be greater than 0, got ",
      num_groups);
  TORCH_CHECK(
      C % num_groups == 0,
      "Expected number of channels in input to be divisible by ",
      "num_groups, but got input of shape ",
      input.sizes(),
      " and "
      "num_groups=",
      num_groups);
  TORCH_CHECK(
      !weight.defined() ||
          (weight.dim() == 1 && at::symint::numel<T>(weight) == C),
      "Expected weight to be a vector of size equal to the number of ",
      "channels in input, but got weight of shape ",
      weight.sizes(),
      " and input of shape ",
      input.sizes());
  TORCH_CHECK(
      !bias.defined() || (bias.dim() == 1 && at::symint::numel<T>(bias) == C),
      "Expected bias to be a vector of size equal to the number of ",
      "channels in input, but got bias of shape ",
      weight.sizes(),
      " and input of shape ",
      input.sizes());
}

std::tuple<Tensor, Tensor, Tensor> XPUNativeFunctions::native_group_norm(
    const Tensor& X,
    const std::optional<Tensor>& gamma_opt,
    const std::optional<Tensor>& beta_opt,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
  const Tensor& gamma = *gamma_maybe_owned;
  const Tensor& beta = c10::value_or_else(beta_opt, [] { return Tensor(); });

  // repeated check so expanded weights can call native_group_norm directly but
  // save mean and variance from forward
  check_group_norm_inputs(X, gamma, beta, C, group);
  auto memory_format = X.device().is_cpu() ? X.suggest_memory_format()
                                           : at::MemoryFormat::Contiguous;

  TORCH_CHECK(X.is_contiguous(memory_format));

  bool mixed_type = at::native::is_mixed_type(X, gamma, beta);
  if (mixed_type) {
    at::native::check_mixed_data_type(X, gamma, beta);
  }

  Tensor Y = at::native::empty_like(
      X,
      c10::nullopt /* dtype */,
      c10::nullopt /* layout */,
      c10::nullopt /* device */,
      c10::nullopt /* pin_memory */,
      memory_format);
  const auto dtype = at::native::param_scalar_type(X, mixed_type);
  Tensor mean = at::empty({N, group}, X.options().dtype(dtype));
  Tensor rstd = at::empty({N, group}, X.options().dtype(dtype));
  native::xpu::group_norm_kernel(
      X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd, dtype);
  return std::make_tuple(Y, mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> XPUNativeFunctions::
    native_group_norm_backward(
        const Tensor& dY,
        const Tensor& X,
        const Tensor& mean,
        const Tensor& rstd,
        const c10::optional<Tensor>& gamma_opt,
        int64_t N,
        int64_t C,
        int64_t HxW,
        int64_t group,
        std::array<bool, 3> grad_input_mask) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
  const Tensor& gamma = *gamma_maybe_owned;
  TORCH_CHECK(
      X.scalar_type() == dY.scalar_type(),
      "Expected scalar types of X and dY are same.");
  bool mixed_type = at::native::is_mixed_type(X, mean, rstd);
  if (mixed_type) {
    at::native::check_mixed_data_type(X, mean, rstd);
  }
  auto memory_format = X.device().is_cpu() ? X.suggest_memory_format()
                                           : at::MemoryFormat::Contiguous;

  Tensor dX;
  Tensor dgamma;
  Tensor dbeta;
  if (grad_input_mask[0]) {
    dX = at::native::empty_like(
        X,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        memory_format);
  }
  if (grad_input_mask[1]) {
    dgamma = at::native::empty_like(
        gamma,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        at::MemoryFormat::Contiguous);
  }
  if (grad_input_mask[2]) {
    dbeta = at::native::empty_like(
        gamma,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        at::MemoryFormat::Contiguous);
  }
  native::xpu::group_norm_backward_kernel(
      dY, X, mean, rstd, gamma, N, C, HxW, group, dX, dgamma, dbeta);
  return std::make_tuple(dX, dgamma, dbeta);
}

} // namespace at
