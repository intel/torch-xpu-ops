#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

std::tuple<Tensor, Tensor> batch_norm_stats_kernel(
    const Tensor& self,
    double epsilon);

void batch_norm_elemt_kernel(
    Tensor& out,
    const Tensor& self,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const Tensor& mean_,
    const Tensor& invstd_);

std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_backward_reduce_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& invstd,
    const c10::optional<Tensor>& weight_opt,
    bool input_g,
    bool weight_g,
    bool bias_g);

Tensor batch_norm_backward_elemt_kernel(
    const Tensor& self,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& invstd,
    const c10::optional<Tensor>& weight_opt,
    const Tensor& sum_dy,
    const Tensor& sum_dy_xmu,
    const Tensor& count);

std::tuple<Tensor, Tensor> batch_norm_update_stats_kernel(
    const Tensor& self,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    double momentum);

std::tuple<Tensor&, Tensor&, Tensor&> batch_norm_out_kernel(
    const Tensor& self,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    bool train,
    double momentum,
    double epsilon,
    Tensor& output,
    Tensor& save_mean,
    Tensor& save_invstd);

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_kernel(
    const Tensor& grad_out,
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    const c10::optional<Tensor>& save_mean_opt,
    const c10::optional<Tensor>& save_invstd_opt,
    bool train,
    double epsilon,
    std::array<bool, 3> grad_input_mask);

} // namespace xpu
} // namespace native
} // namespace at
