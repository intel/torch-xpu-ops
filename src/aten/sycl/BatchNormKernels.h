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

std::tuple<Tensor, Tensor> batch_norm_gather_stats_kernel(
    const Tensor& self,
    const Tensor& mean,
    const Tensor& invstd,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    double momentum,
    double epsilon,
    int64_t count);

std::tuple<Tensor, Tensor> batch_norm_gather_stats_with_counts_kernel(
    const Tensor& self,
    const Tensor& mean,
    const Tensor& invstd,
    const c10::optional<Tensor>& running_mean_opt /* optional */,
    const c10::optional<Tensor>& running_var_opt /* optional */,
    double momentum,
    double epsilon,
    const Tensor& counts);

} // namespace xpu
} // namespace native
} // namespace at
