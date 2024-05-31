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

} // namespace xpu
} // namespace native
} // namespace at
