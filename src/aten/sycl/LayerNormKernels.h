#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

std::tuple<Tensor, Tensor, Tensor> layer_norm_kernel(
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double epsilon);

std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const Tensor& mean,
    const Tensor& rstd,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    std::array<bool, 3> grad_input_mask);

} // namespace xpu
} // namespace native
} // namespace at
