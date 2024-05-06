#pragma once
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

std::tuple<Tensor&, Tensor&> launch_nll_loss_forward_kernel(
    const Tensor& self,
    const Tensor& target,
    const OptionalTensorRef weight_opt,
    // const optional<Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    Tensor& output,
    Tensor& total_weight);

Tensor& launch_nll_loss_backward_kernel(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    // const optional<Tensor>& weight,
    const OptionalTensorRef weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight,
    Tensor& grad_input);

} // namespace at::native::xpu