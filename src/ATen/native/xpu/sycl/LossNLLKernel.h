#pragma once
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void nll_loss_forward_kernel(
    const Tensor& self,
    const Tensor& target,
    const OptionalTensorRef weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& output,
    const Tensor& total_weight);

TORCH_XPU_API void nll_loss_backward_kernel(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const OptionalTensorRef weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight,
    const Tensor& grad_input);

} // namespace at::native::xpu
