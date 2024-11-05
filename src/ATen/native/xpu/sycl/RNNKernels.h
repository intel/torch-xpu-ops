#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API std::tuple<Tensor, Tensor> _thnn_fused_gru_cell_kernel(
    const Tensor& input_gates,
    const Tensor& hidden_gates,
    const Tensor& hx,
    const c10::optional<Tensor>& input_bias_opt,
    const c10::optional<Tensor>& hidden_bias_opt);

TORCH_XPU_API std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
_thnn_fused_gru_cell_backward_kernel(
    const Tensor& grad_hy,
    const Tensor& workspace,
    bool has_bias);

} // namespace at::native::xpu