#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API std::tuple<Tensor, Tensor, Tensor> _thnn_fused_lstm_cell_kernel(
    const Tensor& input_gates,
    const Tensor& hidden_gates,
    const Tensor& cx,
    const std::optional<Tensor>& input_bias_opt,
    const std::optional<Tensor>& hidden_bias_opt);

TORCH_XPU_API std::tuple<Tensor, Tensor, Tensor>
_thnn_fused_lstm_cell_backward_kernel(
    const std::optional<Tensor>& grad_hy_opt,
    const std::optional<Tensor>& grad_cy_opt,
    const Tensor& cx,
    const Tensor& cy,
    const Tensor& workspace,
    bool has_bias);

TORCH_XPU_API std::tuple<Tensor, Tensor> _thnn_fused_gru_cell_kernel(
    const Tensor& input_gates,
    const Tensor& hidden_gates,
    const Tensor& hx,
    const std::optional<Tensor>& input_bias_opt,
    const std::optional<Tensor>& hidden_bias_opt);

TORCH_XPU_API std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
_thnn_fused_gru_cell_backward_kernel(
    const Tensor& grad_hy,
    const Tensor& workspace,
    bool has_bias);

} // namespace at::native::xpu
