#include <ATen/ATen.h>
#include <ATen/native/xpu/sycl/RNNKernels.h>

namespace at::native {

std::tuple<Tensor, Tensor, Tensor> _thnn_fused_lstm_cell_xpu(
    const Tensor& input_gates,
    const Tensor& hidden_gates,
    const Tensor& cx,
    const std::optional<Tensor>& input_bias_opt,
    const std::optional<Tensor>& hidden_bias_opt) {
  return native::xpu::_thnn_fused_lstm_cell_kernel(
      input_gates, hidden_gates, cx, input_bias_opt, hidden_bias_opt);
}

std::tuple<Tensor, Tensor, Tensor> _thnn_fused_lstm_cell_backward_xpu(
    const std::optional<Tensor>& grad_hy_opt,
    const std::optional<Tensor>& grad_cy_opt,
    const Tensor& cx,
    const Tensor& cy,
    const Tensor& workspace,
    bool has_bias) {
  return native::xpu::_thnn_fused_lstm_cell_backward_kernel(
      grad_hy_opt, grad_cy_opt, cx, cy, workspace, has_bias);
}

std::tuple<at::Tensor, at::Tensor> _thnn_fused_gru_cell_xpu(
    const Tensor& input_gates,
    const Tensor& hidden_gates,
    const Tensor& hx,
    const std::optional<at::Tensor>& input_bias,
    const std::optional<at::Tensor>& hidden_bias) {
  return native::xpu::_thnn_fused_gru_cell_kernel(
      input_gates, hidden_gates, hx, input_bias, hidden_bias);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
_thnn_fused_gru_cell_backward_xpu(
    const Tensor& grad_hy,
    const Tensor& workspace,
    bool has_bias) {
  return native::xpu::_thnn_fused_gru_cell_backward_kernel(
      grad_hy, workspace, has_bias);
}

} // namespace at::native
