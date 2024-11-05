#include <ATen/ATen.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <ATen/native/xpu/sycl/RNNKernels.h>

namespace at::native {

std::tuple<at::Tensor, at::Tensor> _thnn_fused_gru_cell_xpu(
    const Tensor& input_gates,
    const Tensor& hidden_gates,
    const Tensor& hx,
    const std::optional<at::Tensor>& input_bias = {},
    const std::optional<at::Tensor>& hidden_bias = {}) {
  return native::xpu::_thnn_fused_gru_cell_kernel(
      input_gates, hidden_gates, hx, input_bias, input_bias);
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