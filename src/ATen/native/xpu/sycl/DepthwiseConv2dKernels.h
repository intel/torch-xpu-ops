
#pragma once
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void conv_depthwise2d_forward_kernel(
    const Tensor& input,
    const Tensor& output,
    const Tensor& weight,
    const Tensor& bias,
    const int kW,
    const int kH,
    const int dW,
    const int dH,
    const int padW,
    const int padH,
    const int dilationW,
    const int dilationH);

TORCH_XPU_API void conv_depthwise2d_backward_kernel(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& grad_input,
    const Tensor& weight,
    const int kW,
    const int kH,
    const int dW,
    const int dH,
    const int padW,
    const int padH,
    const int dilationW,
    const int dilationH);

TORCH_XPU_API void conv_depthwise2d_grad_weight_kernel(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& grad_weight,
    const int kW,
    const int kH,
    const int dW,
    const int dH,
    const int padW,
    const int padH,
    const int dilationW,
    const int dilationH);

} // namespace at::native::xpu
