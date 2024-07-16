#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {
std::tuple<Tensor, Tensor> _weight_norm_interface_kernel(
    const Tensor& v,
    const Tensor& g,
    int64_t dim);

std::tuple<Tensor, Tensor> _weight_norm_interface_backward_kernel(
    const Tensor& grad_w,
    const Tensor& saved_v,
    const Tensor& saved_g,
    const Tensor& saved_norms,
    int64_t dim);
} // namespace at::native::xpu