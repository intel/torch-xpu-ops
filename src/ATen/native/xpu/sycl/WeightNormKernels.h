#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API std::tuple<Tensor, Tensor> weight_norm_kernel(
    const Tensor& v,
    const Tensor& g,
    int64_t dim);

TORCH_XPU_API std::tuple<Tensor, Tensor> weight_norm_backward_kernel(
    const Tensor& grad_w,
    const Tensor& saved_v,
    const Tensor& saved_g,
    const Tensor& saved_norms,
    int64_t dim);

} // namespace at::native::xpu
