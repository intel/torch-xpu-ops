#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API void layer_norm_kernel(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    double eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd);

TORCH_XPU_API std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_kernel(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor& dX,
    Tensor& dgamma,
    Tensor& dbeta,
    std::array<bool, 3> grad_input_mask);

} // namespace xpu
} // namespace native
} // namespace at
