#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API std::tuple<Tensor, Tensor> dropout_kernel(
    const Tensor& self,
    double p,
    c10::optional<bool> train);

TORCH_XPU_API Tensor
dropout_backward_kernel(const Tensor& grad, const Tensor& mask, double scale);

} // namespace xpu
} // namespace native
} // namespace at
