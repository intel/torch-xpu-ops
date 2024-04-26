#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

std::tuple<Tensor, Tensor> native_dropout_kernel(
    const Tensor& self,
    double p,
    c10::optional<bool> train);

Tensor native_dropout_backward_kernel(
    const Tensor& grad,
    const Tensor& mask,
    double scale);

} // namespace xpu
} // namespace native
} // namespace at
