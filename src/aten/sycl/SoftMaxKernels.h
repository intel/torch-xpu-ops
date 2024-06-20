#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

Tensor& _softmax_kernel(
    const Tensor& input,
    const int64_t dim,
    const bool half_to_float,
    Tensor& output);

Tensor& _log_softmax_kernel(
    const Tensor& input,
    const int64_t dim,
    const bool half_to_float,
    Tensor& output);

Tensor& _softmax_backward_kernel(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    bool half_to_float,
    Tensor& grad_input);

Tensor& _log_softmax_backward_kernel(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    bool half_to_float,
    Tensor& grad_input);

} // namespace xpu
} // namespace native
} // namespace at
