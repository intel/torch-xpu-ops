#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void glu_kernel(const Tensor& self, int64_t dim, Tensor& out);

void glu_backward_kernel(
    const Tensor& grad_output,
    const Tensor& self,
    int64_t dim,
    Tensor& grad_input);

} // namespace at::native::xpu
