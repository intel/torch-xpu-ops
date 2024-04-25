#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void adaptive_avg_pool2d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input);

} // namespace at::native::xpu