#pragma once

#include <ATen/ATen.h>

#include <aten/sycl/DilatedMaxPool2d.h>

namespace at {
namespace native {
namespace xpu {

Tensor& max_pool2d_with_indices_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& indices,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode);

}
} // namespace native
} // namespace at