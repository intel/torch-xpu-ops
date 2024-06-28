#pragma once

#include <comm/xpu_aten.h>

namespace at::native::xpu {

void col2im_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride);

} // namespace at::native::xpu
