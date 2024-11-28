#pragma once

#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API Tensor quantized_max_pool2d_kernel(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode);

} // namespace at::native::xpu
