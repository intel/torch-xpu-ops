#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {
namespace xpu {

void col2im_out_template(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride);

}
} // namespace native
} // namespace at