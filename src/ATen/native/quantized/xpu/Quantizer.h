#pragma once

#include <ATen/ATen.h>
#include <ATen/quantized/Quantizer.h>

namespace at::native::xpu {

TORCH_XPU_API Tensor new_qtensor(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer);

} // namespace at::native::xpu
