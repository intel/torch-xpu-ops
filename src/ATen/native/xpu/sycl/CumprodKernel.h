#pragma once

#include <ATen/core/TensorBase.h>

namespace at::native::xpu {

TORCH_XPU_API void launch_cumprod_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim);

} // namespace at::native::xpu
