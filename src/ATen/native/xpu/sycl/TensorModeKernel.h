#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API void mode_kernel(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim);

} // namespace at::native::xpu
