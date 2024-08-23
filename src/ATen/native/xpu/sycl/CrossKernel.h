#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API void linalg_cross_kernel(
    const Tensor& result,
    const Tensor& x1,
    const Tensor& x2,
    int64_t dim);

} // namespace at::native::xpu
