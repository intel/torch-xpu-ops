#pragma once

#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void unfold_backward_kernel(
    Tensor& grad_out,
    const Tensor& grad_in,
    int64_t dim,
    int64_t size,
    int64_t step);

} // namespace at::native::xpu
