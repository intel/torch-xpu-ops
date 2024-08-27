#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

void mode_kernel(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices);

} // namespace at::native::xpu
