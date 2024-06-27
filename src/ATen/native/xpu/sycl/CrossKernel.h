#pragma once
#include <ATen/ATen.h>
namespace at::native::xpu {
Tensor& linalg_cross_kernel(
    const Tensor& self,
    const Tensor& other,
    int64_t dim,
    Tensor& out);
} // namespace at::native::xpu