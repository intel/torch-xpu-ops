#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {
void searchsorted_kernel(
    Tensor& result,
    const Tensor& input,
    const Tensor& boundaries,
    bool out_int32,
    bool right,
    const Tensor& sorter);
} // namespace at::native::xpu