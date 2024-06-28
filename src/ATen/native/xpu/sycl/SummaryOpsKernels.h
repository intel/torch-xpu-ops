#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

Tensor bincount_kernel(
    const Tensor& self,
    const Tensor& weights,
    int64_t minlength);

} // namespace at::native::xpu
