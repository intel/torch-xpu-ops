#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API Tensor
bincount_kernel(const Tensor& self, const Tensor& weights, int64_t minlength);

} // namespace at::native::xpu
