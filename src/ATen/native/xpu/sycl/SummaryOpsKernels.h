#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API Tensor
bincount_kernel(const Tensor& self, const Tensor& weights, int64_t minlength);

} // namespace at::native::xpu
