#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API Tensor
bincount_kernel(const Tensor& self, const Tensor& weights, int64_t minlength);

TORCH_XPU_API Tensor _histc_kernel(
    const Tensor& self,
    int64_t nbins,
    const Scalar& min,
    const Scalar& max);

} // namespace at::native::xpu
