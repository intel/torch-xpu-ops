#pragma once
#include <cstdint>

namespace at::native::xpu {

TORCH_XPU_API void cumsum_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim);

TORCH_XPU_API void cumprod_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim);

} // namespace at::native::xpu
