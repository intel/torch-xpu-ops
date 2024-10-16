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

TORCH_XPU_API void cummax_kernel(
    const Tensor& self,
    Tensor& values,
    Tensor& indices,
    int64_t dim);

TORCH_XPU_API void cummin_kernel(
    const Tensor& self,
    Tensor& values,
    Tensor& indices,
    int64_t dim);

TORCH_XPU_API Tensor& logcumsumexp_kernel(
    const Tensor& self,
    int64_t dim,
    Tensor& result);

} // namespace at::native::xpu
