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

TORCH_XPU_API void launch_cumsum_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim);

TORCH_XPU_API void launch_cumprod_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim);

TORCH_XPU_API void launch_cummax_kernel(
    const Tensor& self,
    const Tensor& values,
    const Tensor& indices,
    int64_t dim);

TORCH_XPU_API void launch_cummin_kernel(
    const Tensor& self,
    const Tensor& values,
    const Tensor& indices,
    int64_t dim);

TORCH_XPU_API void launch_logcumsumexp_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim);

} // namespace at::native::xpu
