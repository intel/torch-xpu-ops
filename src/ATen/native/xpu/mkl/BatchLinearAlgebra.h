#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::xpu {

TORCH_XPU_API void svd_mkl(
    const Tensor& A,
    const bool full_matrices,
    const bool compute_uv,
    const c10::optional<c10::string_view>& driver,
    const Tensor& U,
    const Tensor& S,
    const Tensor& Vh);

} // namespace at::native::xpu
