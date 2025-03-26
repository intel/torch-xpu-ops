#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::xpu {

TORCH_XPU_API
std::tuple<Tensor&, Tensor&, Tensor&> svd_mkl(
    const Tensor& A,
    bool full_matrices,
    bool compute_uv,
    c10::optional<c10::string_view> driver,
    Tensor& U,
    Tensor& S,
    Tensor& Vh);

} // namespace at::native::xpu
