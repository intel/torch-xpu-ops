#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::xpu {

TORCH_XPU_API void lu_solve_mkl(
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& B,
    TransposeType trans);

TORCH_XPU_API void lu_factor_mkl(
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& info,
    bool pivot);

TORCH_XPU_API void linalg_qr_kernel(
    const at::Tensor& A,
    std::string_view mode,
    const at::Tensor& Q,
    const at::Tensor& R);

} // namespace at::native::xpu
