#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/TensorOptions.h>

namespace at::native::xpu {

TORCH_XPU_API Tensor tril_indices_kernel(
    int64_t row,
    int64_t col,
    int64_t offset,
    const TensorOptions& options);

TORCH_XPU_API Tensor triu_indices_kernel(
    int64_t row,
    int64_t col,
    int64_t offset,
    const TensorOptions& options);

} // namespace at::native::xpu
