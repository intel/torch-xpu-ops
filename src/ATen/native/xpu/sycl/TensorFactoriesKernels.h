#pragma once

#include <ATen/ATen.h>
#include <c10/core/TensorOptions.h>

namespace at::native::xpu {

Tensor tril_indices_kernel(
    int64_t row,
    int64_t col,
    int64_t offset,
    const TensorOptions& options);

Tensor triu_indices_kernel(
    int64_t row,
    int64_t col,
    int64_t offset,
    const TensorOptions& options);

} // namespace at::native::xpu
