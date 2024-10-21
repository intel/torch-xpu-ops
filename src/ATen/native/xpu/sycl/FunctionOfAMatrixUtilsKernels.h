#pragma once
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void _compute_linear_combination_kernel(
    TensorIterator& iter,
    int64_t in_stride,
    int64_t coeff_stride,
    int64_t num_summations);

} // namespace at::native::xpu
