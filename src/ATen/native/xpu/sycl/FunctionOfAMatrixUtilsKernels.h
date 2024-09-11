#pragma once
#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void _compute_linear_combination_kernel(
    TensorIterator& iter,
    int64_t in_stride,
    int64_t coeff_stride,
    int64_t num_summations);

} // namespace at::native::xpu
