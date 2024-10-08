#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void sort_stable_kernel(
    const TensorBase& self_base,
    const TensorBase& values_base,
    const TensorBase& indices_base,
    int64_t dim,
    bool descending,
    bool stable);

TORCH_XPU_API void launch_median_kernel(
    const TensorBase& vals,
    const TensorBase& inds,
    const TensorBase& self,
    int64_t dim,
    bool ignore_nan);

} // namespace at::native::xpu
