#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void argmax_kernel(TensorIterator& iter);

TORCH_XPU_API void argmin_kernel(TensorIterator& iter);

TORCH_XPU_API void and_kernel(TensorIterator& iter);

TORCH_XPU_API void or_kernel(TensorIterator& iter);

TORCH_XPU_API void mean_kernel(TensorIterator& iter);

TORCH_XPU_API void sum_kernel(TensorIterator& iter);

TORCH_XPU_API void prod_kernel(TensorIterator& iter);

TORCH_XPU_API void nansum_kernel(TensorIterator& iter);

TORCH_XPU_API void std_var_kernel(
    TensorIterator& iter,
    double correction,
    bool take_sqrt);

TORCH_XPU_API void aminmax_kernel(TensorIterator& iter);

TORCH_XPU_API void aminmax_allreduce_kernel(TensorIterator& iter);

} // namespace at::native::xpu
