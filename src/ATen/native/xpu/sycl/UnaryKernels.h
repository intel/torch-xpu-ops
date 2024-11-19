#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void sqrt_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void rsqrt_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void bitwise_not_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void exp_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void expm1_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void nan_to_num_kernel(
    TensorIteratorBase& iter,
    std::optional<double> nan,
    std::optional<double> pos_inf,
    std::optional<double> neg_inf);

TORCH_XPU_API void frexp_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
