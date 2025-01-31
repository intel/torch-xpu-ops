#pragma once

#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void convert_indices_from_coo_to_csr_structured_kernel(
    const Tensor& input,
    const int64_t size,
    const bool out_int32,
    const Tensor& result);

TORCH_XPU_API void convert_indices_from_csr_to_coo_structured_kernel(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const bool out_int32,
    const bool transpose,
    const Tensor& result);
} // namespace at::native::xpu
