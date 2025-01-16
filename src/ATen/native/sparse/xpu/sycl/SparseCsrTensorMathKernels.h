#pragma once

#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

using namespace at::sparse;

TORCH_XPU_API Tensor& convert_indices_from_coo_to_csr_xpu_kernel(
    const Tensor& t,
    const int64_t size,
    const bool out_int32,
    const Tensor& result);

} // namespace at::native::xpu
