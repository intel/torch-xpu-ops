#pragma once

#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

using namespace at::sparse;

TORCH_XPU_API SparseTensor coalesce_sparse_kernel(const SparseTensor& self);

TORCH_XPU_API Tensor
flatten_indices_kernel(const Tensor& indices, IntArrayRef size);

} // namespace at::native::xpu
