#pragma once

#include <ATen/TensorIterator.h>
#include <ATen/native/SparseTensorUtils.h>

using namespace at::sparse;
namespace at::native::xpu {

TORCH_XPU_API void add_out_dense_sparse_kernel(
    Tensor& r_,
    const Tensor& dense,
    const SparseTensor& sparse,
    const Scalar& value);

TORCH_XPU_API void add_out_sparse_kernel(
    const SparseTensor& t,
    const SparseTensor& src,
    const Scalar& value,
    SparseTensor& r_);
} // namespace at::native::xpu
