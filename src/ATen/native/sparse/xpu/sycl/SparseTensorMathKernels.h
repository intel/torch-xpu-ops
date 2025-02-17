#pragma once

#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

using namespace at::sparse;

TORCH_XPU_API SparseTensor& add_sparse_kernel(
    const SparseTensor& t,
    const SparseTensor& src,
    const Scalar& value,
    SparseTensor& r_);

TORCH_XPU_API SparseTensor& mul_sparse_kernel(
    const Tensor& t_,
    const Tensor& src_,
    SparseTensor& r_);

TORCH_XPU_API Tensor _sparse_sum_backward_kernel(
    const Tensor& grad_,
    const SparseTensor& input_,
    IntArrayRef dims_to_sum);

} // namespace at::native::xpu
