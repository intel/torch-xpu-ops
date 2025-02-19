#pragma once

#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

using namespace at::sparse;

TORCH_XPU_API Tensor softmax_sparse_xpu_kernel(
    const Tensor& input_,
    const int64_t dim_,
    const bool half_to_float);

TORCH_XPU_API Tensor log_softmax_sparse_xpu_kernel(
    const Tensor& input_,
    const int64_t dim_,
    const bool half_to_float);

TORCH_XPU_API Tensor softmax_backward_sparse_xpu_kernel(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    const Tensor& input_);

TORCH_XPU_API Tensor log_softmax_backward_sparse_xpu_kernel(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    const Tensor& input_);

} // namespace at::native::xpu
