#pragma once

#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

using namespace at::sparse;
using OptTensor = std::optional<Tensor>;

TORCH_XPU_API void mul_sparse_sparse_kernel(
    Tensor& result,
    const Tensor& x,
    const Tensor& y);

TORCH_XPU_API void sparse_mask_intersection_kernel(
    Tensor& result,
    const Tensor& x,
    const Tensor& y,
    const OptTensor& x_hash_opt = std::nullopt);

TORCH_XPU_API void sparse_mask_projection_kernel(
    Tensor& result,
    const Tensor& x,
    const Tensor& y,
    const OptTensor& x_hash_opt,
    bool accumulate_matches);

} // namespace at::native::xpu
