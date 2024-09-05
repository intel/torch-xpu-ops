#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API void gather_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim,
    const Tensor& index);

TORCH_XPU_API void scatter_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src);

TORCH_XPU_API void scatter_fill_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& src);

TORCH_XPU_API void scatter_add_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src);

TORCH_XPU_API void scatter_reduce_kernel(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const ReductionType& reduce);

TORCH_XPU_API void scatter_reduce_two_kernel(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const ReductionType& reduce);

TORCH_XPU_API void scatter_scalar_reduce_kernel(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Scalar& value,
    const ReductionType& reduce);

} // namespace xpu
} // namespace native
} // namespace at
