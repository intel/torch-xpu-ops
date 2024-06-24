#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

void gather_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim,
    const Tensor& index);

void scatter_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src);

void scatter_fill_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& src);

void scatter_add_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src);

void scatter_reduce_kernel(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const ReductionType& reduce);

void scatter_reduce_two_kernel(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const ReductionType& reduce);

void scatter_scalar_reduce_kernel(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Scalar& value,
    const ReductionType& reduce);

} // namespace xpu
} // namespace native
} // namespace at
