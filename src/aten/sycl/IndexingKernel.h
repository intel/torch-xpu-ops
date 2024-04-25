#pragma once
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void index_kernel(
    TensorIterator& iter,
    at::IntArrayRef index_size,
    at::IntArrayRef index_stride,
    at::IntArrayRef non_index_size,
    at::IntArrayRef non_index_stride);

void index_select_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& out);

void masked_fill_kernel(TensorIterator& iter, const Scalar& value);

void index_add_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha,
    const Tensor& out);

void index_put_kernel(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    IntArrayRef non_index_size,
    IntArrayRef non_index_stride,
    bool accumulate);

void index_put_deterministic_kernel(
    Tensor& self,
    const c10::List<c10::optional<Tensor>>& indices,
    const Tensor& value,
    bool accumulate,
    bool unsafe);

} // namespace at::native::xpu
