#pragma once
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

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

} // namespace at::native::xpu
