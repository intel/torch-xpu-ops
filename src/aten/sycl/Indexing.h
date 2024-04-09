#pragma once
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

Tensor& index_select_out_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Tensor& out);

void masked_fill_kernel(TensorIterator& iter, const Scalar& value);

} // namespace at::native::xpu
