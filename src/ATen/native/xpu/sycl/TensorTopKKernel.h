#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

void topk_kernel(
    const at::Tensor& input,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    const at::Tensor& values,
    const at::Tensor& indices);

} // namespace xpu
} // namespace native
} // namespace at
