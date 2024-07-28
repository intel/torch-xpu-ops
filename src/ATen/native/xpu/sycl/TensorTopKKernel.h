#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

std::tuple<at::Tensor&, at::Tensor&> topk_kernel(
    const at::Tensor& input,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    at::Tensor& values,
    at::Tensor& indices);

} // namespace xpu
} // namespace native
} // namespace at
