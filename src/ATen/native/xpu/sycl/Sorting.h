#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

std::tuple<Tensor&, Tensor&> sort_stable_kernel(
    const Tensor& self,
    c10::optional<bool> stable,
    Tensor& values,
    Tensor& indices,
    int dim,
    bool descending);

} // namespace at::native::xpu
