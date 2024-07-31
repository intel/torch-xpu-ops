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

void launch_median_kernel(
    const TensorBase& vals,
    const TensorBase& inds,
    const TensorBase& self,
    int64_t dim,
    bool ignore_nan);

} // namespace at::native::xpu
