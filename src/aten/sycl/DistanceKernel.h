#pragma once
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {
Tensor cdist_impl(
    const Tensor& x1,
    const Tensor& x2,
    const double p,
    c10::optional<int64_t> compute_mode);
} // namespace at::native::xpu