#pragma once
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {
Tensor bincount_kernel(
    const Tensor& self,
    const c10::optional<Tensor>& weights,
    int64_t minlength);
} // namespace at::native::xpu