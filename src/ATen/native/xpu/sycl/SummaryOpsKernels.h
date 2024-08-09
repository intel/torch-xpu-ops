#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

Tensor bincount_kernel(
    const Tensor& self,
    const Tensor& weights,
    int64_t minlength);

Tensor _histc_kernel(
    const Tensor& self,
    int64_t nbins,
    const Scalar& min,
    const Scalar& max);

} // namespace at::native::xpu
