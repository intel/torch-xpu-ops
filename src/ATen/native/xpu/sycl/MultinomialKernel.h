#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API void multinomial_kernel(
    Tensor& result,
    const Tensor& self,
    const int64_t n_sample,
    std::optional<Generator> generator);

} // namespace at::native::xpu
