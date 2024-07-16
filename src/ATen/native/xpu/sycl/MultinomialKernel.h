#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

void multinomial_kernel(
    Tensor& result,
    const Tensor& self,
    const int64_t n_sample,
    c10::optional<Generator> generator);

} // namespace at::native::xpu