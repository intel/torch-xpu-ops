#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

void nonzero_kernel(const Tensor& self, Tensor& out);

} // namespace at::native::xpu
