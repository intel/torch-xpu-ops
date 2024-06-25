#pragma once
#include <cstdint>

namespace at::native::xpu {

void cumsum_kernel(const Tensor& result, const Tensor& self, int64_t dim);

} // namespace at::native::xpu
