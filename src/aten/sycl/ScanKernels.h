#pragma once
#include <cstdint>

namespace at {
namespace native {
namespace xpu {

// NOTE: these functions require output tensors to be contiguous
void launch_cumsum_xpu_kernel(
    const Tensor& out,
    const Tensor& self,
    int64_t dim);

} // namespace xpu
} // namespace native
} // namespace at
