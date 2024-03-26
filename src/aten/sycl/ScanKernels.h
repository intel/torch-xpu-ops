#pragma once
#include <cstdint>

namespace at {
class TensorBase;

namespace native {
namespace xpu {

// NOTE: these functions require output tensors to be contiguous
void launch_cumsum_xpu_kernel(const TensorBase& result, const TensorBase& self, int64_t dim);

}}}  // namespace at::native::xpu
