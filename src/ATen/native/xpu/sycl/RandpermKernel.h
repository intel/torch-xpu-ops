#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API Tensor
randperm_kernel(Tensor& result, int64_t n, c10::optional<Generator> generator);

} // namespace at::native::xpu
