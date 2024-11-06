#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void foreach_copy_list_kernel_(TensorList self, TensorList src);

} // namespace at::native::xpu
