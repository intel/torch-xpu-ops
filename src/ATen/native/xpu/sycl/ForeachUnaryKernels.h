#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API std::vector<Tensor> foreach_sqrt_kernel(TensorList tensors);

TORCH_XPU_API void foreach_sqrt_kernel_(TensorList tensors);

} // namespace at::native::xpu
