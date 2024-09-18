#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API void foreach_lerp_list_kernel(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3,
    TensorList result);

TORCH_XPU_API void foreach_lerp_list_kernel_(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3);

} // namespace at::native::xpu
