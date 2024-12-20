#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void foreach_lerp_scalarlist_kernel(
    TensorList tensors1,
    TensorList tensors2,
    at::ArrayRef<Scalar> scalars,
    TensorList result);

TORCH_XPU_API void foreach_lerp_scalarlist_kernel_(
    TensorList tensors1,
    TensorList tensors2,
    at::ArrayRef<Scalar> scalars);

} // namespace at::native::xpu
