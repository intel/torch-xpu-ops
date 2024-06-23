#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

void foreach_lerp_scalar_kernel(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& weight,
    TensorList result);

void foreach_lerp_scalar_kernel_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& weight);

} // namespace at::native::xpu
