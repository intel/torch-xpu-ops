#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API std::vector<Tensor> foreach_norm_kernel(
    TensorList tensors,
    const Scalar& ord,
    double p,
    c10::optional<ScalarType> dtype);

TORCH_XPU_API std::vector<Tensor> foreach_max_kernel(TensorList tensors);

} // namespace at::native::xpu
