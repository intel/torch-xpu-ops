#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

std::vector<Tensor> foreach_norm_kernel(
    TensorList tensors,
    const Scalar& ord,
    double p,
    c10::optional<ScalarType> dtype);

} // namespace at::native::xpu
